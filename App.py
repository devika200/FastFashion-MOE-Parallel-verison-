import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ================================================================
# Expert module
# ================================================================
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ================================================================
# Worker function that lives in each process
# ================================================================
def expert_worker(pipe, expert):
    torch.set_num_threads(1)
    while True:
        msg = pipe.recv()
        if msg == "STOP":
            break
        x = msg
        with torch.no_grad():
            y = expert(x)
        pipe.send(y)


# ================================================================
# MoE Model using persistent multiprocessing
# ================================================================
class ParallelMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(input_dim, num_experts)

        self.manager_pipes = []
        self.worker_pipes = []
        self.processes = []

        for _ in range(num_experts):
            parent_conn, child_conn = mp.Pipe()
            expert = Expert(input_dim, hidden_dim, output_dim)
            p = mp.Process(target=expert_worker, args=(child_conn, expert))
            p.start()
            self.manager_pipes.append(parent_conn)
            self.worker_pipes.append(child_conn)
            self.processes.append(p)

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=1)
        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=1)

        outputs = []
        for i in range(batch_size):
            experts_to_use = topk_idx[i]
            combined = 0

            for j, exp_id in enumerate(experts_to_use):
                self.manager_pipes[exp_id.item()].send(x_flat[i].unsqueeze(0))

            for j, exp_id in enumerate(experts_to_use):
                y = self.manager_pipes[exp_id.item()].recv()
                combined += topk_vals[i, j] * y

            outputs.append(combined)

        return torch.cat(outputs, dim=0)

    def close(self):
        for pipe in self.manager_pipes:
            pipe.send("STOP")
        for p in self.processes:
            p.join()


# ================================================================
# MAIN ENTRY POINT (required on Windows)
# ================================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Windows requires spawn

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root="./Fashion_MNIST", train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root="./Fashion_MNIST", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    input_dim = 28 * 28
    hidden_dim = 256
    output_dim = 10
    num_experts = 4
    top_k = 2

    model = ParallelMoE(input_dim, hidden_dim, output_dim, num_experts, top_k)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.router.parameters(), lr=1e-3)

    for epoch in range(1, 3):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} average loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", ncols=100):
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    print(f"✅ Test Accuracy: {100.0 * correct / total:.2f}%")
    model.close()

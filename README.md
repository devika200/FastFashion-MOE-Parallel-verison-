# 🧠 Parallel Mixture of Experts (MoE) — Experimental PyTorch Implementation using          
#'torch.multiprocessing'

This project implements an **experimental parallel version** of the **Mixture of Experts (MoE)** architecture in PyTorch using **`torch.multiprocessing`**.  
It explores how to run multiple neural network “experts” across different CPU processes in parallel and combine their outputs through a learned gating (router) network.

---

## 🚀 Project Overview

In a traditional MoE model, a **gating network** determines which subset of experts (small neural networks) should process a given input.  
This helps distribute the computational load and allows each expert to specialize.

In this implementation:
- Each expert runs in a **separate process** (one per CPU core) using **multiprocessing**.  
- The **main process (router)** sends inputs to selected experts through **pipes**, receives their outputs, and combines them using weighted probabilities.  
- The router is trained to learn which experts to route each input to.

This project serves as a **proof-of-concept** for parallel computation in MoE models using only CPU multiprocessing.

---

## ⚙️ How It Works

### 1. **Routing and Expert Selection**
- The input image is flattened and passed through a **router (linear layer)**.
- The router outputs a probability for each expert.
- The top-𝑘 experts (highest probabilities) are selected for that input.

### 2. **Parallel Execution**
- Each expert lives inside its own process.
- Communication happens through **bidirectional pipes**:
  - The main process sends the input tensor.
  - The worker process computes the expert’s forward pass.
  - The result is sent back to the main process.

### 3. **Output Combination**
- The main process receives outputs from all selected experts.
- The outputs are **weighted by their gating probabilities** and summed to produce the final prediction.

---

## 🧩 Key Features

✅ Implements a fully functional **Mixture of Experts** model.  
✅ Demonstrates **parallel computation** using `torch.multiprocessing`.  
✅ Persistent worker processes — experts stay alive across forward passes.  
✅ Supports **top-k gating** for dynamic expert selection.  
✅ Uses **Fashion-MNIST** for testing and visualization.

---

## ⚠️ Limitations

❌ **No gradient flow to experts:**  
Due to Python’s multiprocessing and pickling behavior, autograd graphs cannot be shared between processes.  
Gradients can’t propagate through pipes, so **only the router is trained**, while experts remain static.

❌ **Slow training:**  
Multiprocessing adds heavy **overhead** from data serialization (pickling/unpickling) and inter-process communication.  
For small batches, it’s much slower than the sequential version.

❌ **Low accuracy:**  
Because experts don’t update, the model’s learning capacity is limited — accuracy stays low compared to a normal MoE.

---

## 💡 Better Alternatives

For a production-ready parallel MoE:
- Use **PyTorch Distributed RPC Framework** or **DistributedDataParallel (DDP)** for correct gradient synchronization.  
- Explore **DeepSpeed-MoE**, **GShard**, or **FastMoE**, which handle expert parallelism efficiently across GPUs.  
- Consider using **CUDA streams** for true parallelism if working on GPUs.

---

## 🧠 Learning Takeaways

This project is valuable as a **learning experiment** to understand:
- How multiprocessing works in PyTorch.  
- How model components communicate across processes.  
- Why distributed systems are essential for scaling MoE models correctly.

---

## 📦 Dataset
- **Fashion-MNIST** (automatically downloaded via `torchvision.datasets`)

---

## 🧰 Dependencies

```bash
pip install torch torchvision tqdm


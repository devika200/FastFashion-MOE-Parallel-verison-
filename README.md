# FastFashion-MOE-Parallel-verison-
Experimental Mixture of Experts (MoE) model using PyTorch multiprocessing. Each expert runs in a separate process and communicates with the main model via pipes. Parallelism is achieved, but gradients can’t flow across processes, so only the router trains. Accuracy is low due to frozen experts and high inter-process overhead.

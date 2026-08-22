# 07_TRAINING_SYSTEMS — Technical Reference

## 1. Role Relevance
For an ML Engineer (LLM & Agentic Systems), training systems knowledge separates a researcher from an engineer. To fine-tune large models, you must know how to fit massive parameter counts and optimizer states onto GPUs without Out of Memory (OOM) errors, utilizing distributed paradigms like FSDP, Pipeline Parallelism, and Tensor Parallelism.

## 2. Prerequisites
- Backpropagation and AdamW mechanics.
- Deep Learning memory footprint (Params, Grads, States).
- Network interconnects (NVLink, PCIe, InfiniBand).

## 3. First Principles
A GPU has limited VRAM (e.g., 80GB for A100/H100). Training a 70B model requires hundreds of GBs just for the weights and optimizer states. Therefore, training must be distributed across multiple GPUs. The goal of a training system is to partition the memory and computation across GPUs while minimizing communication overhead.

## 4. Mechanistic Breakdown
### The Memory Equation
For mixed-precision training (using Adam), the memory per parameter is:
- Model Weights (FP16/BF16): 2 bytes
- Gradients (FP16/BF16): 2 bytes
- Optimizer States (Adam $m_t, v_t$ in FP32): 8 bytes
- Master Weights (FP32, used for optimizer updates): 4 bytes
Total = 16 bytes per parameter.
A 70B model requires $70 \times 10^9 \times 16$ bytes $\approx 1.12$ TB of VRAM just to exist, not including activations.

### Activation Checkpointing (Gradient Checkpointing)
Activations generated during the forward pass must be saved for the backward pass, consuming massive memory (scales with batch size $\times$ sequence length). Activation Checkpointing trades compute for memory: it drops intermediate activations and recomputes them during the backward pass.

## 5. Distributed Paradigms

### Distributed Data Parallel (DDP)
The model is replicated on every GPU. Each GPU gets a different micro-batch of data. Gradients are synchronized (All-Reduce) across GPUs before the optimizer step.
*Limitation*: Fails if the model + states cannot fit on a single GPU.

### Fully Sharded Data Parallel (FSDP / ZeRO-3)
Shards the model states across all GPUs.
1. **ZeRO-1**: Shards Optimizer States.
2. **ZeRO-2**: Shards Gradients + Optimizer States.
3. **ZeRO-3 (FSDP)**: Shards Parameters + Gradients + Optimizer States.
During the forward pass, a GPU requests the parameters it needs from other GPUs (All-Gather), computes, and then discards them.

### Tensor Parallelism (TP)
Splits individual matrix multiplications across multiple GPUs.
For $Y = XW$, we split $W$ into $W_1, W_2$. GPU 1 computes $XW_1$, GPU 2 computes $XW_2$. They communicate the results (All-Reduce or All-Gather). Requires extremely fast interconnects (NVLink).

### Pipeline Parallelism (PP)
Splits the layers of the model across GPUs. GPU 1 has layers 1-10, GPU 2 has 11-20.
*Limitation*: Causes a "pipeline bubble" where GPUs wait for the forward/backward pass of previous GPUs. Solved using micro-batch interleaving (e.g., 1F1B schedule).

## 6. Implementation
**PyTorch FSDP Setup:**
```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Model is partitioned automatically across devices
model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16)
)
```

## 7. Computational Complexity
- **Communication Volume (FSDP)**: Requires $1.5\times$ the parameter count in communication (All-Gather for forward, All-Gather for backward, Reduce-Scatter for gradients) per layer per batch.
- **Compute Volume**: Matrix multiplications dominate, but communication often bottlenecks the system if network bandwidth is low.

## 8. Hardware / GPU Behavior
- **NVLink**: 900 GB/s bandwidth between GPUs on the *same node*. Used for Tensor Parallelism.
- **InfiniBand / RoCE**: 400 Gbps (50 GB/s) between GPUs across *different nodes*. Used for Data Parallelism and Pipeline Parallelism.

## 9. Production Architecture
**3D Parallelism (Megatron-Turing)**:
To train massive models (e.g., 400B+), we combine all three:
1. TP within a node (fast NVLink).
2. PP across nodes (reduces communication volume compared to TP).
3. DP across the rest of the cluster (scales batch size).

## 10. Scalability & Bottlenecks
- **Straggler Problem**: If one GPU out of 1,024 is running 10% slower due to thermal throttling, the entire cluster runs 10% slower because of synchronous All-Reduce operations.

## 11. Failure Modes
- **Hardware Failures**: On large clusters, a GPU fails every few hours. Checkpointing frequency must be mathematically optimized based on MTBF (Mean Time Between Failures).
- **Loss Divergence**: Sudden spikes in loss due to FP16 overflow. Solved by using BF16 (Bfloat16).

## 12. Debugging
- **OOM during Training**: If using FSDP, enable activation checkpointing. If still OOM, reduce micro-batch size. If still OOM, apply CPU offloading (sends optimizer states to CPU RAM, severely penalizing speed).
- **Low MFU (Model FLOPs Utilization)**: If MFU is < 30%, the system is communication-bound. You must profile the NCCL communication rings, increase batch size, or optimize the parallelism strategy (e.g., switch from ZeRO-3 to TP/PP).

## 13. Trade-offs
- **FSDP vs 3D Parallelism**: FSDP is incredibly easy to set up natively in PyTorch and works well up to ~70B parameters on fast networks. 3D Parallelism is complex (Megatron-LM) but mandatory for >100B parameter models to avoid massive cross-node All-Gather overheads.

## 14. Principal-Level Reasoning
"When building a fine-tuning platform for production, I would standardize on PyTorch FSDP (ZeRO-3) with BF16 mixed precision for all models up to 70B parameters. This allows any engineer to submit an SFT job without reasoning about Tensor Parallelism, while still utilizing cross-node training. I would heavily monitor NCCL bandwidth; if we are deployed on AWS without EFA/InfiniBand, FSDP will stall, and I would fall back to ZeRO-2 with gradient checkpointing."

## 15. Interview Interrogation
- *Level 2*: What is the difference between FP16 and BF16?
- *Level 4*: Why does Adam optimizer require $3\times$ the memory of the model?
- *Level 7*: Walk me through exactly what happens during the backward pass of FSDP.
- *Level 9*: Your 64-GPU training run has 20% MFU. Where do you look first?
- *Level 10*: Architect the training system for continuous post-training on streaming human-feedback data without blocking the main serving cluster.

# 07_TRAINING_SYSTEMS — Mathematical & Systems Engineering Reference

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide an exhaustive mathematical and systems reference on Distributed LLM Training — covering training memory breakdowns, ZeRO-1/2/3 (FSDP-2) sharding proofs, Megatron 3D Parallelism (TP, PP, DP), Context Parallelism (Ring Attention vs Ulysses), and Pipeline Bubble calculus.

---

## 1. Memory Anatomy of Large Language Model Training

Let a Transformer model contain $\Phi$ parameters trained with mixed precision (BF16/FP16 forward/backward, FP32 AdamW optimizer).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│             Static Training Memory Footprint (16 Bytes per Parameter)       │
├───────────────────────────────┬───────────────────────────────┬─────────────┤
│ Model Parameters (FP16/BF16)  │ Gradients (FP16/BF16)         │ Adam States │
│ 2 Bytes × Φ                   │ 2 Bytes × Φ                   │ 12 Bytes × Φ│
└───────────────────────────────┴───────────────────────────────┴─────────────┘
                                                                       │
                         ┌─────────────────────────────────────────────┤
                         ▼                                             ▼
          ┌─────────────────────────────┐               ┌─────────────────────────────┐
          │ FP32 Master Weights (4B × Φ)│               │ FP32 First Moment m (4B × Φ)│
          ├─────────────────────────────┤               ├─────────────────────────────┤
          │ FP32 Second Moment v(4B × Φ)│               │ Total Adam = 12 Bytes × Φ   │
          └─────────────────────────────┘               └─────────────────────────────┘
```

### 1.1 Exact Memory Formulations

1. **Parameters ($\Phi$)**: $2 \Phi \text{ Bytes}$ (in FP16/BF16).
2. **Gradients ($G$)**: $2 \Phi \text{ Bytes}$ (in FP16/BF16).
3. **AdamW Optimizer States ($O$)**:
   - FP32 Master Parameters: $4 \Phi \text{ Bytes}$ (prevents underflow during tiny updates).
   - FP32 First Moment ($m_t$): $4 \Phi \text{ Bytes}$.
   - FP32 Second Moment ($v_t$): $4 \Phi \text{ Bytes}$.
   - **Total Optimizer State**: $\mathbf{12 \Phi \text{ Bytes}}$.

$$\mathbf{\text{Total Static Memory} = 2\Phi + 2\Phi + 12\Phi = 16\Phi \quad [\text{Bytes}]}$$

#### Concrete Sizing Example (LLaMA-3 70B):
$$ \Phi = 70 \times 10^9 \implies \text{Static Memory} = 16 \times (70 \times 10^9) = \mathbf{1,120 \text{ GB (1.12 TB)}} $$
*Conclusion*: Fitting a 70B model's static memory requires at least **14 NVIDIA H100 (80GB) GPUs** *before* allocating a single byte for activation memory!

---

### 1.2 Activation Memory & Gradient Checkpointing (Activation Recomputation)

Activation memory scales with sequence length $S$, batch size $B$, layers $L$, hidden dimension $d$, and attention heads $N_h$.

For a standard Transformer layer without checkpointing:
$$ \text{Memory}_{\text{act}} \approx B \cdot S \cdot d \cdot L \cdot \left( 34 + 5 \frac{N_h \cdot S}{d} \right) \quad [\text{Bytes}] $$

#### Activation Recomputation (Gradient Checkpointing):
- **Full Checkpointing**: Discards all intermediate activations during the forward pass, retaining only layer input activations. During the backward pass, recomputes activations on-the-fly.
  - *Memory Savings*: Reduces activation memory from $O(L)$ to $O(1)$ per layer.
  - *Compute Overhead*: Adds exactly **1 extra forward pass per layer** ($\sim 33\%$ additional total FLOPs).
- **Selective Checkpointing**: Discards only memory-intensive, low-compute activations (e.g. Attention Softmax and Dropout) while keeping GEMM activations, eliminating $70\%$ of activation memory with $< 5\%$ FLOP overhead.

---

## 2. ZeRO Optimization & PyTorch FSDP-2

The **Zero Redundancy Optimizer (ZeRO)** (Rajbhandari et al., 2020) eliminates memory redundancy across data-parallel GPUs ($N_d$ devices) without changing the mathematical model outputs.

```
Standard DDP:  Each GPU holds: [ Parameters (2Φ) | Gradients (2Φ) | Adam States (12Φ) ] (16Φ on EVERY GPU!)
ZeRO-1:        Each GPU holds: [ Parameters (2Φ) | Gradients (2Φ) | Adam States (12Φ / N_d) ]
ZeRO-2:        Each GPU holds: [ Parameters (2Φ) | Gradients (2Φ / N_d) | Adam States (12Φ / N_d) ]
ZeRO-3 (FSDP): Each GPU holds: [ Parameters (2Φ / N_d) | Gradients (2Φ / N_d) | Adam States (12Φ / N_d) ]
```

### 2.1 Stage-by-Stage Memory & Communication Proofs

| ZeRO Stage | Parameter Memory | Gradient Memory | Optimizer State Memory | Total Static Memory per GPU | Communication Volume per Step |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline DDP** | $2\Phi$ | $2\Phi$ | $12\Phi$ | $16\Phi$ | $2\Phi$ (`All-Reduce` gradients) |
| **ZeRO-1 ($P_{os}$)** | $2\Phi$ | $2\Phi$ | $\frac{12\Phi}{N_d}$ | $4\Phi + \frac{12\Phi}{N_d}$ | $2\Phi$ (`All-Reduce` gradients) |
| **ZeRO-2 ($P_{os+g}$)** | $2\Phi$ | $\frac{2\Phi}{N_d}$ | $\frac{12\Phi}{N_d}$ | $2\Phi + \frac{14\Phi}{N_d}$ | $2\Phi$ (`Reduce-Scatter` gradients) |
| **ZeRO-3 / FSDP-2 ($P_{os+g+p}$)** | $\frac{2\Phi}{N_d}$ | $\frac{2\Phi}{N_d}$ | $\frac{12\Phi}{N_d}$ | $\mathbf{\frac{16\Phi}{N_d}}$ | $\mathbf{3\Phi}$ ($2\times$ `All-Gather` + $1\times$ `Reduce-Scatter`) |

#### Detailed Communication Walkthrough of ZeRO-3 / FSDP-2:
1. **Forward Pass**: Before layer $l$ executes, perform `All-Gather` across $N_d$ GPUs to reconstruct full weights $W_l$ ($1\Phi$ total volume). After execution, immediately discard $W_l$.
2. **Backward Pass**: Before backpropagating through layer $l$, perform `All-Gather` again to reconstruct $W_l$ ($1\Phi$ total volume). Discard $W_l$ immediately.
3. **Gradient Reduction**: Compute local gradients $g_l$, perform `Reduce-Scatter` across $N_d$ GPUs so each GPU retains only its $\frac{1}{N_d}$ gradient slice ($1\Phi$ total volume).
4. **Total Comm Volume**: $1\Phi + 1\Phi + 1\Phi = \mathbf{3\Phi \text{ bytes per step}}$ (only $1.5\times$ baseline DDP).

---

## 3. Megatron 3D Parallelism: TP, PP, and DP

When training models exceeding 100B parameters across thousands of GPUs, combining **Tensor Parallelism (TP)**, **Pipeline Parallelism (PP)**, and **Data Parallelism (DP)** is mandatory.

```
Total GPUs (N) = TP × PP × DP
```

### 3.1 Tensor Parallelism (Megatron-LM Intra-Node Splitting)

Tensor Parallelism splits individual weight matrices across $N_{\text{TP}}$ GPUs (typically within the same 8-GPU NVLink node).

```
MLP Layer Splitting in Megatron-LM:
Input X ──► [ Column Parallel Linear (W_1) ] ──► Split Activations [Y_1, Y_2]
                     │ (GeLU Activation applied locally on each GPU)
                     ▼
            [ Row Parallel Linear (W_2) ]    ──► Local Outputs [Z_1, Z_2]
                     │
                     ▼
           ┌────────────────────────┐
           │ All-Reduce Sum Barrier │ ──► Final Output Z = Z_1 + Z_2 (Exact Result!)
           └────────────────────────┘
```

1. **Column Parallel Linear ($W_1 \in \mathbb{R}^{d \times 4d}$)**:
   Splits $W_1$ column-wise into $[W_{1, 1}, \dots, W_{1, N_{\text{TP}}}]$.  
   Each GPU computes $Y_i = X W_{1, i}$ locally with zero communication.
2. **Row Parallel Linear ($W_2 \in \mathbb{R}^{4d \times d}$)**:
   Splits $W_2$ row-wise into $\begin{bmatrix} W_{2, 1} \\ \vdots \\ W_{2, N_{\text{TP}}} \end{bmatrix}$.  
   Each GPU computes $Z_i = \text{GeLU}(Y_i) W_{2, i}$.
3. **The Communication Step**:
   The full output is $Z = \sum_{i=1}^{N_{\text{TP}}} Z_i$. Requires an **`All-Reduce` sum** across the $N_{\text{TP}}$ GPUs.
   - Total communication per Transformer block = **2 `All-Reduce` in forward pass, 2 `All-Reduce` in backward pass** (one for Attention, one for MLP).

---

### 3.2 Pipeline Parallelism & The 1F1B Schedule Bubble

Pipeline Parallelism partitions the $L$ layers of a Transformer across $p$ pipeline stages (nodes).

To minimize idle time, a batch of size $B$ is split into $m$ **micro-batches** ($m \gg p$).

```
1F1B Schedule (p = 4 stages, m = 8 microbatches):
Stage 3: ───[F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][F7][B4][B5][B6][B7]───
Stage 2: ──[F0][F1][F2][B0][F3][B1][F4][B2][F5][B3][F6][B4][F7][B5][B6][B7]────
Stage 1: ─[F0][F1][B0][F2][B1][F3][B2][F4][B3][F5][B4][F6][B5][F7][B6][B7]─────
Stage 0: [F0][B0][F1][B1][F2][B2][F3][B3][F4][B4][F5][B5][F6][B6][F7][B7]──────
         ◄────────────────────────────── Total Time ────────────────────────────►
```

#### Exact Pipeline Bubble Fraction Derivation:
In 1F1B (One-Forward-One-Backward) steady state, the pipeline bubble consists of $p-1$ warmup forward steps and $p-1$ cooldown backward steps.  
Let $t_f$ and $t_b$ be the execution time of one micro-batch forward and backward pass ($t_b \approx 2 t_f$).  
The total ideal execution time is $m (t_f + t_b)$. The idle bubble time is $(p - 1)(t_f + t_b)$.

$$ \mathbf{\text{Bubble Fraction } F_{\text{bubble}} = \frac{(p - 1)(t_f + t_b)}{m(t_f + t_b) + (p - 1)(t_f + t_b)} = \frac{p - 1}{m + p - 1}} $$

*Sizing Rule*: To keep bubble overhead $< 10\%$, set $m \geq 9(p - 1)$.

---

### 3.3 Context Parallelism: Ring Attention vs. DeepSpeed Ulysses

When training with context lengths $S \geq 128\text{k}$ tokens, single GPU VRAM cannot fit even one sequence's attention matrices.

1. **Ring Attention (Liu et al., 2023)**:
   - Partitions sequence $S$ across $N_{\text{cp}}$ GPUs arranged in a logical ring.
   - Overlaps computation of local block Attention with asynchronous P2P sending of Key/Value blocks around the ring.
   - *Communication Volume*: $2 \left(\frac{N_{\text{cp}} - 1}{N_{\text{cp}}}\right) S \cdot d \cdot b_{\text{kv}}$. Perfectly memory-bounded and scalable to million-token contexts.
2. **DeepSpeed Ulysses (Jacobs et al., 2023)**:
   - Partitions sequence dimension across GPUs before attention.
   - Performs an `All-to-All` collective to transpose from Sequence-Partitioned to Head-Partitioned ($N_h / N_{\text{cp}}$ heads per GPU).
   - Computes standard FlashAttention locally, then performs a second `All-to-All` to transpose back.
   - *Advantage*: Leverages standard FlashAttention kernels with zero changes, but requires high-bandwidth interconnects (InfiniBand/NVLink).

---

## 4. PyTorch FSDP-2 Configuration Reference

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def setup_fsdp_model(model: nn.Module, transformer_layer_cls) -> FSDP:
    """
    Production-grade FSDP-2 configuration with BF16 mixed precision,
    Transformer layer wrapping, and backward prefetching.
    """
    # 1. Mixed Precision Policy (BF16 computation, FP32 master reduction)
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )

    # 2. Layer Wrapping Policy: Shards each Transformer Block independently
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={transformer_layer_cls}
    )

    # 3. Wrap Model in FSDP (ZeRO-3 Full Shard)
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD, # ZeRO-3
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE, # Overlap All-Gather with backward GEMM
        cpu_offload=CPUOffload(offload_params=False), # False for high NVLink bandwidth clusters
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True # Prevents CPU memory bloat from in-flight All-Gathers
    )
    return fsdp_model
```

---

## 5. Deep Interview Interrogation Ladder

- **Level 1 (Concept)**: What are the three components of static training memory in AdamW mixed-precision training?
- **Level 3 (Derivation)**: Calculate why the static memory of a 70B parameter model is exactly 1,120 GB in FP16 training with AdamW.
- **Level 5 (FSDP Mechanics)**: Walk through the exact communication primitives (`All-Gather` vs `Reduce-Scatter`) executed during the forward and backward pass of FSDP (ZeRO-3), and prove that total communication volume is $3\Phi$.
- **Level 7 (Pipeline Parallelism)**: Derive the exact Pipeline Bubble Fraction $F_{\text{bubble}} = \frac{p - 1}{m + p - 1}$ for a 1F1B schedule with $p$ stages and $m$ micro-batches.
- **Level 9 (Long Context)**: Compare Ring Attention and DeepSpeed Ulysses. What are their communication trade-offs when scaling sequence length to 512k tokens?
- **Level 10 (Principal Engineering)**: You are provisioning an 8,192 H100 GPU cluster to pre-train a 400B MoE model. Design the exact 4D Parallelism strategy (TP $\times$ PP $\times$ EP $\times$ DP), select the micro-batch size $m$, and calculate the NCCL network bandwidth saturation over InfiniBand NDR400.

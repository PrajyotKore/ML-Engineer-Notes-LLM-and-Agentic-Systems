# 08_GPU_AND_PERFORMANCE — Mathematical & Bare-Metal Hardware Reference

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide an exhaustive mathematical and microarchitectural bridge from physical NVIDIA GPU hardware (H100/A100) to PyTorch kernel execution — covering the Roofline model, SRAM tiling proofs in FlashAttention-1/2/3, Tensor Core MMA instructions, and memory bandwidth optimization.

---

## 1. NVIDIA GPU Architecture & Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NVIDIA H100 GPU (Hopper)                           │
│  132 Streaming Multiprocessors (SMs) | 80GB HBM3 (3.35 TB/s Bandwidth)       │
│  Peak FP16 Tensor Core Compute: 989 TFLOPs (Dense) / 1,978 TFLOPs (Sparse)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
┌───────────────────────────┐                   ┌───────────────────────────┐
│     SM 0 (Streaming MP)   │                   │    SM 131 (Streaming MP)  │
│ ┌───────────────────────┐ │                   │ ┌───────────────────────┐ │
│ │ 256KB Register File   │ │                   │ │ 256KB Register File   │ │
│ ├───────────────────────┤ │                   │ ├───────────────────────┤ │
│ │ 228KB Shared Mem/SRAM │ │                   │ │ 228KB Shared Mem/SRAM │ │
│ ├───────────────────────┤ │                   │ ├───────────────────────┤ │
│ │ 4x 4th Gen Tensor Core│ │                   │ │ 4x 4th Gen Tensor Core│ │
│ ├───────────────────────┤ │                   │ ├───────────────────────┤ │
│ │ 4x Warp Schedulers    │ │                   │ │ 4x Warp Schedulers    │ │
│ └───────────────────────┘ │                   │ └───────────────────────┘ │
└───────────────────────────┘                   └───────────────────────────┘
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
                  ┌───────────────────────────────────────┐
                  │          50MB L2 Cache                │
                  └───────────────────┬───────────────────┘
                                      ▼
                  ┌───────────────────────────────────────┐
                  │       80GB HBM3 Memory (3.35 TB/s)    │
                  └───────────────────────────────────────┘
```

### 1.1 Memory Hierarchy Latency & Bandwidth (NVIDIA H100 SXM5)

| Memory Level | Location | Capacity | Latency | Bandwidth |
| :--- | :--- | :--- | :--- | :--- |
| **Registers** | On-Chip per SM | 256 KB per SM ($\sim 33 \text{ MB}$ total) | $\sim 1$ cycle ($0.5 \text{ ns}$) | $\sim 30 \text{ TB/s}$ aggregate |
| **Shared Memory (SRAM)** | On-Chip per SM | Up to 228 KB per SM ($\sim 30 \text{ MB}$ total) | $\sim 15-30$ cycles ($10 \text{ ns}$) | $\sim 15 \text{ TB/s}$ aggregate |
| **L2 Cache** | On-Chip global | 50 MB | $\sim 100-200$ cycles ($50 \text{ ns}$) | $\sim 5.5 \text{ TB/s}$ |
| **High Bandwidth Memory (HBM3)** | Off-Chip package | 80 GB | $\sim 400-800$ cycles ($200 \text{ ns}$) | **3.35 TB/s** |
| **Host RAM via PCIe Gen5 / NVLink** | Off-Board CPU | 512 GB - 2 TB | $\sim 5,000$ cycles ($2 \text{ }\mu\text{s}$) | $128 \text{ GB/s (PCIe)} / 900 \text{ GB/s (NVLink)}$ |

*The Golden Rule of GPU Programming*: Reading from HBM is **$200\times$ slower** than reading from Shared Memory (SRAM). Every unnecessary round-trip to HBM stalls the Streaming Multiprocessors.

---

## 2. The Roofline Model & Arithmetic Intensity

The **Roofline Model** determines whether a kernel is bounded by memory bandwidth or compute throughput.

### 2.1 Arithmetic Intensity (AI)
$$ \text{Arithmetic Intensity (AI)} = \frac{\text{Total Floating Point Operations (FLOPs)}}{\text{Total Memory Transferred from HBM (Bytes)}} \quad \left[ \frac{\text{FLOPs}}{\text{Byte}} \right] $$

- **Attainable Performance ($P$)**:
  $$ P = \min\left( P_{\text{peak}}, \; \text{AI} \times B_{\text{peak}} \right) \quad [\text{TFLOPs/s}] $$
  Where:
  - $P_{\text{peak}}$ is the theoretical peak compute capacity (e.g. $989 \text{ TFLOPs}$ for H100 FP16).
  - $B_{\text{peak}}$ is the peak memory bandwidth (e.g. $3.35 \text{ TB/s}$ for H100 HBM3).

- **The Critical Ridge Point ($I^*$)**:
  $$ I^* = \frac{P_{\text{peak}}}{B_{\text{peak}}} = \frac{989 \times 10^{12} \text{ FLOPs/s}}{3.35 \times 10^{12} \text{ Bytes/s}} \approx \mathbf{295.2 \text{ FLOPs/Byte}} $$

```
  Attainable Performance (TFLOPs/s)
  989 ┼                                     ┌─────────────────────── (Compute Bound Peak)
      │                                    /
      │                                   /
      │                                  /
      │   Memory Bound Region           /
      │   Slope = Memory Bandwidth     /
      │                               /
    0 ┼──────────────────────────────┼──────────────────────────────► Arithmetic Intensity (FLOPs/Byte)
                                  I* = 295.2 FLOPs/Byte (Ridge Point)
```

### 2.2 Operational Regimes in Large Language Models:
1. **Memory-Bound Regime ($\text{AI} < I^* = 295$)**:
   - **Autoregressive Decoding (Batch Size 1)**: For a 70B parameter model in FP16 ($140\text{ GB}$ weights), generating 1 token requires loading $140\text{ GB}$ to perform $2 \times 70\times 10^9 = 140\text{ GFLOPs}$.  
     $\text{AI} = \frac{140 \times 10^9 \text{ FLOPs}}{140 \times 10^9 \text{ Bytes}} = \mathbf{1.0 \text{ FLOP/Byte}} \ll 295.2$.  
     *Hardware Utilization*: The H100 achieves only $\frac{1 \times 3.35 \text{ TB/s}}{989 \text{ TFLOPs}} \approx \mathbf{0.34\%}$ of its peak compute capacity!
   - **Element-wise Kernels**: LayerNorm, RMSNorm, GeLU, Softmax, Residual addition ($\text{AI} \approx 0.25 - 2 \text{ FLOPs/Byte}$).
2. **Compute-Bound Regime ($\text{AI} \geq I^* = 295$)**:
   - **Prefill Phase (Large Prompts)**: Processing a prompt of sequence length $S = 4096$ with batch size $B=8$.  
     Matrix Multiply $X W$: $\text{FLOPs} = 2 \cdot B \cdot S \cdot d_{\text{in}} \cdot d_{\text{out}}$.  
     Memory read: $d_{\text{in}} \cdot d_{\text{out}} \times 2\text{ bytes}$.  
     $\text{AI} = \frac{2 \cdot B \cdot S \cdot d^2}{2 \cdot d^2} = B \cdot S = 8 \times 4096 = \mathbf{32,768 \text{ FLOPs/Byte}} \gg 295.2$.  
     *Hardware Utilization*: Tensor Cores operate at near 100% capacity ($> 80\%$ MFU).

---

## 3. FlashAttention Mechanics: Mathematical SRAM Tiling

### 3.1 The Standard Attention Memory Wall
Standard PyTorch Attention computes:
$$ S = Q K^T \in \mathbb{R}^{N \times N}, \quad P = \text{Softmax}(S) \in \mathbb{R}^{N \times N}, \quad O = P V \in \mathbb{R}^{N \times d} $$
- **HBM Round-Trips**: Writes intermediate $N \times N$ matrices $S$ and $P$ to off-chip HBM, then reads them back to compute $P V$.
- **HBM Memory Access Complexity**: $O(N d + N^2)$ bytes transferred. For sequence length $N = 64\text{k}$, $N^2 = 4.096 \times 10^9$ elements ($8.2\text{ GB}$ of VRAM per attention head just for attention scores!).

---

### 3.2 Online Softmax Normalization (Milakov & Gimelshtein / Rabe & Staats)

To compute attention in small SRAM blocks without materializing the $N \times N$ matrix in HBM, we must maintain running softmax normalizers across split chunks.

Let a vector $x \in \mathbb{R}^N$ be split into two blocks $x^{(1)}, x^{(2)} \in \mathbb{R}^{N/2}$.  
Let local maximums and normalizers be:
$$ m^{(1)} = \max_i x_i^{(1)}, \quad l^{(1)} = \sum_i e^{x_i^{(1)} - m^{(1)}} $$
$$ m^{(2)} = \max_i x_i^{(2)}, \quad l^{(2)} = \sum_i e^{x_i^{(2)} - m^{(2)}} $$

The global maximum and combined normalizer are:
$$ m = \max(m^{(1)}, m^{(2)}) $$
$$ l = e^{m^{(1)} - m} l^{(1)} + e^{m^{(2)} - m} l^{(2)} $$

#### Incremental Output Aggregation Update Formula:
Let $O^{(1)} = \sum_j \frac{e^{x_j^{(1)} - m^{(1)}}}{l^{(1)}} V_j^{(1)}$. When combining with block 2:

$$ \mathbf{O = \frac{l^{(1)} e^{m^{(1)} - m}}{l} O^{(1)} + \frac{e^{m^{(2)} - m}}{l} \left( \sum_j e^{x_j^{(2)} - m^{(2)}} V_j^{(2)} \right)} $$

---

### 3.3 FlashAttention Tiling Algorithm & IO Complexity Proof

```
High Bandwidth Memory (HBM)
┌────────────────────────────────────────────────────────┐
│  Q (N x d)        K (N x d)        V (N x d)           │
└───────────────────────┬────────────────────────────────┘
                        │ Block Tiling (Load B_r, B_c into SRAM)
                        ▼
Streaming Multiprocessor SRAM (Fast On-Chip Memory: ~228KB)
┌────────────────────────────────────────────────────────┐
│  Q_i (B_r x d)   x   K_j^T (d x B_c)  ──► S_ij (B_r x B_c)│
│  Online Softmax Accumulation in Registers               │
│  O_i = Rescale(O_prev) + P_ij · V_j                    │
└───────────────────────┬────────────────────────────────┘
                        │ Write final output block directly to HBM
                        ▼
┌────────────────────────────────────────────────────────┐
│  O (N x d) in HBM (Zero N x N Intermediate Matrices!)  │
└────────────────────────────────────────────────────────┘
```

Let SRAM capacity be $M$ bytes. Block sizes are chosen such that $B_c, B_r \approx \Theta\left(\frac{M}{d}\right)$.
- **Outer Loop**: Iterate over blocks of $K, V$ of size $B_c \times d$ (loaded into SRAM once).
- **Inner Loop**: Iterate over blocks of $Q$ of size $B_r \times d$, update local output $O_i$ in registers, and write back only the final $N \times d$ output to HBM.

#### IO Complexity Proof (Dao et al., 2022):
- Standard Attention Memory IO: $\Theta(N d + N^2)$ accesses to HBM.
- FlashAttention Memory IO: $\Theta\left(\frac{N^2 d^2}{M}\right)$ accesses to HBM.
- For typical SRAM size $M \approx 100\text{KB}$ and $d = 128$: FlashAttention requires **$10\times$ to $20\times$ fewer HBM accesses**, achieving a **$3\times - 5\times$ end-to-end wall-clock speedup**.

---

### 3.4 FlashAttention-3 Innovations (Hopper H100 Architecture)

1. **Warp Specialization**: Divides SM warps into **Producer Warps** (loading data from HBM to Shared Memory via asynchronous TMA instructions) and **Consumer Warps** (executing Tensor Core MMA instructions without waiting on memory barriers).
2. **Ping-Pong Buffering**: Overlaps GEMM computation of block $k$ with asynchronous memory copy of block $k+1$.
3. **FP8 Tensor Core GEMMs**: Leverages FP8 low-precision arithmetic with block quantization, doubling theoretical peak compute to $1.98 \text{ PFLOPs/s}$.

---

## 4. CUDA Execution Mechanics: Warps, Divergence & Bank Conflicts

### 4.1 Warps and SIMT Execution
On NVIDIA GPUs, threads are grouped into **Warps of 32 threads**. All 32 threads in a warp execute the exact same instruction at the same clock cycle in a Single Instruction, Multiple Threads (SIMT) fashion.

- **Warp Divergence Penalty**:
  ```cuda
  if (threadIdx.x % 2 == 0) {
      // Path A (16 threads active, 16 threads masked/idle)
  } else {
      // Path B (16 threads active, 16 threads masked/idle)
  }
  ```
  *Cost*: Total execution time = $\text{Time}(\text{Path A}) + \text{Time}(\text{Path B})$. Effective compute throughput drops by $50\%$.

---

### 4.2 Shared Memory Bank Conflicts

Shared memory (SRAM) is organized into **32 independent banks** (each 4 bytes wide).  
Successive 32-bit words are assigned to successive banks: $\text{Bank Index} = (\text{Byte Address} / 4) \pmod{32}$.

```
Bank 0    Bank 1    Bank 2   ...  Bank 31
┌───────┐ ┌───────┐ ┌───────┐     ┌───────┐
│ Word 0│ │ Word 1│ │ Word 2│ ... │Word 31│  <-- Threads 0..31 access words 0..31 (0 conflicts: 1 cycle)
├───────┤ ├───────┤ ├───────┤     ├───────┤
│Word 32│ │Word 33│ │Word 34│ ... │Word 63│  <-- Threads 0..31 access words 0, 2, 4.. (2-way conflict: 2 cycles)
└───────┘ └───────┘ └───────┘     └───────┘
```

- **Conflict-Free Access**: All 32 threads in a warp access distinct banks in parallel (serviced in 1 cycle).
- **$k$-Way Bank Conflict**: If $k$ threads in the same warp request distinct addresses belonging to the *same bank*, the requests are serialized, taking **$k$ clock cycles**.
- **Padding Solution**: Pad 2D shared memory arrays (e.g. `__shared__ float smem[32][33]` instead of `[32][32]`) to skew row strides and eliminate bank collisions.

---

## 5. Python / Triton Implementation: Online Softmax Attention Block

```python
import torch
import math

def online_softmax_reference(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """
    Step-by-step Python simulation of FlashAttention's Online Softmax Tiling.
    Q, K, V: [Seq_Len, d_head]
    """
    S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    O = torch.zeros_like(Q)
    
    # Running statistics per row
    running_max = torch.full((S,), float('-inf'), device=Q.device)
    running_sum = torch.zeros(S, device=Q.device)

    # Outer loop: Iterate over K, V blocks (simulating loading into SRAM)
    for j_start in range(0, S, block_size):
        j_end = min(j_start + block_size, S)
        K_j = K[j_start:j_end] # [B_c, D]
        V_j = V[j_start:j_end] # [B_c, D]

        # Inner loop: Iterate over Q blocks
        for i_start in range(0, S, block_size):
            i_end = min(i_start + block_size, S)
            Q_i = Q[i_start:i_end] # [B_r, D]

            # Compute block dot products: [B_r, B_c]
            S_ij = torch.matmul(Q_i, K_j.t()) * scale
            
            # Causal mask for the block
            for r in range(i_start, i_end):
                for c in range(j_start, j_end):
                    if c > r:
                        S_ij[r - i_start, c - j_start] = float('-inf')

            # Local block maximum & exponential sums
            block_max, _ = torch.max(S_ij, dim=-1)
            new_max = torch.maximum(running_max[i_start:i_end], block_max)
            
            # Rescaling factors
            alpha = torch.exp(running_max[i_start:i_end] - new_max)
            P_ij = torch.exp(S_ij - new_max.unsqueeze(-1))
            P_ij = torch.nan_to_num(P_ij, nan=0.0)
            
            # Update running normalizer
            new_sum = running_sum[i_start:i_end] * alpha + P_ij.sum(dim=-1)

            # Incremental output update
            O_prev = O[i_start:i_end]
            O_block = torch.matmul(P_ij, V_j)
            
            # Rescale previous output and accumulate
            O[i_start:i_end] = (O_prev * (running_sum[i_start:i_end] * alpha).unsqueeze(-1) + O_block) / (new_sum.unsqueeze(-1) + 1e-8)
            
            running_max[i_start:i_end] = new_max
            running_sum[i_start:i_end] = new_sum

    return O

if __name__ == "__main__":
    S, D = 64, 32
    q, k, v = torch.randn(S, D), torch.randn(S, D), torch.randn(S, D)
    out_online = online_softmax_reference(q, k, v, block_size=16)
    
    # Standard Attention Baseline
    scores = torch.matmul(q, k.t()) / math.sqrt(D)
    mask = torch.triu(torch.full((S, S), float('-inf')), diagonal=1)
    weights = torch.softmax(scores + mask, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    out_standard = torch.matmul(weights, v)
    
    assert torch.allclose(out_online, out_standard, atol=1e-4), "Online softmax mismatch with standard attention!"
    print("Online Softmax mathematical tiling verified successfully.")
```

---

## 6. Deep Interview Interrogation Ladder

- **Level 1 (Concept)**: What is the primary difference between memory-bound and compute-bound operations?
- **Level 3 (Roofline)**: Calculate the arithmetic intensity of autoregressive token generation for a 70B parameter model at batch size 1, and explain why GPU compute utilization is $< 1\%$.
- **Level 5 (Hardware)**: What is a Shared Memory Bank Conflict, and how does array padding resolve it?
- **Level 7 (Derivation)**: Derive the incremental output update formula in FlashAttention's online softmax algorithm when combining two disjoint blocks.
- **Level 9 (Microarchitecture)**: Explain how FlashAttention-3 leverages TMA (Tensor Memory Accelerator) and Warp Specialization on NVIDIA Hopper GPUs to achieve asynchronous memory-compute overlap.
- **Level 10 (Principal Engineering)**: You are tasked with optimizing a custom mixture-of-experts model on an 8xH100 node. All-to-All communication between GPUs is stalling SM execution. Walk through how you would trace the bottleneck using NVIDIA Nsight Systems (`nsys`) and re-architect the dispatch kernel to overlap computation with NVLink transfers.

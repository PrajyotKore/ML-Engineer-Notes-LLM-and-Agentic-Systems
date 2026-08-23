# 09_INFERENCE_SYSTEMS — Mathematical & Systems Engineering Reference

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide an exhaustive mathematical, algorithmic, and systems reference on LLM Inference — covering PagedAttention, SGLang's RadixAttention, Disaggregated Prefill/Decode (PD split), Chunked Prefill scheduling, Speculative Decoding acceptance proofs, and FP8/INT4 quantization.

---

## 1. The Two-Phase Inference Lifecycle & Latency Modeling

Autoregressive LLM generation operates in two radically distinct phases:

```
[Request Arrives]
      │
      ▼
┌────────────────────────────────────────────────────────┐
│  Phase 1: Prefill (Prompt Processing)                  │
│  - Input: Entire Prompt of S_p tokens in parallel       │
│  - Nature: Compute-Bound (Dense GEMM on Tensor Cores)  │
│  - Metric: Time To First Token (TTFT)                  │
└───────────────────────┬────────────────────────────────┘
                        │ Initial KV Cache Generated & Cached
                        ▼
┌────────────────────────────────────────────────────────┐
│  Phase 2: Decode (Autoregressive Token Generation)     │
│  - Input: 1 new token per step, reads past S_t KV cache │
│  - Nature: Memory-Bound (Loading weights & KV from HBM)│
│  - Metric: Time Per Output Token (TPOT) / Inter-Token  │
└────────────────────────────────────────────────────────┘
```

### 1.1 First-Principles Latency & Throughput Formulations

Let:
- $P$: Model parameter count (e.g. $70 \times 10^9$).
- $S_p$: Prompt sequence length.
- $S_o$: Generated output sequence length.
- $B$: Concurrent batch size.
- $C_{\text{peak}}$: Peak GPU compute throughput (FLOPs/s, e.g. $989 \times 10^{12}$ for H100 FP16).
- $B_{\text{mem}}$: Peak GPU memory bandwidth (Bytes/s, e.g. $3.35 \times 10^{12}$ for H100 HBM3).
- $b_{\text{model}}$: Bytes per model parameter (2 for FP16, 1 for FP8).
- $b_{\text{kv}}$: Bytes per KV cache element (2 for FP16, 1 for FP8).

#### 1. Time To First Token (TTFT):
During prefill, the model computes $2 P$ FLOPs per prompt token across batch $B$:
$$ \text{FLOPs}_{\text{prefill}} = 2 \cdot P \cdot B \cdot S_p + 2 \cdot L \cdot N_h \cdot d_k \cdot B \cdot S_p^2 $$
Assuming compute dominates ($S_p \gg 1$):
$$ \mathbf{\text{TTFT} \approx \frac{2 \cdot P \cdot S_p}{\text{MFU} \cdot C_{\text{peak}}}} $$
*Example*: Processing $S_p = 4096$ prompt tokens on a 70B model using an 8xH100 node ($\text{MFU} \approx 0.60$, $C_{\text{peak}} = 8 \times 989\text{ TFLOPs}$):
$$ \text{TTFT} = \frac{2 \times (70 \times 10^9) \times 4096}{0.60 \times (8 \times 989 \times 10^{12})} \approx \mathbf{0.121 \text{ seconds (121 ms)}} $$

#### 2. Time Per Output Token (TPOT / Decode Step Latency):
At each decode step, the model loads all $P$ weights and the cumulative KV cache for all $B$ requests:
$$ \text{Bytes Transferred per Step} = (P \cdot b_{\text{model}}) + B \cdot \text{KVCache}_{\text{size}}(S_t) $$
$$ \mathbf{\text{TPOT}(t) = \frac{P \cdot b_{\text{model}} + B \cdot (2 \cdot L \cdot N_{kv} \cdot d_h \cdot b_{\text{kv}} \cdot S_t)}{B_{\text{mem}}}} $$

---

### 1.2 The KV Cache Memory Equation

For a Transformer with $L$ layers, $N_{kv}$ Key/Value heads, head dimension $d_h$, sequence length $S$, and batch size $B$:

$$ \mathbf{\text{Memory}_{\text{KV}} = 2 \times B \times S \times L \times N_{kv} \times d_h \times b_{\text{kv}} \quad [\text{Bytes}]} $$

#### Concrete Production Sizing Example:
- Model: LLaMA-3 70B ($L = 80, N_q = 64, N_{kv} = 8, d_h = 128$).
- Precision: FP16 ($b_{\text{kv}} = 2\text{ bytes}$).
- Context Length: $S = 8192$, Batch Size: $B = 32$.

$$ \text{Memory}_{\text{KV}} = 2 \times 32 \times 8192 \times 80 \times 8 \times 128 \times 2 = 85,899,345,920 \text{ Bytes} \approx \mathbf{80.0 \text{ GB}} $$

*Insight*: The KV cache alone occupies an entire 80GB H100 GPU's VRAM. Without memory management, KV cache fragmentation causes premature Out-Of-Memory (OOM) failures.

---

## 2. Memory & Scheduling Engines: PagedAttention vs. RadixAttention

### 2.1 PagedAttention (vLLM Architecture)

In traditional serving, KV caches are allocated contiguously for maximum sequence length ($S_{\max}$), causing **$60\%-80\%$ VRAM waste** due to internal fragmentation, external fragmentation, and reservation waste.

#### OS Virtual Memory Analogy:
PagedAttention partitions the KV cache into fixed-size **Physical Blocks (Pages)** (e.g. 16 or 32 tokens).
- **Logical KV Blocks**: Continuous token indices $0 \dots S-1$.
- **Physical Block Table**: Maps logical blocks to non-contiguous physical pages in GPU VRAM.
- **Copy-on-Write (CoW)**: When an agent branches or forks multiple parallel generation trajectories (e.g. Tree-of-Thought, Beam Search), child processes share parent physical pages until a new token is appended.

```
Logical Blocks (Request A):   [ Block 0 ] ──► Physical Page #14 (VRAM)
                              [ Block 1 ] ──► Physical Page #3  (VRAM)
                              [ Block 2 ] ──► Physical Page #89 (VRAM)
```

$$\text{Memory Waste in PagedAttention} \leq \frac{\text{Block Size} - 1}{\text{Average Sequence Length}} \approx \frac{15}{2048} < \mathbf{0.73\%}$$

---

### 2.2 RadixAttention (SGLang Architecture)

While PagedAttention manages memory per request, multi-turn AI agents and tool-calling loops frequently reuse long shared system prompts, tool schemas, and multi-turn dialogue histories.

#### Radix Tree Structure:
SGLang maintains a global **Radix Tree (Prefix Trie)** over all past and present KV cache pages in GPU VRAM:
- Nodes in the tree represent token sequences with associated KV cache pointers.
- When a new request arrives, SGLang traverses the Radix Tree to find the **longest common prefix match**.
- If matched, the engine **skips prefill entirely** for the matched prefix, reducing TTFT from seconds to milliseconds.
- **Cache Eviction Policy (LRU)**: When VRAM is full, least recently used leaf nodes are recursively evicted.

```
                  [ Root (Empty) ]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
[ System Prompt & Tool Schemas (1.5k tok) ] [ Math Agent Prompt (800 tok) ]
        │
   ┌────┴────────────────────────┐
   ▼                             ▼
[ Turn 1: User Query A ]   [ Turn 1: User Query B ]
   │
[ Turn 2: Tool Execution Result ]
```

---

## 3. Advanced Serving Architectures

### 3.1 Disaggregated Prefill and Decode (PD Split / Mooncake / Splitwise)

In unified serving nodes, heavy prefills and continuous decodes run on the same GPUs, causing:
1. **Preemption Stalls**: A massive prefill monopolizes Tensor Cores, stalling active decode streams and creating massive P99 TPOT latency spikes.
2. **Hardware Inefficiency**: Prefill requires massive compute (FLOPs-bound), while decode requires massive memory bandwidth.

#### The Disaggregated Architecture:
- **Prefill Fleet**: Compute-optimized GPUs (e.g. NVIDIA H100 with high TFLOPs) dedicated exclusively to processing prompt tokens.
- **KV Transfer Network**: High-speed RDMA / PCIe / NVLink transfers the generated KV cache directly to the Decode Fleet.
- **Decode Fleet**: Memory-bandwidth-optimized GPUs (or large clustered pools) dedicated exclusively to autoregressive token generation.

```
Incoming Request ──► [ Prefill Node (Compute Heavy) ] 
                            │ 
                            ▼ (RDMA KV Cache Transfer: ~100 GB/s)
                     [ Decode Node (Bandwidth Heavy) ] ──► Token Stream
```

---

### 3.2 Chunked Prefill & Iteration-Level Continuous Batching (Sarathi-Serve)

To prevent prefill requests from starving decode steps in a unified engine:
- Chunk long prefills into smaller segments (e.g., $C = 512$ tokens).
- Interleave chunked prefill GEMMs with ongoing decode steps within the same continuous batch iteration:
  $$ \text{Batch Budget} = \text{Chunked Prefill}(512 \text{ tokens}) + \sum_{i=1}^{B_{\text{decode}}} \text{Decode}(1 \text{ token}) $$
- Keeps GPU compute utilization near 100% while strictly bounding P99 Inter-Token Latency (ITL).

---

## 4. Speculative Decoding Mathematics

### 4.1 Target Model Verification & Expected Acceptance Length

Let $M_p$ be a small, fast **Draft Model** (e.g. 1B params), and $M_q$ be the large **Target Model** (e.g. 70B params).
1. Draft model autoregressively generates $K$ draft tokens: $(\tilde{x}_1, \dots, \tilde{x}_K)$.
2. Target model evaluates all $K$ tokens in a **single parallel forward pass** (as a prefill).
3. The acceptance criterion for token $\tilde{x}_{n}$ (Leviathan et al., 2023) is:
   $$ P(\text{Accept } \tilde{x}_n) = \min\left( 1, \; \frac{q(\tilde{x}_n \mid x_{<n})}{p(\tilde{x}_n \mid x_{<n})} \right) $$
   If rejected, sample replacement token from the residual distribution:
   $$ P_{\text{res}}(x) = \frac{\max(0, q(x) - p(x))}{1 - \sum_y \min(p(y), q(y))} $$

#### Mathematical Proof: Expected Tokens per Step ($\mathbb{E}[N]$)
Let $\alpha \in [0, 1]$ be the average acceptance probability per token:
$$ \mathbb{E}[N] = 1 + \sum_{k=1}^K \alpha^k = 1 + \frac{\alpha(1 - \alpha^K)}{1 - \alpha} = \mathbf{\frac{1 - \alpha^{K+1}}{1 - \alpha}} $$

- If $\alpha = 0.8$ and $K = 5$:
  $$ \mathbb{E}[N] = \frac{1 - 0.8^6}{1 - 0.8} = \frac{1 - 0.262}{0.2} = \mathbf{3.69 \text{ tokens/step}} $$

- **Theoretical Latency Speedup**:
  Let $t_p$ be the draft step time, and $t_q$ be the target step time ($t_p \ll t_q$).
  $$ \text{Speedup} = \frac{\mathbb{E}[N] \cdot t_q}{K \cdot t_p + t_q} = \frac{\frac{1 - \alpha^{K+1}}{1 - \alpha}}{1 + K \frac{t_p}{t_q}} $$

---

## 5. Quantization Mathematics: FP8, INT8, and INT4

```
FP16:  1 Sign | 5 Exponent | 10 Mantissa  (Dynamic Range: 10^-5 to 6.5x10^4)
FP8 E4M3: 1 Sign | 4 Exponent | 3 Mantissa  (Higher Precision: Best for Weights & Activations)
FP8 E5M2: 1 Sign | 5 Exponent | 2 Mantissa  (Wider Dynamic Range: Best for Gradients & KV Cache)
```

### 5.1 Symmetric Block Quantization
Given continuous weight vector $W \in \mathbb{R}^N$:
$$ s = \frac{\max(|W|)}{2^{b-1} - 1} \quad \text{(Quantization Scale Factor)} $$
$$ W_{\text{quant}} = \text{clamp}\left( \left\lfloor \frac{W}{s} \right\rceil, \; -2^{b-1}, \; 2^{b-1} - 1 \right) $$
$$ W_{\text{dequant}} = W_{\text{quant}} \times s $$

### 5.2 SmoothQuant: Outlier Migration (Xiao et al., 2023)
In LLMs, activations have extreme systematic outlier channels ($> 100\times$ normal values), making activation quantization difficult.  
SmoothQuant mathematically migrates the quantization difficulty from activations $X$ to weights $W$ by scaling channels with diagonal matrix $S = \text{diag}(s)$:

$$ Y = X W = (X S^{-1}) (S W) = \hat{X} \hat{W} $$
Where the per-channel scale $s_j$ is:
$$ s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1 - \alpha}} \quad (\alpha = 0.5 \text{ splits difficulty equally}) $$

---

## 6. PyTorch Simulation: Paged KV-Cache Allocator

```python
import torch
from typing import List, Dict

class PagedKVCacheManager:
    """
    Physical Block Memory Allocator for PagedAttention.
    """
    def __init__(self, num_blocks: int, block_size: int, num_layers: int, num_heads: int, head_dim: int):
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Allocate physical memory pool: [Num_Blocks, Num_Layers, 2 (K/V), Block_Size, Num_Heads, Head_Dim]
        self.gpu_memory_pool = torch.zeros(
            (num_blocks, num_layers, 2, block_size, num_heads, head_dim),
            dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Free list tracking available physical block IDs
        self.free_blocks = list(range(num_blocks))
        self.block_tables: Dict[int, List[int]] = {} # Request ID -> [Physical Block IDs]

    def allocate_request(self, request_id: int, prompt_len: int) -> List[int]:
        num_blocks_needed = (prompt_len + self.block_size - 1) // self.block_size
        assert len(self.free_blocks) >= num_blocks_needed, "Out of Memory (OOM): No free physical blocks!"
        
        allocated = [self.free_blocks.pop(0) for _ in range(num_blocks_needed)]
        self.block_tables[request_id] = allocated
        return allocated

    def append_token(self, request_id: int, current_len: int) -> int:
        """
        Allocates a new block if the current sequence length crosses block boundary.
        """
        if current_len % self.block_size == 0:
            assert len(self.free_blocks) > 0, "OOM during token decode!"
            new_block = self.free_blocks.pop(0)
            self.block_tables[request_id].append(new_block)
            return new_block
        return self.block_tables[request_id][-1]

    def free_request(self, request_id: int):
        if request_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[request_id])
            del self.block_tables[request_id]

if __name__ == "__main__":
    manager = PagedKVCacheManager(num_blocks=100, block_size=16, num_layers=4, num_heads=8, head_dim=64)
    req1_blocks = manager.allocate_request(request_id=1, prompt_len=35) # Requires 3 blocks
    assert len(req1_blocks) == 3
    print(f"Request 1 assigned physical blocks: {req1_blocks}")
    
    # Simulate generating tokens until new page is triggered
    manager.append_token(request_id=1, current_len=48) # 48 % 16 == 0 -> Allocates 4th block
    assert len(manager.block_tables[1]) == 4
    print(f"Request 1 expanded to physical blocks: {manager.block_tables[1]}")
    
    manager.free_request(request_id=1)
    assert len(manager.free_blocks) == 100
    print("Paged KV Cache Allocation and Free Lifecycle Verified.")
```

---

## 7. Deep Interview Interrogation Ladder

- **Level 1 (Concept)**: What is the primary hardware difference between the prefill phase and the decode phase in LLM serving?
- **Level 3 (Derivation)**: Calculate the exact KV cache memory footprint for a 70B parameter model ($L=80, N_{kv}=8, d_h=128$) at batch size 64 with 8k context in FP16.
- **Level 5 (Mechanics)**: How does PagedAttention eliminate external and internal memory fragmentation?
- **Level 7 (Speculative Decoding Proof)**: Prove mathematically why the expected number of tokens accepted per step in Speculative Decoding is $\frac{1 - \alpha^{K+1}}{1 - \alpha}$.
- **Level 9 (Serving Architecture)**: Why does unified serving suffer from P99 latency spikes during long prefills, and how does Disaggregated Prefill/Decode (PD Disaggregation) resolve it?
- **Level 10 (Principal Engineering)**: Design an inference cluster architecture serving a high-volume agentic workflow with shared system prompts and dynamic multi-step branching. Detail the cache eviction policy, KV transfer protocol across nodes, and speculative draft validation strategy.

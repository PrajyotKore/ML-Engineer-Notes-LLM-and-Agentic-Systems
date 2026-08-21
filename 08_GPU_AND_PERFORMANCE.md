# 08_GPU_AND_PERFORMANCE — Technical Reference

## 1. Role Relevance
For an A1 Technical Lead, GPU performance is not a black box. You must understand how PyTorch operations map to physical hardware. If a training run or inference deployment is scaling poorly, you must trace the bottleneck down to the Streaming Multiprocessor (SM) or High Bandwidth Memory (HBM) level.

## 2. Prerequisites
- Matrix Multiplications.
- Basic C++/CUDA execution model.
- Memory hierarchies.

## 3. First Principles
A GPU is a massive parallel processor designed for high-throughput arithmetic. Unlike a CPU which optimizes for low latency (large caches, branch prediction), a GPU optimizes for throughput by executing thousands of threads simultaneously. When one thread stalls waiting for memory, the GPU instantly switches to another thread to hide the latency (latency hiding).

## 4. Mechanistic Breakdown
### GPU Architecture Hierarchy
1. **HBM (High Bandwidth Memory)**: The main VRAM (e.g., 80GB on H100). Very large, but relatively slow ($\sim 3.3$ TB/s).
2. **L2 Cache**: Shared across the entire GPU.
3. **Streaming Multiprocessors (SMs)**: The actual compute engines. An H100 has 132 SMs.
4. **SRAM (Shared Memory / L1 Cache)**: Extremely fast memory local to an SM ($\sim 33$ TB/s).
5. **Registers**: The fastest memory, tied to specific threads.
6. **Tensor Cores**: Specialized arithmetic units inside SMs designed specifically to perform $4 \times 4$ matrix multiply-accumulate operations (MMA) in a single clock cycle.

### Execution Model
A CUDA **Kernel** is launched on a grid of **Thread Blocks**. Each Block is assigned to one SM. Inside the SM, threads are executed in groups of 32 called **Warps**. All threads in a warp execute the exact same instruction at the same time (SIMT).

## 5. Mathematical Foundations

### Arithmetic Intensity & Roofline Model
To determine if a kernel is compute-bound or memory-bound, we calculate its Arithmetic Intensity (AI).

$$ \text{AI} = \frac{\text{Total FLOPs executed}}{\text{Total Bytes read/written to HBM}} $$

**The Roofline:**
- If $\text{AI} < \frac{\text{GPU Peak FLOPs}}{\text{GPU Peak Bandwidth}}$, the operation is **Memory-Bound**.
- If $\text{AI} \geq \frac{\text{GPU Peak FLOPs}}{\text{GPU Peak Bandwidth}}$, the operation is **Compute-Bound**.

*Example: Vector addition ($Z = X + Y$ in FP16).*
FLOPs = 1 (addition).
Bytes = 2 (read X) + 2 (read Y) + 2 (write Z) = 6 bytes.
AI = $1/6 \approx 0.16$ FLOPs/Byte.
On an H100 (Peak AI $\approx 300$ for BF16), vector addition is massively memory-bound.

## 6. Implementation
**Kernel Fusion:**
Because element-wise operations (like ReLU or Dropout) have terrible Arithmetic Intensity (memory-bound), we fuse them.
Instead of:
1. Load $X$ from HBM, compute $Y = XW$, write $Y$ to HBM.
2. Load $Y$ from HBM, compute $Z = \text{ReLU}(Y)$, write $Z$ to HBM.

A **Fused Kernel** does:
1. Load $X$ from HBM, compute $Y = XW$ in registers, compute $Z = \text{ReLU}(Y)$ in registers, write $Z$ to HBM.
*Result: Halves the memory bandwidth requirement.*

## 7. Hardware / GPU Behavior
### FlashAttention
FlashAttention is the ultimate example of hardware-aware optimization. Standard attention computes $S = QK^T$, writes $S$ (size $L \times L$) to HBM, then reads $S$ to compute $\text{softmax}(S)$, writes to HBM, reads to multiply by $V$.
This $O(L^2)$ HBM read/write destroys performance.

FlashAttention tiles the $Q, K, V$ matrices. It loads small blocks into the extremely fast SRAM, computes the softmax locally using an algebraic trick (online softmax), and writes the final output directly to HBM without ever materializing the $L \times L$ matrix in HBM.

## 8. Production Architecture
- **CUDA Graphs**: Launching thousands of small kernels from Python (CPU) to the GPU has overhead. If the kernel takes 5$\mu$s but the CPU takes 10$\mu$s to launch it, the GPU sits idle. CUDA Graphs trace the sequence of operations and launch them all at once directly on the GPU. Essential for low-latency inference.

## 9. Scalability & Bottlenecks
- **Occupancy**: The ratio of active warps on an SM to the maximum possible warps. Low occupancy means the SM cannot hide memory latency. Often caused by a kernel demanding too many registers per thread.
- **Warp Divergence**: If an `if/else` statement causes half the threads in a warp to take the `if` branch and half to take `else`, both branches execute sequentially, halving compute efficiency.

## 10. Failure Modes
- **Uncoalesced Memory Access**: If thread 0 reads index 0, and thread 1 reads index 100, the GPU must fetch two separate cache lines from HBM. Memory bandwidth collapses.
- **Tensor Core Unaligned Dimensions**: If a matrix dimension is not a multiple of 8 or 16, the GPU falls back to slow CUDA cores instead of Tensor Cores.

## 11. Debugging
- **Profiling with Nsight Systems (nsys)**: Shows the CPU timeline, kernel launches, and GPU execution. If there are huge gaps between green GPU execution blocks, you are CPU-bound (Python overhead).
- **Profiling with Nsight Compute (ncu)**: Shows SM utilization, memory bandwidth utilization, and Arithmetic Intensity for a specific kernel.

## 12. Principal-Level Reasoning
"If my inference cluster throughput is suddenly 30% lower after a model update, I don't just guess. I use `nsys` to capture a trace. I check if we broke CUDA Graph capture, causing CPU launch overhead. If the GPU is fully active, I use `ncu` on the dense layers. If I see unaligned dimensions, I know someone changed a projection size to a non-multiple of 8, disabling Tensor Cores and forcing the GPU to use CUDA cores."

## 13. Interview Interrogation
- *Level 2*: What is the difference between HBM and SRAM?
- *Level 5*: Explain the Roofline model.
- *Level 7*: Why is calculating standard cross-entropy over a 100k vocabulary memory-bound?
- *Level 9*: How does FlashAttention compute softmax in blocks without knowing the global maximum for the denominator?
- *Level 10*: You are writing a custom GPU kernel for a new MoE routing mechanism. How do you ensure maximum memory bandwidth utilization?

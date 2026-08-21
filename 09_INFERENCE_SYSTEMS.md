# 09_INFERENCE_SYSTEMS — Technical Reference

## 1. Role Relevance
For an A1 Technical Lead, inference is where the unit economics of the product are determined. You must serve a proactive agent reliably with low latency (TTFT, TPOT) while maximizing GPU utilization (Throughput/Cost). Understanding how to schedule requests, batch dynamically, and manage the KV cache is critical to production.

## 2. Prerequisites
- Transformer Forward Pass (Prefill vs. Decode).
- KV Cache memory footprint.
- CUDA memory hierarchies (SRAM vs. HBM).

## 3. First Principles
LLM inference consists of two distinct phases:
1. **Prefill**: Processes the entire prompt in parallel. Compute-bound (matrix multiplication).
2. **Decode**: Generates one token at a time autoregressively. Memory-bound (reading the massive KV cache from HBM for every token).

The goal of an inference engine (like vLLM or TRT-LLM) is to maximize the batch size during the decode phase to amortize the memory bandwidth costs.

## 4. Mechanistic Breakdown
### Continuous Batching
Traditional static batching waits for all requests in a batch to finish generating before starting a new batch. Continuous batching (or iteration-level scheduling) inserts new requests into the batch at the very next decode step as soon as an older request finishes, keeping GPU utilization near 100%.

### PagedAttention
The KV cache grows dynamically as tokens are generated. Storing it contiguously causes memory fragmentation. PagedAttention divides the KV cache into fixed-size "pages" (e.g., 16 tokens) that can be stored non-contiguously in GPU VRAM, similar to virtual memory in an OS.

### Speculative Decoding
Since decode is memory bandwidth-bound, the GPU's compute units (Tensor Cores) sit idle. Speculative decoding uses a smaller, faster "draft" model to predict $K$ tokens. The large "target" model then verifies these $K$ tokens in a single parallel step. If $N$ tokens are accepted, you get $N$ tokens for the latency of 1 step.

## 5. Mathematical Foundations

### KV Cache Memory Size
Memory per token per layer = $2 \times \text{num\_heads} \times d_{head} \times 2 \text{ bytes (FP16/BF16)}$.
Total memory for $L$ tokens = $\text{Memory per token per layer} \times \text{num\_layers} \times L$.

### Inference Latency Model
**Time To First Token (TTFT)**: Driven by the compute time of the prefill matrix multiplication.
$$ \text{TTFT} \approx \frac{2 \cdot N_{params} \cdot L_{prompt}}{\text{GPU\_Compute\_Bandwidth}} $$

**Time Per Output Token (TPOT)**: Driven by the memory bandwidth to load the model weights and the KV cache.
$$ \text{TPOT} \approx \frac{(N_{params} \cdot 2 \text{ bytes}) + \text{Total\_KVCache\_Bytes}}{\text{GPU\_Memory\_Bandwidth}} $$

*Note: TPOT assumes the batch size is small enough that compute does not dominate.*

## 6. Implementation
**Prefix Caching Example:**
When deploying agents, multiple users often share the same system prompt or tools. Instead of recomputing the prefill for the system prompt on every request, the engine hashes the system prompt tokens and stores their KV cache in a shared memory pool.

## 7. Computational Complexity
- **Prefill**: $O(L^2 \cdot d_{model})$ compute.
- **Decode**: $O(L \cdot d_{model})$ memory access per token.

## 8. Hardware / GPU Behavior
- **Memory Bandwidth Bottleneck**: An H100 has $\sim 3.3$ TB/s of memory bandwidth. A 70B parameter model in FP16 is $140$ GB. Generating one token requires loading all $140$ GB into SRAM.
Thus, maximum theoretical speed for batch size 1 = $\frac{3300}{140} \approx 23$ tokens/second.
This is why large batching is mandatory for throughput.

## 9. Production Architecture
- **Router / Admission Control**: Sits in front of the inference engines. Tracks the active KV cache utilization of the cluster. If VRAM is 90% full, it queues requests rather than sending them to the GPU to prevent OOM.
- **Tensor Parallelism (TP) for Inference**: Splits the model across multiple GPUs within the same node (e.g., TP=4). This aggregates the memory bandwidth of 4 GPUs, dividing TPOT by 4, drastically reducing latency for large models.

## 10. Scalability
To scale to millions of requests, you deploy identical TP replicas.
Total System Throughput = $\text{Requests per second per replica} \times \text{Number of Replicas}$.

## 11. Bottlenecks
- **KV Cache Fragmentation**: Without PagedAttention, VRAM fragments, reducing maximum batch size by 20-40%.
- **Context Length**: A 128k context prompt takes a massive amount of VRAM, limiting the batch size to 1 or 2, severely degrading cluster throughput.

## 12. Failure Modes
- **Out of Memory (OOM)**: Usually happens during decode when a request generates far more tokens than anticipated and the engine runs out of free KV cache pages. Modern engines swap to CPU RAM or preempt the request.
- **High TTFT Spikes**: Happens when a long-context request triggers a massive prefill, blocking the decode steps of all other active requests in the continuous batch.

## 13. Debugging
- **Low Throughput**: Check GPU Utilization and Memory Utilization. If VRAM is only 50% full, admission control is too conservative.
- **High TPOT**: Batch size might be too large, causing the decode step to become compute-bound rather than memory-bound, or you are running TP=1 on a model that requires TP=2.

## 14. Trade-offs
- **Throughput vs Latency**: Increasing batch size increases total tokens/sec (throughput, lowering cost) but slightly increases TPOT (latency, degrading user experience).

## 15. Principal-Level Reasoning
"If users complain about latency spikes in the agent, I look at the TTFT vs TPOT metrics. If TTFT is high, it means long system prompts are blocking the queue; I would implement chunked prefill to break large prefills into smaller segments that interleave with decodes. If TPOT is high, we are likely hitting a memory bandwidth wall, and I would increase Tensor Parallelism or aggressively quantize the KV cache to FP8."

## 16. Interview Interrogation
- *Level 1*: What is the difference between prefill and decode?
- *Level 3*: Why does generating tokens get slower as the sequence gets longer?
- *Level 6*: Explain how PagedAttention solves KV cache fragmentation.
- *Level 8*: Why does speculative decoding increase throughput without degrading accuracy?
- *Level 10*: Design an autoscaling inference architecture that guarantees P99 TTFT < 1 second for a heavily agentic workload with highly variable prompt lengths.

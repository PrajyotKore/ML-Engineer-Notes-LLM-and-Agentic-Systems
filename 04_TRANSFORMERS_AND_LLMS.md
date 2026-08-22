# 04_TRANSFORMERS_AND_LLMS — Technical Reference

## 1. Role Relevance
For an ML Engineer (LLM & Agentic Systems), understanding the mechanics of Transformers is a P0 requirement. Everything in production—from GPU memory bounds (KV cache) to latency constraints (prefill vs. decode) to alignment constraints (long-context retrieval)—is governed by the mathematical and hardware realities of the Transformer architecture.

## 2. Prerequisites
- Matrix operations, Softmax, Deep Learning Foundations.
- Hardware concepts (Memory Bandwidth vs. Arithmetic Logic Units).

## 3. First Principles
A Transformer maps a sequence of discrete tokens to a dense sequence of contextualized embeddings. It leverages **Self-Attention** to route information across tokens without recurrent state, and **Feed-Forward Networks (FFN)** to compute non-linear feature transformations per-token.

## 4. Mechanistic Breakdown
### The LLM Forward Pass
1. **Tokenization**: Subword splits (e.g., BPE, Tiktoken).
2. **Embeddings**: Token IDs $\rightarrow$ dense vectors.
3. **Positional Encoding (RoPE)**: Injects relative distance into query/key vectors.
4. **Attention Layer**: Tokens read from each other (mixing across sequence length).
5. **Residual & RMSNorm**: Stabilizes gradients; prevents exploding activations.
6. **FFN (SwiGLU)**: Non-linear expansion and projection (mixing feature dimensions).
7. **LM Head**: Projects final embedding back to the vocabulary space.
8. **Sampling**: Converts logits to next-token probabilities via temperature and top-p/k.

## 5. Mathematical Foundations

### Scaled Dot-Product Attention
Attention computes the weighted sum of Value vectors, where weights are derived from the alignment of Query and Key vectors.

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \right)V $$

Where:
- $Q \in \mathbb{R}^{B \times L \times d_k}$ (Queries)
- $K \in \mathbb{R}^{B \times L \times d_k}$ (Keys)
- $V \in \mathbb{R}^{B \times L \times d_v}$ (Values)
- $d_k$ is the dimension of the key/query head.
- $\sqrt{d_k}$ prevents the dot product variance from scaling with $d_k$, which would push the softmax into vanishing gradient regions.

### Rotary Positional Embeddings (RoPE)
Instead of adding absolute positions, RoPE rotates the query and key vectors in 2D pairs according to their absolute position $m$. The dot product of two rotated vectors depends only on their relative distance $(m - n)$.

For a 2D slice of a vector $x$ at position $m$:
$$ R_{\Theta, m} x = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} $$
$$ \theta_i = \Theta^{-2i/d} \quad \text{where typically } \Theta = 10000 $$

### SwiGLU Feed-Forward Network
Used in LLaMA architecture in place of standard ReLU FFN. It introduces a gating mechanism.

$$ \text{SwiGLU}(x) = (\text{Swish}(x W_1) \otimes (x W_2)) W_3 $$
$$ \text{Swish}(z) = z \cdot \sigma(\beta z) $$

*Dimensions*: $W_1, W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $W_3 \in \mathbb{R}^{d_{ff} \times d_{model}}$. To keep parameter count identical to standard FFN, $d_{ff}$ is typically $\approx \frac{8}{3} d_{model}$.

## 6. Implementation
**Causal Masking:**
To ensure tokens can only attend to past tokens (autoregressive property), we apply a causal mask:
$$ M_{i,j} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases} $$
$$ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$

## 7. Computational Complexity
- **Prefill (Prompt Processing)**: Dense matrix multiplication. For sequence length $L$, the attention computation (computing $QK^T$) is $O(L^2 \cdot d_{model})$ FLOPs. Compute-bound.
- **Decode (Generation)**: Generates one token at a time. The attention is computed between a single new Query and $L$ cached Keys/Values. Thus, memory bandwidth (reading the KV cache) dictates speed. Memory-bound.

## 8. Hardware / GPU Behavior
### KV Cache
During decode, we do not recompute Keys and Values for past tokens. We store them in the **KV Cache**.
Memory required per token: $2 \times \text{num\_layers} \times \text{num\_heads} \times d_{head} \times 2 \text{ bytes (FP16)}$.
For a 70B model with 80 layers and 64 heads, the KV cache grows massively, bounding the maximum batch size.

### GQA (Grouped Query Attention)
To reduce KV Cache memory, GQA groups multiple query heads to share a single Key/Value head. E.g., if we have 64 query heads and 8 KV heads, we reduce KV cache memory by $8\times$.

## 9. Production Architecture
- **FlashAttention**: Rewrites the attention kernel to load blocks of Q, K, and V from HBM (slow) into SRAM (fast), compute the local softmax, and write the output back. This avoids materializing the massive $O(L^2)$ attention matrix in HBM, drastically improving latency and saving memory.
- **RMSNorm**: Cheaper than LayerNorm. It ignores the mean and only scales by the Root Mean Square, saving synchronization overhead on the GPU.

## 10. Scalability & Bottlenecks
- **Context Length**: Memory scales $O(L)$ with KV Cache, but compute scales $O(L^2)$. Extending context requires RoPE scaling (e.g., YaRN, linear scaling) and techniques like Ring Attention for distributed long-context.
- **MoE (Mixture of Experts)**: To scale parameter count without scaling active compute (FLOPs), MoE routes a token to 2 out of $N$ experts. This increases RAM footprint significantly but keeps generation fast.

## 11. Failure Modes
- **Lost in the Middle**: LLMs struggle to retrieve information located in the middle of a very long context window, strongly preferring the start and end.
- **Repetition Collapse**: If temperature is too low and penalty isn't applied, the model can enter an infinite loop of repeating tokens.

## 12. Interview Interrogation
- *Level 1*: What is the KV Cache?
- *Level 3*: Why does standard attention scale $O(L^2)$?
- *Level 5*: How does RoPE differ from absolute positional embeddings?
- *Level 8*: Walk me through the memory layout of FlashAttention in SRAM vs HBM.
- *Level 10*: If we deploy a 70B model and want a batch size of 256 for a 4K context, calculate exactly how much GPU VRAM the KV cache consumes and how you would architect the serving layer to handle it.

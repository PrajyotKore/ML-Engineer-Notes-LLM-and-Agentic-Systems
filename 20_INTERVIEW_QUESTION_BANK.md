# 20_INTERVIEW_QUESTION_BANK — 50+ Graded Interview Questions

> **Audience**: Candidates preparing for Senior, Staff, and Principal ML / LLM Systems Engineer interviews.  
> **Structure**: Graded from **Level 1 (Foundations)** to **Level 10 (Principal Systems Architecture)**.

---

## 1. Mathematics, Attention & Transformer Mechanics

- **Level 1**: What is the mathematical difference between an autoregressive decoder and a masked encoder?
- **Level 2**: Why is the Softmax function invariant to constant addition ($z - c$)?
- **Level 3**: Derive the analytical gradient of Cross-Entropy Loss with Softmax: $\nabla_z \mathcal{L} = p - y$.
- **Level 4**: Prove mathematically why Query-Key dot products have variance equal to $d_k$, and why scaling by $\frac{1}{\sqrt{d_k}}$ is mandatory.
- **Level 5**: In Rotary Position Embedding (RoPE), prove that $\langle R_m q, R_n k \rangle = g(q, k, m-n)$ preserves relative distance.
- **Level 6**: Compare MHA, MQA, GQA, and MLA in terms of KV Cache footprint and compute FLOPs.
- **Level 7**: Walk through the low-rank Key-Value compression and decoupled RoPE projections in Multi-Head Latent Attention (MLA).
- **Level 8**: How does YaRN (Yet another RoPE extensioN) scale context windows without degrading local token relationships?
- **Level 9**: Explain why SwiGLU requires $d_{ff} = \frac{8}{3} d_{\text{model}}$ to maintain exact parameter parity with standard ReLU FFN.
- **Level 10**: Derive the memory layout and online softmax incremental update equations for FlashAttention-1, 2, and 3 on NVIDIA H100 GPUs.

---

## 2. Post-Training, Alignment & Reasoning Models

- **Level 1**: What is the difference between SFT and Preference Optimization?
- **Level 2**: Why do we mask prompt tokens from the SFT loss calculation?
- **Level 3**: Derive the Bradley-Terry preference probability model from first principles.
- **Level 4**: In LoRA, why is matrix $B$ initialized to zero and matrix $A$ to Gaussian noise?
- **Level 5**: Walk through the complete mathematical derivation of DPO from the RLHF objective. Why does the partition function $Z(x)$ cancel out?
- **Level 6**: Explain the NormalFloat4 (NF4) quantile construction in QLoRA.
- **Level 7**: How does Group Relative Policy Optimization (GRPO) compute advantages without a Critic model?
- **Level 8**: Why do rule-based verifiable rewards prevent reward hacking compared to neural reward models in mathematical reasoning tasks?
- **Level 9**: Derive the analytical gradient of the DPO loss and explain the function of the implicit error weight $\sigma(\hat{r}_l - \hat{r}_w)$.
- **Level 10**: Design an automated test-time compute scaling harness combining Monte Carlo Tree Search (MCTS), Process Reward Models (PRMs), and GRPO for code generation.

---

## 3. GPU Architecture, Performance & Inference Systems

- **Level 1**: What is the difference between Time To First Token (TTFT) and Time Per Output Token (TPOT)?
- **Level 2**: Why is autoregressive decode memory-bandwidth bound at batch size 1?
- **Level 3**: Calculate the critical ridge point $I^*$ in the Roofline Model for an NVIDIA H100 GPU.
- **Level 4**: Calculate the exact KV cache memory footprint for a 70B model at batch size 32 and 8k context in FP16.
- **Level 5**: How does PagedAttention eliminate memory fragmentation in GPU VRAM?
- **Level 6**: Explain the Radix Tree prefix caching mechanism in SGLang.
- **Level 7**: Prove mathematically why the expected number of tokens accepted per step in Speculative Decoding is $\frac{1 - \alpha^{K+1}}{1 - \alpha}$.
- **Level 8**: How does SmoothQuant mathematically migrate quantization outliers from activations to weights?
- **Level 9**: Compare FP8 E4M3 and E5M2 data formats. Which format is optimal for weights vs. gradients?
- **Level 10**: Architect a Disaggregated Prefill and Decode (PD Split) cluster for a 100k QPS serving platform with RDMA KV-cache transfers.

---

## 4. Distributed Systems & Training Infrastructures

- **Level 1**: What is Data Parallelism vs. Model Parallelism?
- **Level 2**: Why does AdamW require 16 bytes of static memory per parameter?
- **Level 3**: Prove that Ring All-Reduce communication volume per node is $2 \left(\frac{N-1}{N}\right) M$, independent of node count $N$.
- **Level 4**: Explain the memory savings and communication primitives in ZeRO-1, ZeRO-2, and ZeRO-3 (FSDP).
- **Level 5**: How does Megatron-LM split Linear layers in Tensor Parallelism using Column and Row Parallel layers?
- **Level 6**: Derive the Pipeline Bubble Fraction $F_{\text{bubble}} = \frac{p-1}{m+p-1}$ for a 1F1B schedule.
- **Level 7**: Compare Ring Attention and DeepSpeed Ulysses for context parallelism at 512k sequence lengths.
- **Level 8**: What is GPUDirect RDMA and how does it bypass host CPU memory during inter-node transfers?
- **Level 9**: Explain how DeepSeek-V3 implements Auxiliary-Loss-Free load balancing across Mixture-of-Experts (MoE) nodes.
- **Level 10**: Design the 4D Parallelism strategy (TP $\times$ PP $\times$ EP $\times$ DP) and network topology for training a 1-trillion parameter model across 16,384 GPUs.

---

## 5. Agentic Systems, Reliability & Production Engineering

- **Level 1**: What is the ReAct loop?
- **Level 2**: Why does an agent with 95% single-step accuracy fail on a 20-step task?
- **Level 3**: How does FSM logit biasing guarantee 100% valid JSON schema adherence?
- **Level 4**: Derive the Two-Proportion Z-Test formula for A/B testing and calculate the sample size needed to detect a 2% gain.
- **Level 5**: How does Full Jitter in Exponential Backoff eliminate Thundering Herds?
- **Level 6**: Explain the Bradley-Terry ELO rating update formula used in Chatbot Arena.
- **Level 7**: How does the Distributed Saga pattern handle compensation rollbacks during multi-system agent failures?
- **Level 8**: Calculate the Population Stability Index (PSI) and explain the standard threshold bounds for data drift.
- **Level 9**: Design an OpenTelemetry distributed tracing schema for recursive multi-agent reflection loops.
- **Level 10**: Architect an enterprise-grade autonomous agent platform with Firecracker microVM sandboxing, Model Context Protocol (MCP), and Temporal durable execution capable of handling 50,000 concurrent long-running workflows.

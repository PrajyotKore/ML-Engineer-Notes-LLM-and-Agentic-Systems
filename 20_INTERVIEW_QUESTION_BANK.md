# 20_INTERVIEW_QUESTION_BANK — Technical Reference

## 1. Role Relevance
This document centralizes the "Interrogation Ladder" for the ML Engineer (LLM & Agentic Systems) interview. You must be able to navigate from a high-level conceptual question down to the bare metal mathematics.

## 2. LLM Mechanics & Transformers
- **Level 1**: What is the difference between an Encoder (BERT) and a Decoder (GPT)?
- **Level 3**: Explain the mathematics of Scaled Dot-Product Attention. Why divide by $\sqrt{d_k}$?
- **Level 5**: What is the KV Cache? Calculate the memory footprint of the KV cache for a 70B model with a 4K context and batch size of 64.
- **Level 7**: Compare and contrast Multi-Head Attention, Grouped Query Attention (GQA), and Multi-Query Attention (MQA). Why does Llama-3 use GQA?
- **Level 9**: Explain how FlashAttention-2 optimizes SRAM access compared to FlashAttention-1.

## 3. Post-Training (SFT & DPO)
- **Level 1**: What is the purpose of RLHF?
- **Level 3**: How does the DPO loss function differ mathematically from the PPO objective?
- **Level 5**: Derive the LoRA weight update. Why is matrix $B$ initialized to zero?
- **Level 7**: If you fine-tune a model and its evaluation loss drops, but it begins repeating itself infinitely in production, what is the root cause?
- **Level 9**: Design a synthetic data generation pipeline to teach an 8B model to use a completely undocumented internal REST API.

## 4. Agentic ML Systems
- **Level 1**: Explain the ReAct prompt structure.
- **Level 3**: Why do long-running agent workflows fail? (Write the probability equation).
- **Level 5**: How do you enforce JSON schema compliance on an LLM's output?
- **Level 7**: Differentiate between Semantic Memory and Episodic Memory in an agent's RAG pipeline.
- **Level 9**: Design the context-management algorithm for an agent that monitors a Slack channel 24/7. How do you prevent the context from overflowing while maintaining memory of important events?

## 5. Distributed Systems & Training
- **Level 1**: What is Data Parallelism?
- **Level 3**: Explain the memory overhead of the Adam optimizer.
- **Level 5**: How does FSDP (ZeRO-3) shard the model states, and what communication overhead does it incur?
- **Level 7**: What is a Pipeline Bubble, and how does the 1F1B schedule mitigate it?
- **Level 9**: Your 1,024 GPU cluster is achieving only 15% MFU (Model FLOPs Utilization). Walk me through how you isolate the bottleneck using NCCL metrics.

## 6. Inference & Production ML
- **Level 1**: What is Time To First Token (TTFT)?
- **Level 3**: Why is LLM decode memory-bound? Explain the Roofline Model.
- **Level 5**: How does PagedAttention reduce VRAM fragmentation?
- **Level 7**: Explain Speculative Decoding. Why doesn't it degrade the quality of the final output?
- **Level 9**: A new model is deployed to 1% of users (Canary). Latency is normal, but the task completion rate drops by 10%. How do you debug a semantic regression in production?

## 7. Architecture & System Design
- **Level 5**: Design a rate-limiter for the A1 inference API.
- **Level 7**: Design a durable workflow engine that guarantees idempotency for agent tool calls.
- **Level 10**: Architect the complete end-to-end A1 AI Assistant platform, from data ingestion to RLHF training to low-latency inference routing. Include fault-tolerance and cross-region failover.

# 16_SYSTEM_DESIGN — Whiteboard Architecture & Calculations

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide three end-to-end, production-grade system designs with quantitative dimensioning, back-of-the-envelope math, component diagrams, and failure isolation boundaries.

---

## 1. The 6-Step Principal ML System Design Framework

When presented with an open-ended system design prompt in a technical interview:

1. **Clarify Requirements & Constraints** (Functional vs. Non-Functional, Latency targets, QPS, Token budgets).
2. **Back-of-the-Envelope Dimensioning** (VRAM, FLOPs, Bandwidth, Storage, Cluster Size).
3. **High-Level API Design** (gRPC / REST / Server-Sent Events).
4. **Data Schemas & Storage Architecture** (Relational, Vector, Key-Value, Blob).
5. **Detailed Component Architecture & Control Flow** (Drawing boxes and data pipelines).
6. **Deep Dives & Failure Modes** (OOM prevention, tail latency mitigation, failover).

---

## 2. Case Study 1: Ultra-High-Throughput Distributed LLM Serving Platform (100k QPS)

**Prompt**: *"Design a distributed inference system to serve a 70B parameter model at 100,000 requests per minute with P99 TTFT $< 200\text{ ms}$ and TPOT $< 25\text{ ms/token}$."*

```
                                    ┌────────────────────────────────────────────────────────┐
                                    │                Anycast DNS & Cloudflare                │
                                    └───────────────────────────┬────────────────────────────┘
                                                                │
                                                                ▼
                                    ┌────────────────────────────────────────────────────────┐
                                    │         API Gateway & Envoy Load Balancers             │
                                    │  (TLS Termination, Rate Limiting, User Authentication) │
                                    └───────────────────────────┬────────────────────────────┘
                                                                │
                                                                ▼
                                    ┌────────────────────────────────────────────────────────┐
                                    │            Global Radix Router Fleet                   │
                                    │  - Evaluates Prefix Matches (RadixAttention Cache)     │
                                    │  - Dispatches Prefill vs Decode Jobs via gRPC          │
                                    └─────────────┬────────────────────────────┬─────────────┘
                                                  │                            │
                   ┌──────────────────────────────┘                            └──────────────────────────────┐
                   ▼                                                                                          ▼
┌────────────────────────────────────────────────────────┐                                 ┌────────────────────────────────────────────────────────┐
│           Prefill GPU Fleet (H100 SXM5)                │                                 │            Decode GPU Fleet (H100 SXM5)                │
│  - Compute-Bound Dense GEMMs (FlashAttention-3)        │                                 │  - Memory-Bandwidth-Bound Iteration Batching           │
│  - Chunked Prefill (512 token blocks)                  │                                 │  - Continuous Batching (vLLM / SGLang)                 │
│  - Generates initial KV Cache                          │                                 │  - FP8 Quantized Weights & KV Cache                    │
└──────────────────────────┬─────────────────────────────┘                                 └──────────────────────────▲─────────────────────────────┘
                           │                                                                                          │
                           └────────────────────────────── RDMA KV Transfer ──────────────────────────────────────────┘
                                                          (100 GB/s InfiniBand NDR)
```

### 2.1 Back-of-the-Envelope Dimensioning Math
- **Throughput**: $100,000 \text{ RPM} \approx 1,667 \text{ QPS}$.
- **Average Workload**: $S_p = 2,048$ prompt tokens, $S_o = 256$ generated tokens.
- **Total Prompt Tokens/sec**: $1,667 \times 2,048 \approx 3.41 \times 10^6 \text{ tokens/s}$.
- **Total Generation Tokens/sec**: $1,667 \times 256 \approx 4.27 \times 10^5 \text{ tokens/s}$.
- **Model Weight Footprint (70B in FP8)**: $70\text{ GB}$.
- **Prefill Compute Needs**:
  $$ \text{FLOPs/s} = 3.41 \times 10^6 \times 2 \times (70 \times 10^9) = 4.77 \times 10^{17} \text{ FLOPs/s} = 477 \text{ PFLOPs/s} $$
  Assuming an H100 achieves $600\text{ TFLOPs}$ effective prefill compute:
  $$ \text{Prefill GPUs Needed} = \frac{477 \times 10^{15}}{600 \times 10^{12}} \approx \mathbf{795 \text{ H100 GPUs}} $$
- **Decode Memory Bandwidth Needs**:
  Each decode GPU with TP=4 has $4 \times 3.35 \text{ TB/s} = 13.4 \text{ TB/s}$ bandwidth.
  Loading 70B model ($70\text{ GB}$) takes $\frac{70}{13.4 \times 10^3} \approx 5.22\text{ ms/step}$.
  With continuous batch size $B=128$, tokens/sec per 4-GPU replica $= \frac{128}{0.00522} \approx 24,500 \text{ tokens/s}$.
  $$ \text{Decode 4-GPU Replicas Needed} = \frac{427,000}{24,500} \approx 18 \text{ Replicas} = \mathbf{72 \text{ GPUs}} $$

---

## 3. Case Study 2: Proactive Autonomous Agent Runtime with Durable Execution

**Prompt**: *"Architect the backend platform for an AI assistant that executes multi-step web scraping, code execution, and financial transactions lasting up to 48 hours."*

```
Webhooks / Cron ──► [ Kafka Event Bus ] ──► [ Temporal Workflow Worker Pool ]
                                                    │
                 ┌──────────────────────────────────┼──────────────────────────────────┐
                 ▼                                  ▼                                  ▼
      [ Step 1: LLM Planner ]             [ Step 2: Tool Worker ]             [ Step 3: HITL Gating ]
      - Structured JSON via FSM           - Idempotency-Key Gateway           - Async SMS/App Push
      - Outlines / Pydantic               - gVisor / Firecracker Sandboxes    - Suspends Workflow
```

---

## 4. Case Study 3: Enterprise Continuous Self-Improving RAG Pipeline

```
Knowledge Corpus (PDFs, Notion, Confluence, GitHub)
        │
        ▼
[ Document Parser & Chunker ] (Semantic Chunking, 512 tokens + 10% overlap)
        │
        ├───► Dense Embedding (OpenAI text-embed-3) ──► [ Qdrant / Milvus Vector DB ]
        └───► Sparse Shingles Tokenizer ─────────────► [ Elasticsearch / BM25 ]
                                                              │
User Query ──► [ Query Expansion ] ───────────────────────────┤
                                                              ▼
                                               [ Reciprocal Rank Fusion ]
                                                              │ (Top-30 Candidates)
                                                              ▼
                                               [ Cross-Encoder Neural Reranker ]
                                                              │ (Top-5 High-Signal Passages)
                                                              ▼
                                               [ Context Compactor ] ──► LLM Generation
```

---

## 5. Deep Interview Interrogation Ladder

- **Level 1 (Concept)**: What is the purpose of load-balancing via affinity routing in LLM serving?
- **Level 3 (Math)**: Walk through the exact back-of-the-envelope calculation to size the number of GPUs needed for 1,000 QPS on a 70B model.
- **Level 5 (Serving Architecture)**: Why does Disaggregated Prefill/Decode achieve higher hardware efficiency than unified continuous batching?
- **Level 7 (RAG)**: Explain why Cross-Encoder neural rerankers are placed downstream of Hybrid Dense/Sparse retrieval rather than being used for initial search.
- **Level 9 (Reliability)**: In an enterprise RAG system serving 50,000 employees, document permissions change dynamically. How do you design real-time access-control filtering into vector search without rebuilding indexes?
- **Level 10 (Principal Engineering)**: Architect a globally distributed, multi-region agentic platform with active-active failover, KV cache synchronization across regions, and zero data loss under a full datacenter outage.

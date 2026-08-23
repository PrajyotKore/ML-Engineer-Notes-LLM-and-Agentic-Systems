# 10_AGENTIC_ML_SYSTEMS — Mathematical & Systems Engineering Reference

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide an exhaustive mathematical and architectural reference on Agentic Systems — covering chained reasoning failure mathematics, Grammar-Constrained Decoding via FSMs (Outlines/XGrammar), Model Context Protocol (MCP), Hybrid RAG with Reciprocal Rank Fusion (RRF), and Multi-Agent orchestration.

---

## 1. Agent Cognitive Architectures & The Execution Loop

An **Autonomous AI Agent** wraps an autoregressive LLM inside a state machine equipped with tools, memory, and reflection loops.

```
                  ┌──────────────────────────────────────────────┐
                  │             1. Perception & Context           │
                  │  System Prompt + User Request + Memory + MCP │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │            2. Reasoning (Thought)            │
                  │  Chain-of-Thought Planning & Decision Making │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         3. Structured Action (Tool Call)     │
                  │  FSM-Constrained JSON: {tool: ..., args: ...}│
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │            4. Deterministic Execution        │
                  │  Sandboxed API / Database / Code Execution   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │            5. Observation & Reflection       │
                  │  Evaluate Output -> Self-Correct or Finalize │
                  └──────────────────────────────────────────────┘
```

### 1.1 Core Agentic Patterns
1. **ReAct (Yao et al., 2022)**: Interleaves `Thought: ...`, `Action: ...`, and `Observation: ...` in an autoregressive stream.
2. **Plan-and-Solve (Wang et al., 2023)**: Generates a full execution DAG upfront, executing sub-tasks with specialized sub-agents.
3. **Reflexion (Shinn et al., 2023)**: Logs execution failures into episodic memory, prompting the model to reflect on *why* it failed before retrying.

---

## 2. Mathematical Foundations: The Chained Reasoning Failure Problem

Let an agent task require a sequence of $N$ interdependent steps (tool calls, API queries, reasoning jumps).  
Let $p_i \in [0, 1]$ denote the success probability of step $i$.

### 2.1 The Multiplicative Degradation Law
Assuming step failures are conditionally independent:

$$ \mathbf{P(\text{End-to-End Task Success}) = \prod_{i=1}^N p_i} $$

If all steps have identical accuracy $p$:
$$ P(\text{Success}) = p^N $$

```
  End-to-End Success Rate P(N)
  1.0 ┼────── p = 0.99 (99% per step) ───────► P(20) = 81.8%
      │      \
  0.5 ┼       \───── p = 0.95 (95% per step) ─► P(20) = 35.8%
      │        \
      │         \─── p = 0.90 (90% per step) ─► P(20) = 12.2%
    0 ┼──────────┴────────────────────────────► Number of Steps N
      0          5          10         15         20
```

#### The Engineering Implication:
Even if a model achieves a stellar $95\%$ accuracy per step ($p = 0.95$), a 20-step workflow succeeds only **$35.8\%$ of the time**.  
*Conclusion*: You **cannot** solve agent reliability purely by making the LLM smarter. You **must** architect deterministic recovery systems: **State Checkpoints, Reflection Loops, and Grammar-Constrained Logit Biasing**.

---

## 3. Structured Outputs & Grammar-Constrained Decoding (FSM / CFG)

Prompting an LLM to "output valid JSON" leads to catastrophic production failures: missing commas, hallucinated fields, unescaped quotes, or explanatory conversational filler.

### 3.1 Finite State Machine (FSM) Logit Biasing (Outlines / XGrammar / llguidance)

Any JSON Schema or Regular Expression can be compiled into a **Deterministic Finite Automaton (DFA)** $\mathcal{M} = (Q, \Sigma, \delta, q_0, F)$:
- $Q$: Set of states.
- $\Sigma$: Vocabulary of tokens $\mathcal{V}$.
- $\delta: Q \times \Sigma \to Q$: State transition function.
- $q_0$: Initial state.
- $F \subseteq Q$: Set of accepting states.

#### Mathematical Formulation of Logit Masking:
At step $t$, let the current FSM state be $s_t \in Q$.  
We compute an element-wise logit bias vector $M_t \in \mathbb{R}^{|\mathcal{V}|}$:

$$ M_t(v) = \begin{cases} 0 & \text{if } \delta(s_t, v) \text{ is a valid transition in } \mathcal{M} \\ -\infty & \text{otherwise} \end{cases} \quad \forall v \in \mathcal{V} $$

The modified logits $\tilde{z}_t$ fed into Softmax are:
$$ \tilde{z}_t = z_t + M_t \implies P_\theta(v \mid x_{<t}) = \frac{\exp(z_t(v) + M_t(v))}{\sum_{u \in \mathcal{V}} \exp(z_t(u) + M_t(u))} $$

```
FSM State: Parsing JSON Key: {"action": "
┌────────────────────────────────────────────────────────┐
│  Vocabulary Token  │  FSM Transition  │  Logit Bias M  │
├────────────────────┼──────────────────┼────────────────┤
│  "search_db"       │  Valid (-> Arg)  │       0        │
│  "calculate"       │  Valid (-> Arg)  │       0        │
│  "Sure! Here is.." │  INVALID         │     -inf       │
│  12345             │  INVALID         │     -inf       │
└────────────────────────────────────────────────────────┘
Result: 100% Mathematically Guaranteed Schema Adherence at Step 0.
```

---

## 4. Advanced Retrieval-Augmented Generation (RAG) & Memory Architecture

```
User Query ──► [ Query Transformer / Expansion ]
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[ Dense Vector Search (HNSW) ]  [ Sparse Keyword Search (BM25) ]
  Cosine Similarity Top-50        Probabilistic Term Frequency Top-50
        │                           │
        └─────────────┬─────────────┘
                      ▼
        [ Reciprocal Rank Fusion (RRF) ] ──► Top-20
                      │
                      ▼
        [ Cross-Encoder Neural Reranker ] ──► Top-5 Re-ranked Documents
                      │
                      ▼
        [ Context Window Compression ] ──► LLM Prompt
```

### 4.1 Hybrid Search: Dense Vectors + BM25 Okapi

1. **Dense Vector Search (Semantic Meaning)**:
   $$ \text{Cosine}(q, d) = \frac{q \cdot d}{\|q\|_2 \|d\|_2} $$
2. **Sparse BM25 Search (Exact Keyword / Entity Match)**:
   $$ \text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left( 1 - b + b \cdot \frac{|d|}{\text{avgdl}} \right)} $$
   Where $k_1 \approx 1.5, b \approx 0.75$, and $\text{IDF}(t) = \ln\left( \frac{N - n(t) + 0.5}{n(t) + 0.5} + 1 \right)$.

---

### 4.2 Reciprocal Rank Fusion (RRF) Mathematics

To merge heterogeneous rankings from Dense Search (ranks $r_{\text{dense}}(d)$) and Sparse Search (ranks $r_{\text{sparse}}(d)$) without calibrating arbitrary score distributions:

$$ \mathbf{\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}} $$

Where $M = \{\text{dense}, \text{sparse}\}$, $r_m(d) \in \{1, 2, \dots, K\}$ is the 1-indexed rank of document $d$ in system $m$, and $k = 60$ is a smoothing constant.

---

### 4.3 Cross-Encoder Neural Reranking
Bi-encoders (embedding search) compress an entire document into a single 1536-dimensional vector, losing fine-grained cross-token attention.  
A **Cross-Encoder** concatenates Query and Candidate Document $[q; d]$ into a single Transformer forward pass, computing all-to-all cross-attention:
$$ \text{Score}(q, d) = \sigma\left( \text{Transformer}([q; d]) W_{\text{score}} \right) $$

---

## 5. Tool Integration & Model Context Protocol (MCP)

### 5.1 Dynamic Tool Retrieval
When an enterprise platform possesses $T = 1,000+$ distinct tools, dumping all JSON schemas into the system prompt consumes $100\text{k}+$ context tokens and degrades tool selection accuracy.

- **Two-Tier Tool Selection**:
  1. Index all tool docstrings into a vector database.
  2. Embed the agent's current thought: $e_{\text{thought}} = \text{Embed}(t_{\text{current}})$.
  3. Retrieve the Top-$K$ ($K=5$) most relevant tool schemas via Cosine Similarity.
  4. Dynamically inject only those $5$ schemas into the active context window.

---

## 6. Python Implementation: FSM Logit Processor & Reciprocal Rank Fusion

```python
import torch
import numpy as np
from typing import List, Dict

def reciprocal_rank_fusion(dense_ranks: Dict[str, int], sparse_ranks: Dict[str, int], k: int = 60) -> List[tuple]:
    """
    Combines dense and sparse search rankings using Reciprocal Rank Fusion (RRF).
    """
    all_doc_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
    rrf_scores = {}

    for doc_id in all_doc_ids:
        score = 0.0
        if doc_id in dense_ranks:
            score += 1.0 / (k + dense_ranks[doc_id])
        if doc_id in sparse_ranks:
            score += 1.0 / (k + sparse_ranks[doc_id])
        rrf_scores[doc_id] = score

    # Sort descending by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs

class SimpleFSMConstrainedLogitsProcessor:
    """
    Applies -inf logit mask to tokens that violate allowed transitions.
    """
    def __init__(self, allowed_token_ids_per_state: Dict[int, List[int]]):
        self.allowed_tokens = allowed_token_ids_per_state

    def __call__(self, current_fsm_state: int, logits: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(logits, float('-inf'))
        valid_ids = self.allowed_tokens.get(current_fsm_state, [])
        mask[valid_ids] = 0.0
        return logits + mask

if __name__ == "__main__":
    # Test RRF
    dense = {"doc_1": 1, "doc_2": 2, "doc_3": 3}
    sparse = {"doc_2": 1, "doc_4": 2, "doc_1": 3}
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    print("RRF Combined Rankings:")
    for rank, (doc, score) in enumerate(fused, 1):
        print(f"  Rank {rank}: {doc} (Score: {score:.5f})")
    assert fused[0][0] == "doc_2", "Doc 2 should be ranked #1 due to strong consensus!"
    print("RRF validation passed successfully.")
```

---

## 7. Deep Interview Interrogation Ladder

- **Level 1 (Concept)**: What is the core loop of a ReAct agent?
- **Level 3 (Math)**: Show mathematically why an agent with 95% single-step tool accuracy fails more than 60% of the time on a 20-step task.
- **Level 5 (Mechanics)**: How does FSM logit biasing mathematically guarantee 100% valid JSON schema generation without prompting retries?
- **Level 7 (RAG)**: Derive the Reciprocal Rank Fusion (RRF) scoring formula. Why is RRF superior to simple weighted score addition when combining Dense Vector and BM25 search?
- **Level 9 (Tool Scale)**: When an agent has access to 2,000 corporate APIs, how do you architect a two-tier dynamic tool retrieval pipeline to keep prompt tokens bounded and avoid hallucinations?
- **Level 10 (Principal Engineering)**: Architect an enterprise-scale multi-agent system where a Supervisor Agent delegates tasks to specialized Worker Agents across distributed nodes. Detail the state handoff protocol, token-budget enforcement, and deadlock-prevention state machine.

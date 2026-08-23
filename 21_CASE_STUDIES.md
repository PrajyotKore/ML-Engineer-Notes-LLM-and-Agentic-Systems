# 21_CASE_STUDIES — Production Incident Root Cause Analyses (RCAs)

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide structured, battle-tested Root Cause Analyses (RCAs) for five high-stakes production incidents — detailing the symptom, diagnostic hypotheses, telemetry isolation, root cause, and permanent architectural remediation.

---

## Case Study 1: The Step 4,500 Training Loss Spike & NaN Divergence

```
Step 0 ────────► Step 4,499 (Loss = 1.25) ──► Step 4,500 (Loss = 8.92 -> NaN!)
                                                       │
                 ┌─────────────────────────────────────┴─────────────────────────────────────┐
                 ▼                                     ▼                                     ▼
        [ Hypothesis 1 ]                      [ Hypothesis 2 ]                      [ Hypothesis 3 ]
        Data batch corruption /               LR Scheduler ended warmup             FP16 Gradient Overflow /
        unmasked padding tokens               & hit maximum peak                    Unbounded Attention Scores
```

### 1. Diagnosis Protocol:
1. **Gradient Norm Telemetry**: Isolate the gradient norm before clipping. The metric spiked from $0.8$ to $142.0$ at step 4,500.
2. **Layer-Wise Gradient Inspection**: Isolate which layer generated the explosion. Layer 78 (final FFN down-projection) showed exploding gradients while early layers were normal.
3. **Data Inspection**: The batch at step 4,500 contained a corrupted UTF-8 string that was tokenized into an extreme sequence of 4,096 identical tokens, causing attention logits to saturate.
4. **Root Cause**: Unclipped gradients combined with FP16 underflow/overflow in the final layer.

### 2. Permanent Architectural Remediation:
- **Immediate**: Revert training to checkpoint at step 4,000. Switch training precision from FP16 to **BF16** (8-bit exponent matches FP32 dynamic range).
- **Long-Term**: Implement strict data pipeline validation filtering out repetitive sequence anomalies, and enforce global gradient norm clipping (`clip_grad_norm_ = 1.0`).

---

## Case Study 2: The Agent Infinite Tool-Calling Loop

- **Symptom**: Production alerts fire. An autonomous agent has been executing for 52 minutes on a single request, issuing 600 consecutive API calls and burning $1,200 in API tokens.
- **Trace Isolation**: OpenTelemetry trace shows the agent called `get_calendar_events`, received an HTTP 401 Unauthorized error, apologized in its `<think>` scratchpad, and called `get_calendar_events` again 600 times.
- **Root Cause**: The system relied on the non-deterministic LLM to decide when to stop retrying after an authorization failure, lacking a deterministic circuit breaker at the execution boundary.
- **Permanent Architectural Remediation**:
  - Implement a **Deterministic Circuit Breaker** at the tool gateway: cap identical consecutive tool errors to $3$.
  - Upon 3 consecutive failures, inject a hard system interruption token into context and trigger a **Compensation Transaction** via the Durable Workflow Engine.

---

## Case Study 3: The P99 TTFT Starvation Crisis

- **Symptom**: System-wide throughput (tokens/sec) is hitting targets, but P99 Time To First Token (TTFT) degrades to 18 seconds (users experience a frozen UI before generation begins).
- **Diagnosis**: Telemetry shows average prompt length is 800 tokens, but P99 prompt length is 45,000 tokens (users uploading entire legal PDFs). In continuous batching, a 45k prefill monopolizes GPU Tensor Cores for several seconds, blocking the prefill queue of all concurrent requests.
- **Permanent Architectural Remediation**:
  - Implement **Chunked Prefill** (cap prefill execution chunks to 512 tokens per iteration, interleaving with decode steps).
  - Deploy **Disaggregated Prefill and Decode (PD Split)** on dedicated nodes so prompt processing compute never starves decode workers.

---

## Case Study 4: The 15% Offline Eval Gain vs. 5% Online Retention Drop

- **Symptom**: A newly aligned model scored $15\%$ higher on the offline LLM-as-a-Judge benchmark. After a 10% Canary deployment, user retention dropped by $5\%$.
- **Diagnosis**: 
  - The offline LLM judge suffered from **Verbosity Bias**: the new model generated $3.2\times$ longer answers. The judge rewarded the length, but real-world users found the responses bloated, slow, and unhelpful.
  - Slower generations ($3.2\times$ tokens) increased user-perceived latency by $250\%$.
- **Permanent Architectural Remediation**:
  - Re-align the model using **Length-Penalized DPO** (penalizing response length in the preference loss).
  - Update the LLM-as-a-Judge benchmark to strictly penalize verbosity and evaluate pairwise rankings under length-normalized metrics.

---

## Case Study 5: Mixture-of-Experts All-to-All Straggler Stalls

- **Symptom**: During 512-GPU pre-training of an MoE model, Model FLOPs Utilization (MFU) drops from 58% to 22%.
- **Diagnosis**: `nsys` profiling reveals that GPUs are spending 45% of execution time blocked on `ncclAllToAll` collective barriers during expert token dispatch.
- **Root Cause**: Gating network routing collapse — 3 popular experts received $80\%$ of all tokens across the cluster, while other expert GPUs sat idle waiting for the overloaded GPUs to finish computation.
- **Permanent Architectural Remediation**:
  - Implement **DeepSeek-V3 Auxiliary-Loss-Free Expert Biases** ($b_i \leftarrow b_i + \gamma(1/N - f_i)$) to dynamically redirect token flow away from overloaded nodes.
  - Enforce an **Expert Capacity Factor (ECF = 1.2)** to drop or route overflow tokens to default shared experts.

# 21_CASE_STUDIES — Technical Reference

## 1. Role Relevance
Principal Engineers are hired to solve things no one else can. Interviewers will present vague, high-stakes production incidents. You must structure your diagnosis systematically.

## Case Study 1: The Training Loss Spike
**Scenario**: You are fine-tuning a 70B model on 64 H100s using FSDP and BF16. At step 4,500, the loss spikes from 1.2 to 8.5 and does not recover.
**Diagnosis Framework**:
1. *Hypothesis 1: Data Corruption*. Check the batch of data at step 4,500. Are there sequence length anomalies? Is there unmasked padding?
2. *Hypothesis 2: Learning Rate*. Check the LR scheduler. Did the warmup end and the LR peak exactly at step 4,500?
3. *Hypothesis 3: Numerical Instability*. Even with BF16, gradients can explode. Check the gradient norm logs.
**Mitigation**: If gradient norm spiked, apply aggressive gradient clipping. Revert to the step 4,000 checkpoint, reduce the LR slightly, and resume.

## Case Study 2: The Agent Infinite Loop
**Scenario**: Production alerts fire. An A1 agent has been executing for 45 minutes on a single user request.
**Diagnosis Framework**:
1. *Trace Isolation*: Pull the OpenTelemetry trace for that `workflow_id`.
2. *Observation*: The LLM calls `get_calendar_events`. The API returns `401 Unauthorized`. The LLM apologizes in its thought process and calls `get_calendar_events` again. It has done this 400 times.
3. *Root Cause*: The workflow engine does not have a semantic circuit breaker for identical consecutive failures.
**Mitigation (Immediate)**: Terminate the workflow.
**Mitigation (Long-Term)**: Introduce a Max Retries limit at the System boundary (not trusting the LLM to stop itself). Return a hard `TERMINATE` token to the LLM if a tool fails 3 times.

## Case Study 3: The Latency Mystery
**Scenario**: Inference throughput (Tokens/sec) is hitting targets, but Time To First Token (TTFT) has a massive P99 tail (some users wait 15 seconds before the agent starts typing).
**Diagnosis Framework**:
1. *Hypothesis 1: Queueing*. Check the router queue length. If it's long, the cluster is under-provisioned.
2. *Hypothesis 2: Massive Prefills*. Check the input sequence lengths. Are some users sending 100k token prompts?
**Root Cause**: Continuous batching treats prefill and decode differently. If a 100k prefill enters the batch, it monopolizes the Tensor Cores, starving the decode steps of all other users in the batch.
**Mitigation**: Implement **Chunked Prefill**. Break the 100k prefill into chunks of 4k tokens and interleave them with decode steps to guarantee bounded latency for all users.

## Case Study 4: Evaluation Mismatch
**Scenario**: The offline LLM-as-a-judge scores the new V2 model 15% higher than V1. V2 is deployed, but user retention drops by 5%.
**Diagnosis Framework**:
1. *Hypothesis 1: Judge Bias*. Did the LLM judge suffer from length bias? Check the average output length of V2 vs V1. (If V2 is 3x longer, the judge loved it, but humans hated reading it).
2. *Hypothesis 2: Latency Regression*. Longer outputs = higher perceived latency.
3. *Hypothesis 3: Distribution Shift*. The offline Golden Dataset is 6 months old and mostly coding tasks. Current production users are mostly asking for creative writing.
**Mitigation**: Rewrite the LLM Judge prompt to strictly penalize verbosity. Sample 1,000 live queries from yesterday to build a new Golden Dataset. Roll back to V1 immediately.

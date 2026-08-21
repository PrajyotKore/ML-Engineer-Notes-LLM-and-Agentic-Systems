# 19_LEADERSHIP_AND_TECHNICAL_JUDGMENT — Technical Reference

## 1. Role Relevance
At the Technical Lead level, you are evaluated heavily on *judgment*, not just knowledge. The interviewer is asking: "If I give this person a $5M compute budget and a team of 5 engineers, will they build the right thing?" 
You must demonstrate the ability to navigate trade-offs, say "no" to hype, and focus on product outcomes.

## 2. Core Trade-off Matrix

### A. Full Fine-Tuning (SFT) vs LoRA vs Prompt Engineering
- **Prompt Engineering**: Do this first. Fastest time to market. Zero training cost. High latency cost (long context), high inference cost.
- **LoRA**: Do this when prompt engineering hits a ceiling, or when context limits are exhausted by massive few-shot examples. Cheap training, no inference overhead (if merged).
- **Full SFT**: Do this only when the model's fundamental representation needs to change (e.g., teaching a new language, completely altering the conversational tone), and you have massive, high-quality data.

### B. Large Model (70B) vs Small Model (8B) + RAG
- **70B**: Highly capable, but expensive and high TTFT/TPOT. Use for complex, multi-step planning (the Router/Planner Agent).
- **8B + RAG**: Fast, cheap. Use for extraction, summarization, and basic tool execution. At A1, the vast majority of execution steps should be routed to a small model to maintain unit economics.

### C. Build vs Buy
- Do not build a vector database from scratch. Use Pinecone or pgvector.
- Do not build a durable execution engine from scratch. Use Temporal.
- *Do* build the core agent orchestration loop, the evaluation pipeline, and the SFT data flywheel, because these are A1's proprietary IP.

## 3. Incident Leadership Framework
When a major production incident occurs (e.g., "The A1 agent just hallucinated and emailed a user's boss an empty draft."):
1. **Mitigate**: Stop the bleeding immediately. Revert the model to the last known good version or disable the email tool globally. Do not try to debug the root cause yet.
2. **Communicate**: Inform stakeholders of the impact and mitigation.
3. **Investigate**: Pull the OpenTelemetry trace. Find the exact prompt and LLM output.
4. **Root Cause**: Determine why the guardrail failed (e.g., the safety model timed out and failed open instead of closed).
5. **Prevent**: Add the trajectory to the regression test suite. Update the durable workflow to fail *closed* on guardrail timeouts.

## 4. Managing Complexity (The "No" Framework)
Principal engineers actively fight complexity.
- *Researcher*: "Let's implement a dynamic MoE router with RLHF!"
- *Tech Lead*: "Our current offline evaluation accuracy is 85%. Before we introduce training complexity, let's look at the data. 50% of the failures are due to bad JSON formatting. I'm going to implement strict JSON schema enforcement at the system boundary instead. That will get us to 92% accuracy with zero training cost."

## 5. Architectural Judgment Scenarios

**Scenario 1:** The team wants to use continuous RLHF in production based on user thumbs up/down.
**Your Response:** "RLHF is notoriously unstable and susceptible to reward hacking. If we train continuously, a malicious user group could poison the model. Instead, we should log the thumbs up/down, run an offline pipeline to filter for high-quality trajectories, use an LLM-as-a-judge to verify them, and run standard SFT on the filtered dataset."

**Scenario 2:** Inference costs are too high.
**Your Response:** "I would attack this hierarchically. 
1. Cache: Are we caching identical system prompts? (Prefix caching).
2. Routing: Can we route simple tasks to an 8B model instead of the 70B model?
3. Systems: Are we using PagedAttention and continuous batching to maximize GPU occupancy?
4. Math: Can we quantize the KV cache to FP8 without degrading evaluation metrics?
We measure the engineering effort vs. cost reduction for each and prioritize."

## 6. Interview Interrogation
- *Level 5*: When would you choose to NOT use Machine Learning for a problem?
- *Level 8*: Walk me through how you decide when to promote a staging model to production.
- *Level 10*: A1 has 6 months of runway to prove the proactive assistant works. You have 3 ML engineers. Lay out your roadmap for exactly what you build in Month 1, Month 3, and Month 6.

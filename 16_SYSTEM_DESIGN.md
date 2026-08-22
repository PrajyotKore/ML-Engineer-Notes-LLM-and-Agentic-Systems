# 16_SYSTEM_DESIGN — Technical Reference

## 1. Role Relevance
A ML Engineer (LLM & Agentic Systems) does not just train models; they architect the platforms that serve them. In an interview, you will likely be asked to design the end-to-end infrastructure for a proactive AI assistant on a whiteboard. You must demonstrate mastery over APIs, data models, control flow, scaling, and cost.

## 2. Core System Design Framework
When given a system design prompt, strictly follow this structure:
1. **Requirements Clarification**: (Functional vs Non-Functional). "Do we need streaming?", "What is the P99 latency target?", "What is our QPS?"
2. **Back-of-the-Envelope Math**: Calculate VRAM needs, network bandwidth, and storage.
3. **High-Level API Design**: Define the gRPC/REST endpoints.
4. **Data Model**: Schemas for databases (Postgres, Redis, VectorDB).
5. **High-Level Architecture (Box Drawing)**.
6. **Deep Dives**: (Bottlenecks, Scaling, Reliability).

## 3. Case 1: Large-Scale LLM Inference Platform
**Prompt**: "Design a platform to serve a 70B parameter model at 10,000 requests per minute with strict P99 latency bounds."

### 1. Math
- 70B Model in BF16 = 140GB.
- 10k RPM $\approx$ 166 QPS.
- KV Cache: Assume 2K input tokens, 500 output tokens. Requires $\sim$ 2.5GB per request.
- One 8xH100 node has 640GB VRAM. 140GB for weights $\rightarrow$ 500GB for KV Cache.
- Max batch size per node $\approx 200$.

### 2. Architecture
1. **API Gateway (Envoy/Nginx)**: Terminates TLS, handles rate limiting.
2. **Router / Load Balancer**: Tracks the KV cache state across the fleet. Implements **Affinity Routing** (sending requests to nodes that already have the system prompt cached).
3. **Inference Workers (vLLM)**: Nodes running Tensor Parallelism (TP=4). Uses Continuous Batching and PagedAttention.
4. **Autoscaler (KEDA)**: Monitors the queue length. If queue TTFT exceeds 500ms, provisions new GPU nodes.

### 3. Deep Dive: Handling Spikes
If a popular event triggers a massive spike in QPS, spinning up new H100s takes 5+ minutes (pulling the 140GB image). We must implement **Shedding/Admission Control** at the Router to return 429 Too Many Requests, rather than accepting them and letting the KV cache OOM or TTFT spike to 30 seconds.

## 4. Case 2: Durable Agent Runtime
**Prompt**: "Design the backend for a proactive assistant that books flights based on monitoring a user's inbox."

### 1. Architecture
1. **Event Bus (Kafka)**: Ingests webhooks from the email provider.
2. **Workflow Engine (Temporal)**: Manages the state machine.
3. **LLM Node**: Pure function. Takes (State, New Email) $\rightarrow$ Outputs (Action).
4. **Tool Execution Node**: Parses the Action, validates against JSON schema, makes the HTTP call to the Airline API.
5. **State DB (PostgreSQL)**: Stores the serialized execution history.

### 2. Deep Dive: Reliability
- **Idempotency**: The Tool Execution Node passes an `Idempotency-Key: hash(workflow_id + step_id)` to the Airline API.
- **Recovery**: If the LLM Node crashes, the Workflow Engine detects the timeout and safely retries. If the Tool Node crashes, the Workflow Engine replays the event history from Postgres and retries the tool call safely because of the idempotency key.

## 5. Case 3: ML Evaluation Platform
**Prompt**: "Design a system to continuously evaluate the performance of our agent in production."

### 1. Architecture
1. **Log Collector (Fluentd)**: Collects asynchronous traces from the Inference and Agent nodes.
2. **Data Lake (S3/Iceberg)**: Stores petabytes of historical trajectories.
3. **Evaluation Job Queue (Celery/SQS)**: Batches recent trajectories.
4. **LLM-as-a-Judge Fleet**: A dedicated cluster of cheap, fast GPUs running an 8B model to score trajectories for safety, hallucination, and task completion.
5. **Metrics Dashboard (Grafana)**: Displays the moving average of evaluation scores.

### 2. Deep Dive: Preventing Eval Drift
To prevent the LLM judge from making systemic errors, we route 1% of the trajectories to a human labeling queue (Scale AI/Labelbox). We continuously calculate the correlation between the LLM Judge scores and the Human scores. If correlation drops below 0.8, we pause automatic deployments and update the Judge's prompt.

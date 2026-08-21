# 13_PRODUCTION_ML — Technical Reference

## 1. Role Relevance
For an A1 Technical Lead, ML is a software engineering discipline. Training a model in a notebook is useless if it cannot be versioned, deployed safely, rolled back instantly, and monitored for degradation. You must own the entire ML lifecycle from raw data to production serving.

## 2. Prerequisites
- CI/CD pipelines (GitHub Actions, Jenkins).
- Containerization (Docker, Kubernetes).
- Model Serving (vLLM, TensorRT-LLM, Triton).

## 3. First Principles
Production ML (MLOps) applies DevOps principles to machine learning. It ensures reproducibility (Code + Data + Hyperparameters = Identical Model), deployment safety (Canarying), and continuous monitoring (Distribution Drift).

## 4. Mechanistic Breakdown
### The MLOps Lifecycle
1. **Model Registry**: Centralized store (e.g., MLflow, Weights & Biases) containing versioned artifacts, metadata, and evaluation metrics for every model trained.
2. **Experiment Tracking**: Logging every single hyperparameter, loss curve, and gradient norm during training. If a model performs well, you must be able to recreate exactly how it was built.
3. **Continuous Deployment (CD)**: Automatically deploying a model to a staging environment once it passes the offline evaluation pipeline.
4. **Safe Rollout**: Shifting live traffic slowly to the new model (Canary) and rolling back instantly if latency spikes or errors increase.

## 5. Mathematical Foundations
### Detecting Distribution Drift
Over time, the inputs a model sees in production change (e.g., a new popular slang, a new API format). We measure this drift using the Kullback-Leibler (KL) Divergence or Population Stability Index (PSI) between the training distribution $P$ and the production distribution $Q$.

$$ D_{KL}(P || Q) = \sum_{x \in X} P(x) \log \left( \frac{P(x)}{Q(x)} \right) $$

If the KL divergence exceeds a threshold, an automated alert triggers a retraining job.

## 6. Implementation
**Canary Deployment Strategy:**
```yaml
# Kubernetes / Istio Traffic Routing
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: a1-agent-router
spec:
  hosts:
  - a1-agent.production
  http:
  - route:
    - destination:
        host: model-v1 # Old Model
      weight: 95
    - destination:
        host: model-v2 # New Canary Model
      weight: 5
```

## 7. Computational Complexity
- **Shadow Inference**: Running a model in shadow mode means doubling your inference compute costs, as every request is processed by both v1 and v2. This requires highly elastic GPU provisioning.

## 8. Hardware / GPU Behavior
- **Containerization**: GPUs require specific drivers (CUDA, cuDNN) to be mounted into the Docker container (NVIDIA Container Toolkit). A mismatch between the host CUDA version and the container's PyTorch compiled version will cause silent performance degradation or crashes.

## 9. Production Architecture
**The A1 Rollback Mechanism:**
Because A1 serves long-running agent workflows, rolling back a model mid-workflow is dangerous.
1. V1 model starts Workflow A.
2. SRE rolls back from V2 to V1.
3. *Affinity Routing*: The system must ensure that workflows started on V1 continue on V1 until completion, while new workflows are routed to V2, avoiding "Split Brain" agent logic.

## 10. Scalability & Bottlenecks
- **Artifact Size**: A 70B model checkpoint is 140GB. Pulling this container image across 1,000 Kubernetes nodes takes hours. We use peer-to-peer image distribution (like Dragonfly or Kraken) or mount weights directly from S3 via fast network file systems to reduce pod startup time.

## 11. Failure Modes
- **Silent Degradation**: The model's API contract doesn't change, no exceptions are thrown, latency is fine, but the agent's actual success rate drops by 10%. Only caught by LLM-as-a-judge monitoring or user telemetry.
- **Dependency Hell**: A small library update in the training environment is missing in the production inference container, causing subtle tokenization mismatches.

## 12. Debugging
- **Reproducibility Crisis**: "It worked on my machine." To debug, you must lock all random seeds (`torch.manual_seed`), lock dependencies (`requirements.txt` hashes), and ensure the exact same PyTorch and CUDA versions are used.

## 13. Principal-Level Reasoning
"I do not allow manual deployments of models at A1. Every model must be registered via an automated CI pipeline. When it passes the offline eval suite, it is automatically deployed to a Shadow environment. We collect 24 hours of Shadow data, evaluate the difference, and only then allow a 1% Canary rollout. If P99 latency increases by 50ms during Canary, it automatically rolls back."

## 14. Interview Interrogation
- *Level 2*: What is the purpose of a Model Registry?
- *Level 4*: Explain the difference between A/B testing, Canarying, and Shadow mode.
- *Level 7*: How does KL Divergence help monitor production models?
- *Level 9*: Your V2 model was deployed and latency immediately spiked 3x. The model architecture is identical to V1. What infrastructure metrics do you check? (Answer: Check batch size, KV cache hit rate, or if Tensor Cores were disabled due to driver mismatch).
- *Level 10*: Architect the MLOps pipeline for A1 to continuously fine-tune the agent on user feedback and deploy it nightly with zero downtime.

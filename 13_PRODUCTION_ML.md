# 13_PRODUCTION_ML — Mathematical & MLOps Engineering Reference

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide an exhaustive reference on production machine learning and MLOps — covering mathematical drift detection (PSI, KL Divergence, KS Test), safe rollout architectures (Canary, Shadow, Blue-Green), session pinning during rolling deployments, and automated CI/CD model promotion pipelines.

---

## 1. Mathematical Foundations of Distribution Drift Detection

In production, user queries and data distributions change continuously over time (Covariate Shift: $P(X)$ changes while $P(Y \mid X)$ remains fixed; Concept Drift: $P(Y \mid X)$ changes).

```
Training Distribution P_train(X) ────────► Model Frozen
                                               │
                                               ▼
Production Distribution P_prod(X) ───────► Measure Drift: PSI, KL Divergence, KS-Test
                                               │
                                               ▼ (If PSI > 0.25)
                                          Trigger Automated Retraining Pipeline
```

### 1.1 Population Stability Index (PSI)

Let the continuous feature or token distribution be binned into $K$ discrete quantile buckets.  
Let $E_i$ be the expected proportion of samples in bucket $i$ from the baseline/training distribution ($P$), and $A_i$ be the actual proportion of samples in bucket $i$ from production ($Q$).

$$ \mathbf{\text{PSI} = \sum_{i=1}^K (A_i - E_i) \cdot \ln\left( \frac{A_i}{E_i} \right)} $$

#### Mathematical Properties of PSI:
- Each term $(A_i - E_i) \ln(A_i / E_i) \geq 0$ because if $A_i > E_i$, then $\ln(A_i / E_i) > 0$, and if $A_i < E_i$, then $\ln(A_i / E_i) < 0$.
- **Symmetric Metric**: PSI represents the symmetric KL divergence: $\text{PSI} = D_{KL}(A \parallel E) + D_{KL}(E \parallel A)$.
- **Standard Industry Thresholds**:
  - $\text{PSI} < 0.10$: **No Significant Shift** (Baseline stable).
  - $0.10 \leq \text{PSI} < 0.25$: **Moderate Drift** (Warn / Investigate).
  - $\text{PSI} \geq 0.25$: **Actionable Severe Drift** (Block canary deployment / Trigger retraining).

---

### 1.2 Two-Sample Kolmogorov-Smirnov (KS) Test

For continuous metrics (e.g. prompt token lengths, latency distributions):
Let $F_{\text{train}}(x)$ and $F_{\text{prod}}(x)$ be empirical cumulative distribution functions (eCDFs).

The KS statistic $D$ is the supremum distance between the two eCDFs:
$$ \mathbf{D = \sup_x |F_{\text{train}}(x) - F_{\text{prod}}(x)|} $$

Reject the null hypothesis (distributions are identical) at significance level $\alpha = 0.05$ if:
$$ D > c(\alpha) \sqrt{\frac{n_1 + n_2}{n_1 n_2}} \quad \left(c(0.05) = 1.36\right) $$

---

## 2. Production Deployment Strategies: Canary, Shadow, and Blue-Green

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Blue-Green Deployment: Instant Cutover (100% Blue -> 100% Green)          │
│    - Zero downtime, but maximum blast radius if Green has a hidden bug!     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Canary Deployment: Incremental Traffic Shifting (1% -> 5% -> 25% -> 100%)│
│    - Minimizes blast radius, monitors P99 latency and error rates online    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Shadow / Dark Launch: Dual-Inference (100% Live -> V1, Mirror -> V2)     │
│    - V2 runs silently on live traffic; outputs compared offline (Zero Risk) │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Session Pinning & Affinity Routing in Agentic Systems

In multi-step long-running agent workflows:
If a rolling deployment updates Model V1 to Model V2 while an agent is at Step 3 of a 10-step plan:
- **The Split-Brain Failure**: Model V2 receives the intermediate context, interprets tool schemas slightly differently, and emits incompatible action arguments, crashing the workflow.

#### The Architectural Solution:
- **Affinity Session Pinning**: The API Gateway inspects the `workflow_id`.
- Active workflows remain strictly pinned to Model V1 replicas until completion.
- New workflow requests are routed to the Model V2 Canary.

```
Incoming Request ──► [ Istio / Envoy Router ]
                            │
            ┌───────────────┴───────────────┐
            ▼ (Existing Workflow ID)         ▼ (New Session)
    [ Model V1 Replicas ]           [ Model V2 Canary (5%) ]
    (Pinned until completion)       (Evaluated on new traffic)
```

---

## 4. Python Implementation: Population Stability Index (PSI) Calculator

```python
import numpy as np

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10, eps: float = 1e-4) -> float:
    """
    Calculates the Population Stability Index (PSI) between baseline and production distributions.
    """
    # Define quantile bin boundaries based on expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    # Compute frequency counts in each bucket
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to proportions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Apply epsilon smoothing to prevent log(0) or division by zero
    expected_pct = np.clip(expected_pct, eps, 1.0)
    actual_pct = np.clip(actual_pct, eps, 1.0)

    # PSI = ∑ (Actual - Expected) * ln(Actual / Expected)
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)

if __name__ == "__main__":
    np.random.seed(42)
    baseline_tokens = np.random.normal(loc=500, scale=100, size=10000)
    stable_production = np.random.normal(loc=505, scale=102, size=5000)
    drifted_production = np.random.normal(loc=750, scale=150, size=5000)

    psi_stable = calculate_psi(baseline_tokens, stable_production)
    psi_drift = calculate_psi(baseline_tokens, drifted_production)

    print(f"PSI (Stable Production): {psi_stable:.4f} (Status: STABLE)")
    print(f"PSI (Drifted Production): {psi_drift:.4f} (Status: SEVERE DRIFT)")

    assert psi_stable < 0.10, "Expected stable PSI < 0.10!"
    assert psi_drift > 0.25, "Expected drifted PSI > 0.25!"
    print("PSI Drift Engine Verified Successfully.")
```

---

## 5. Deep Interview Interrogation Ladder

- **Level 1 (Concept)**: What is the purpose of a Model Registry in MLOps?
- **Level 3 (Math)**: Write the mathematical formula for Population Stability Index (PSI) and explain the standard threshold values.
- **Level 5 (Deployments)**: Compare Blue-Green, Canary, and Shadow (Dark Launch) deployments in terms of cost, latency, and blast radius.
- **Level 7 (Drift)**: Explain the mathematical difference between Covariate Shift and Concept Drift.
- **Level 9 (Agent Workflows)**: Why does rolling model deployment break long-running agent workflows, and how does session-affinity routing prevent split-brain state?
- **Level 10 (Principal Engineering)**: Design an automated MLOps CI/CD platform that continuously fine-tunes a 70B agent on high-quality production logs, validates against a 5,000-prompt offline suite, provisions a Shadow deployment, evaluates PSI drift, and promotes to Canary with zero human intervention.

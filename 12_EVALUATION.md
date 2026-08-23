# 12_EVALUATION — Mathematical & Statistical Reference

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide an exhaustive mathematical and statistical reference on evaluating LLMs and Agentic systems — covering Two-Proportion Z-Tests, Sample Size calculations, Bradley-Terry ELO rating derivations, LLM-as-a-Judge debiasing, and benchmark harnesses (SWE-bench / GAIA).

---

## 1. The Three Tiers of ML Evaluation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 1. Model Evaluation (Offline Static Weights)                │
│  Perplexity, MMLU-Pro, HumanEval, GSM8K, MATH                               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴──────────────────────────────────────┐
│                 2. System / Agent Evaluation (Offline Dynamics)              │
│  SWE-bench, GAIA, Tool-Calling Precision/Recall, Context Retrieval Precision│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴──────────────────────────────────────┐
│                 3. Product Evaluation (Online Live Traffic)                 │
│  Task Completion Rate, User Retention, Inter-Token Latency, Cost per Task   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Statistical Significance in A/B Testing: Mathematics & Sample Sizing

When testing if Model $B$ improves agent success rate over Model $A$ ($p_B > p_A$):

### 2.1 The Two-Proportion $Z$-Test Formulation

Let:
- $n_A, n_B$: Sample size of evaluated trajectories for Model $A$ and Model $B$.
- $X_A, X_B$: Number of successfully completed tasks.
- $\hat{p}_A = \frac{X_A}{n_A}$, $\hat{p}_B = \frac{X_B}{n_B}$: Sample proportions.
- $\hat{p} = \frac{X_A + X_B}{n_A + n_B}$: Pooled proportion under the null hypothesis $H_0: p_A = p_B$.

The standard error of the difference is:
$$ \text{SE} = \sqrt{\hat{p}(1 - \hat{p}) \left( \frac{1}{n_A} + \frac{1}{n_B} \right)} $$

The test statistic $Z$ is:
$$ \mathbf{Z = \frac{\hat{p}_B - \hat{p}_A}{\text{SE}} = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1 - \hat{p}) \left( \frac{1}{n_A} + \frac{1}{n_B} \right)}}} $$

- For significance level $\alpha = 0.05$ (two-tailed), reject $H_0$ if $|Z| \geq 1.96$ ($p\text{-value} \leq 0.05$).

---

### 2.2 Sample Size Estimation for Minimum Detectable Effect (MDE)

To detect an absolute improvement $\delta = p_B - p_A$ with statistical power $1 - \beta = 0.80$ and significance $\alpha = 0.05$:

$$ \mathbf{N = \frac{\left( Z_{\alpha/2} \sqrt{2 \bar{p}(1 - \bar{p})} + Z_\beta \sqrt{p_A(1 - p_A) + p_B(1 - p_B)} \right)^2}{\delta^2}} $$

Where $Z_{\alpha/2} = 1.96$ and $Z_\beta = 0.8416$.

#### Concrete Production Sizing Example:
- Baseline agent success rate: $p_A = 0.80$.
- We want to detect a $+2\%$ improvement ($p_B = 0.82 \implies \delta = 0.02$).
$$ N \approx \frac{(1.96 + 0.84)^2 \cdot 2 \cdot (0.80 \cdot 0.20)}{(0.02)^2} = \frac{7.84 \cdot 0.32}{0.0004} \approx \mathbf{6,272 \text{ trajectories per model}} $$

*Insight*: You **cannot** declare model victory by testing on 50 or 100 prompts. You mathematically need over $6,000$ trials to distinguish a true $2\%$ gain from random noise.

---

## 3. Bradley-Terry ELO Rating Mechanics (LMSYS Chatbot Arena)

In pairwise blind evaluation, model strength is represented on an ELO scale $R \in \mathbb{R}$.

### 3.1 Expected Win Probability
The expected probability that Model $A$ defeats Model $B$ is governed by the logistic curve:

$$ \mathbf{E_A = \mathbb{P}(A \succ B) = \frac{1}{1 + 10^{(R_B - R_A)/400}} = \frac{1}{1 + e^{(R_B - R_A) / \xi}}} \quad \left(\xi = \frac{400}{\ln 10} \approx 173.7\right) $$

```
  Expected Win Probability E_A
  1.0 ┼                                        ┌─────────────
      │                                       /
  0.5 ┼───────────────────┼───────────────────
      │                  / (Equal Rating: E_A = 0.5)
  0.0 ┼─────────────────┘
      └───────────────────┼───────────────────► Rating Difference (R_A - R_B)
                        R_A = R_B
```

### 3.2 Online ELO Update Rule
After observing the actual match outcome $S_A \in \{1.0 \text{ (Win)}, 0.5 \text{ (Tie)}, 0.0 \text{ (Loss)}\}$:

$$ \mathbf{R_A \leftarrow R_A + K \cdot (S_A - E_A)} $$
$$ \mathbf{R_B \leftarrow R_B + K \cdot (S_B - E_B)} $$

Where $K$ is the update step size factor (typically $K = 32$).

---

## 4. LLM-as-a-Judge: Systematic Biases & Debiasing Mathematics

Using an LLM (e.g. GPT-4) to grade model responses is fast and scalable, but prone to systematic biases.

### 4.1 Taxonomy of Judge Biases
1. **Position Bias (First-Order Favoritism)**: The judge strongly prefers whichever candidate response appears first in the prompt.
2. **Verbosity Bias**: The judge consistently awards higher scores to verbose, lengthy responses, even when concise answers are requested.
3. **Self-Preference Bias**: Models evaluate responses generated by their own family higher than competitors.

### 4.2 Position Debiasing: Permuted Pairwise Scoring
For every evaluation pair $(y_A, y_B)$, evaluate the judge under **both orderings**:

$$ \text{Trial 1}: \text{Judge}(x, y_A, y_B) \to S_1 \in \{A, B, \text{Tie}\} $$
$$ \text{Trial 2}: \text{Judge}(x, y_B, y_A) \to S_2 \in \{B, A, \text{Tie}\} $$

#### Symmetrized Score Matrix:
$$ S(A, B) = \begin{cases} 1.0 & \text{if } S_1 = A \text{ and } S_2 = A \text{ (True Win)} \\ 0.0 & \text{if } S_1 = B \text{ and } S_2 = B \text{ (True Loss)} \\ 0.5 & \text{if } S_1 \neq S_2 \text{ (Position Inconsistency / Tie)} \end{cases} $$

---

## 5. Python Implementation: Statistical A/B Significance & ELO Calculator

```python
import numpy as np
from scipy import stats

def two_proportion_z_test(success_a: int, n_a: int, success_b: int, n_b: int) -> dict:
    """
    Computes Two-Proportion Z-Test for A/B testing evaluation.
    """
    p_a = success_a / n_a
    p_b = success_b / n_b
    p_pooled = (success_a + success_b) / (n_a + n_b)
    
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
    z_stat = (p_b - p_a) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        "p_A": p_a,
        "p_B": p_b,
        "absolute_delta": p_b - p_a,
        "z_statistic": z_stat,
        "p_value": p_val,
        "is_significant_95": p_val < 0.05
    }

def update_elo(rating_a: float, rating_b: float, outcome_a: float, k: float = 32.0) -> tuple:
    """
    Computes Bradley-Terry ELO rating updates.
    outcome_a: 1.0 for A win, 0.5 for tie, 0.0 for B win.
    """
    e_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    e_b = 1.0 - e_a
    
    new_rating_a = rating_a + k * (outcome_a - e_a)
    new_rating_b = rating_b + k * ((1.0 - outcome_a) - e_b)
    return new_rating_a, new_rating_b

if __name__ == "__main__":
    # A/B Test Validation: 820/1000 vs 860/1000
    res = two_proportion_z_test(success_a=820, n_a=1000, success_b=860, n_b=1000)
    print(f"A/B Test Results: Delta = +{res['absolute_delta']*100:.1f}%, Z = {res['z_statistic']:.3f}, p = {res['p_value']:.4f}")
    assert res['is_significant_95'], "Expected statistically significant result!"
    
    # ELO Validation
    r_a, r_b = 1500.0, 1500.0
    r_a, r_b = update_elo(r_a, r_b, outcome_a=1.0)
    print(f"Post-Match ELO: Model A = {r_a:.1f}, Model B = {r_b:.1f}")
    assert r_a > 1500.0 and r_b < 1500.0
    print("Statistical Evaluation Modules Verified Successfully.")
```

---

## 6. Deep Interview Interrogation Ladder

- **Level 1 (Concept)**: What is the difference between an offline benchmark and an online A/B test?
- **Level 3 (Math)**: Write the formula for the Two-Proportion Z-Test and explain how to calculate the minimum sample size needed to detect a $1\%$ improvement.
- **Level 5 (Mechanics)**: How does the Bradley-Terry ELO model update ratings after a blind pairwise evaluation match?
- **Level 7 (Debiasing)**: Why do LLM judges suffer from position bias, and how does order permutation mathematically eliminate it?
- **Level 9 (Harness Design)**: Design an end-to-end evaluation harness for SWE-bench (code editing in real GitHub repos). How do you isolate execution environments and prevent metric contamination?
- **Level 10 (Principal Engineering)**: Your offline LLM-as-a-judge reports a $15\%$ improvement in agent task completion, but the online canary deployment shows a $3\%$ drop in user satisfaction. Walk through your statistical and systems investigation step-by-step.

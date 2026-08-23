# 05_POST_TRAINING — Mathematical & Mechanistic Reference

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide an exhaustive mathematical and algorithmic breakdown of Post-Training — spanning SFT, LoRA/QLoRA, the step-by-step derivation of Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO) for reasoning models (DeepSeek-R1 style).

---

## 1. Supervised Fine-Tuning (SFT) Mathematics

Given a dataset of instruction-response pairs $\mathcal{D}_{\text{SFT}} = \{(x^{(i)}, y^{(i)})\}_{i=1}^N$, where $x$ is the prompt tokens $(x_1, \dots, x_{S_p})$ and $y$ is the response tokens $(y_1, \dots, y_{S_r})$.

### 1.1 The Masked Cross-Entropy Objective
We only compute gradients over the **response tokens** $y$, masking out the prompt tokens $x$ from the loss calculation:

$$ \mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \sum_{t=1}^{S_r^{(i)}} \log P_\theta\left(y_t^{(i)} \mid x^{(i)}, y_{<t}^{(i)}\right) $$

```
Prompt Tokens (x_1, x_2, x_3)           Response Tokens (y_1, y_2, y_3, <EOS>)
[ Loss Mask = 0, Gradient = 0 ]        [ Loss Mask = 1, Gradient = (p_t - target) ]
```

*Why Masking Prompt Tokens is Mandatory*: If prompt tokens are included in the loss, the model allocates substantial capacity to predicting user syntax, grammar, and prompt templates rather than mastering instruction-following and tool formatting.

---

## 2. Parameter-Efficient Fine-Tuning: LoRA, QLoRA, and DoRA

### 2.1 Low-Rank Adaptation (LoRA)

Let $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ be a frozen pre-trained weight matrix.  
LoRA decomposes the weight update $\Delta W$ into the product of two low-rank matrices:

$$ W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A $$

Where:
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ is initialized from Gaussian distribution $\mathcal{N}\left(0, \sigma^2 = \frac{1}{r}\right)$.
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$ is initialized to strictly **zero**: $B = 0$.
- $r \ll \min(d_{\text{in}}, d_{\text{out}})$ is the intrinsic rank (typically $r \in [8, 64]$).
- $\alpha$ is a constant scaling hyperparameter (typically $\alpha = 2r \implies \text{scaling} = 2$).

```
                Input Vector x (dim d_in)
                    │
            ┌───────┴───────┐
            ▼               ▼
     Frozen W_0        Matrix A (dim r x d_in)
     (d_out x d_in)         │
            │          Matrix B (dim d_out x r) [Init = 0]
            │               │
            │          Scale (α / r)
            │               │
            └───────┬───────┘
                    ▼
           Output h = W_0 x + (α/r) B A x
```

#### Why $B = 0$ Initialization is Mandatory:
At step $t=0$:
$$ \Delta W = \frac{\alpha}{r} (0) A = 0 \implies W = W_0 $$
This guarantees that training begins exactly from the pre-trained model's output without introducing catastrophic initial perturbation.

#### Gradient Derivations for LoRA Parameters:
Let the incoming gradient from the downstream loss be $G = \frac{\partial \mathcal{L}}{\partial h} \in \mathbb{R}^{B \times d_{\text{out}}}$, and input activation be $X \in \mathbb{R}^{B \times d_{\text{in}}}$.  
The forward pass is $h = X W_0^T + \frac{\alpha}{r} X A^T B^T$.

$$ \frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} G^T (X A^T) \in \mathbb{R}^{d_{\text{out}} \times r} $$
$$ \frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} (G B)^T X \in \mathbb{R}^{r \times d_{\text{in}}} $$
$$ \frac{\partial \mathcal{L}}{\partial X} = G W_0 + \frac{\alpha}{r} G B A \in \mathbb{R}^{B \times d_{\text{in}}} $$

*Zero Inference Overhead*: Before deployment, compute $W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$. Inference latency is identical to the base model.

---

### 2.2 QLoRA: NormalFloat4 (NF4) & Double Quantization (Dettmers et al., 2023)

QLoRA enables fine-tuning a 70B parameter model on a single 80GB GPU by quantizing base weights $W_0$ to 4-bit NormalFloat while training 16-bit LoRA adapters.

1. **NormalFloat4 (NF4) Quantile Construction**:
   Weights of pre-trained models follow a normal distribution $W_0 \sim \mathcal{N}(0, \sigma^2)$.  
   NF4 constructs 16 discrete quantization bins $q_i \in [-1, 1]$ ($i \in \{0, \dots, 15\}$) such that each bin contains an **equal probability mass**:
   $$ q_i = \frac{1}{2} \left( Q_X\left(\frac{i}{2^k}\right) + Q_X\left(\frac{i+1}{2^k}\right) \right) $$
   where $Q_X(\cdot)$ is the quantile function (inverse CDF) of the standard normal distribution $\mathcal{N}(0, 1)$.

2. **Double Quantization (DQ)**:
   Quantization divides weights into blocks of size 64 with a 32-bit FP32 quantization scale $c_1$.  
   $c_1$ adds $\frac{32}{64} = 0.5 \text{ bits/param}$ of overhead.  
   Double Quantization quantizes the scales $c_1$ themselves into 8-bit integers with a second block size of 256 ($c_2$ in FP32):
   $$ \text{Scale Memory Overhead} = \frac{8}{64} + \frac{32}{64 \times 256} = 0.125 + 0.00195 = \mathbf{0.127 \text{ bits/parameter}} $$
   Saves $\sim 0.373$ bits per parameter ($\sim 3.2\text{ GB}$ on a 70B model).

---

## 3. Preference Alignment: Direct Preference Optimization (DPO)

### 3.1 The Bradley-Terry Preference Model

Given prompt $x$ and two candidate responses: winning response $y_w$ and losing response $y_l$ ($y_w \succ y_l$).  
The Bradley-Terry (BT) model posits that human preference is governed by an underlying latent reward function $r^*(x, y)$:

$$ P(y_w \succ y_l \mid x) = \sigma\left(r^*(x, y_w) - r^*(x, y_l)\right) = \frac{1}{1 + e^{-(r^*(x, y_w) - r^*(x, y_l))}} $$

---

### 3.2 Classical RLHF Objective Formulation

In standard PPO-based RLHF, we optimize the policy $\pi_\theta$ to maximize expected reward while penalizing divergence from the reference policy $\pi_{\text{ref}}$ (preventing reward hacking):

$$ \max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}\left[ r(x, y) \right] - \beta \mathbb{D}_{\text{KL}}\left( \pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x) \right) $$

Expanding the KL divergence:
$$ \max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_y \pi_\theta(y \mid x) r(x, y) - \beta \sum_y \pi_\theta(y \mid x) \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right] $$
$$ = \max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}} \left[ \beta \sum_y \pi_\theta(y \mid x) \left( \frac{1}{\beta} r(x, y) - \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) \right] $$

---

### 3.3 The Step-by-Step DPO Derivation (Rafailov et al., 2023)

#### Step 1: Solving the Optimal Policy in Closed Form
Let $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ be the partition function.  
We can rewrite the objective inside the expectation as:
$$ -\beta \sum_y \pi_\theta(y \mid x) \log \frac{\pi_\theta(y \mid x)}{\frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)} + \beta \log Z(x) $$
$$ = -\beta \mathbb{D}_{\text{KL}}\left( \pi_\theta(y \mid x) \;\parallel\; \pi^*(y \mid x) \right) + \beta \log Z(x) $$

Where the optimal policy distribution $\pi^*$ is:
$$ \pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right) $$

Because KL divergence is non-negative ($\mathbb{D}_{\text{KL}} \geq 0$) and minimized to $0$ if and only if $\pi_\theta = \pi^*$, the theoretical optimal policy is:
$$ \mathbf{\pi_\theta^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)} $$

#### Step 2: Re-parameterizing the Reward Function
Taking the natural logarithm on both sides:
$$ \log \pi_\theta^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x) $$
Rearranging to isolate the exact reward function $r(x, y)$:
$$ \mathbf{r(x, y) = \beta \log \frac{\pi_\theta^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)} $$

#### Step 3: Substituting into the Bradley-Terry Preference Likelihood
Substitute the re-parameterized reward $r(x, y)$ into the Bradley-Terry formula $P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$:
$$ r(x, y_w) - r(x, y_l) = \left[ \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right] - \left[ \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right] $$

$$\mathbf{r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}}$$

**Notice that the partition function $Z(x)$ completely cancels out!**

#### Step 4: The Closed-Form DPO Loss Function
Minimizing the Negative Log-Likelihood of preferences yields the DPO loss:

$$ \mathbf{\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]} $$

*Why DPO Revolutionized Alignment*: Completely eliminates the need to train a separate Reward Model, eliminates Actor-Critic networks, eliminates PPO hyperparameter instability, and requires $2\times$ less GPU VRAM.

---

### 3.4 Analytical Gradient of the DPO Loss

Let $u = \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$.  
Since $\frac{d}{du} [-\log \sigma(u)] = -(1 - \sigma(u)) = -\sigma(-u)$:

$$ \nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \cdot \underbrace{\sigma\left( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \right)}_{\text{Implicit Error Weight } w(x, y_w, y_l)} \left[ \underbrace{\nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{Increase likelihood of } y_w} - \underbrace{\nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{Decrease likelihood of } y_l} \right] $$

#### Gradient Mechanics:
- When the model currently prefers the wrong answer ($\hat{r}_\theta(y_l) > \hat{r}_\theta(y_w)$), the error weight $\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w)) \to 1$, applying the maximum gradient update.
- When the model already strongly prefers the winning response ($\hat{r}_\theta(y_w) \gg \hat{r}_\theta(y_l)$), the gradient scales to $0$, avoiding unnecessary updates.

---

## 4. Reasoning Models & Reinforcement Learning: GRPO (DeepSeek-R1)

### 4.1 Group Relative Policy Optimization (GRPO)

Traditional PPO trains an **Actor Model** $\pi_\theta$ and a **Critic Model** $V_\phi$. For a 70B model, the Critic model requires an additional 70B parameters in GPU VRAM and complex Value function fitting.  
DeepSeek introduced **GRPO (Group Relative Policy Optimization)**, which eliminates the Critic network entirely.

```
Prompt x ──────► Generate Group of G Outputs: {y_1, y_2, ..., y_G}
                     │
                     ▼
         Evaluate Rule-Based Rewards: {r_1, r_2, ..., r_G}
         (e.g., Math Correctness = 1.0, Formatting = 0.5)
                     │
                     ▼
         Compute Group-Normalized Advantages:
         A_i = (r_i - Mean({r})) / Std({r})
                     │
                     ▼
         Update Policy π_θ with Clipped Surrogate Loss + KL Regularization
```

### 4.2 Mathematical Formulation of GRPO

For each prompt $x$, the model samples a group of $G$ candidate outputs $\{y_1, y_2, \dots, y_G\} \sim \pi_{\theta_{\text{old}}}(y \mid x)$.

#### 1. Group Advantage Normalization:
Instead of using a learned baseline $V(s)$, GRPO computes advantages relative to the group:
$$ A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\}) + \epsilon} $$

#### 2. The GRPO Objective Function:
$$ \mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min\left( \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)} A_i, \; \text{clip}\left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \mathbb{D}_{\text{KL}}\left(\pi_\theta \parallel \pi_{\text{ref}}\right) \right) \right] $$

Where the unbiased KL divergence estimator is:
$$ \mathbb{D}_{\text{KL}}\left(\pi_\theta \parallel \pi_{\text{ref}}\right) = \frac{\pi_{\text{ref}}(y_i \mid x)}{\pi_\theta(y_i \mid x)} - \log \frac{\pi_{\text{ref}}(y_i \mid x)}{\pi_\theta(y_i \mid x)} - 1 $$

### 4.3 Rule-Based Verifiable Rewards vs. Learned Reward Models

In reasoning tasks (Math, Code, Tool Use), using neural Reward Models causes **Reward Hacking** (the model learns to game the reward model's semantic vulnerabilities).  
GRPO uses deterministic **Rule-Based Verifiers**:
1. **Accuracy Reward**: Executes the generated Python code in a sandbox or parses final answer tags `<answer>...</answer>` against ground-truth LaTeX ($r_{\text{acc}} \in \{0, 1\}$).
2. **Format Reward**: Enforces that reasoning occurs strictly inside `<think>...</think>` tags ($r_{\text{format}} \in \{0, 1\}$).

---

## 5. PyTorch Implementation: DPO and GRPO Loss Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss Module.
    """
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,   # [Batch]
        policy_rejected_logps: torch.Tensor, # [Batch]
        reference_chosen_logps: torch.Tensor,# [Batch]
        reference_rejected_logps: torch.Tensor# [Batch]
    ) -> torch.Tensor:
        # Compute log-ratio differences: log(π_θ(y|x) / π_ref(y|x))
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits)
        return loss.mean()

class GRPOLoss(nn.Module):
    """
    Group Relative Policy Optimization (GRPO) Loss Module.
    """
    def __init__(self, clip_eps: float = 0.2, beta: float = 0.04):
        super().__init__()
        self.clip_eps = clip_eps
        self.beta = beta

    def forward(
        self,
        logp_theta: torch.Tensor,      # [Group_Size, Seq_Len]
        logp_old: torch.Tensor,        # [Group_Size, Seq_Len]
        logp_ref: torch.Tensor,        # [Group_Size, Seq_Len]
        rewards: torch.Tensor          # [Group_Size]
    ) -> torch.Tensor:
        # 1. Compute Group Normalized Advantage: (r - mean) / std
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = ((rewards - mean_r) / std_r).unsqueeze(-1) # [Group_Size, 1]

        # 2. Probability Ratio: π_θ / π_old
        ratios = torch.exp(logp_theta - logp_old)

        # 3. Clipped Surrogate Objective
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 4. KL Divergence Penalty: D_KL(π_θ || π_ref)
        kl = torch.exp(logp_ref - logp_theta) - (logp_ref - logp_theta) - 1.0
        kl_loss = self.beta * kl.mean()

        return policy_loss + kl_loss

if __name__ == "__main__":
    B = 4
    dpo = DPOLoss(beta=0.1)
    p_w, p_l = torch.tensor([-2.1, -1.5, -3.0, -0.8]), torch.tensor([-4.5, -3.2, -4.1, -2.9])
    r_w, r_l = torch.tensor([-2.0, -1.6, -3.1, -0.9]), torch.tensor([-3.8, -3.0, -3.5, -2.5])
    loss = dpo(p_w, p_l, r_w, r_l)
    print(f"DPO Loss computation verified: {loss.item():.4f}")
```

---

## 6. Deep Interview Interrogation Ladder

- **Level 1 (Concept)**: What is the primary difference between SFT and Preference Optimization?
- **Level 3 (Derivation)**: Write out the Bradley-Terry preference model and explain why the partition function $Z(x)$ cancels out in DPO.
- **Level 5 (Mechanics)**: Why is matrix $B$ initialized to zero in LoRA, and what happens mathematically if both $A$ and $B$ are initialized with Gaussian noise?
- **Level 7 (RL for Reasoning)**: Explain how GRPO estimates advantage without a Critic network, and why rule-based verifiers prevent reward hacking in math/code reasoning.
- **Level 9 (Deep Systems)**: In QLoRA, why does NormalFloat4 (NF4) achieve better quantization fidelity than standard uniform 4-bit integer quantization (INT4)?
- **Level 10 (Principal Engineering)**: You need to align a 70B agent for tool-use and long-running workflows. When would you choose DPO vs. GRPO vs. Rejection-Sampling SFT, and how do you prevent mode collapse during iterative post-training?

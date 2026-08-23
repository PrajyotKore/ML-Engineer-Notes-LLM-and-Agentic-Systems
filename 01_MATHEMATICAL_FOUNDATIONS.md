# 01_MATHEMATICAL_FOUNDATIONS — Rigorous Mathematical Reference for ML & LLM Engineers

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide an exhaustive, step-by-step mathematical bridge from fundamental Linear Algebra, Multivariate Calculus, Probability, and Optimization up to the mathematical machinery driving modern LLMs, Transformers, and Agentic Systems.

---

## 1. Linear Algebra & Matrix Decompositions

### 1.1 Vector Spaces, Projections, and Orthogonality

Let $\mathcal{V} = \mathbb{R}^d$ be an inner product space equipped with the standard Euclidean inner product $\langle u, v \rangle = u^T v = \sum_{i=1}^d u_i v_i$.

- **Vector Norms**:
  - $L_p$ Norm: $\|x\|_p = \left( \sum_{i=1}^d |x_i|^p \right)^{1/p}$
  - $L_2$ Euclidean Norm: $\|x\|_2 = \sqrt{x^T x}$
  - $L_1$ Manhattan Norm: $\|x\|_1 = \sum_{i=1}^d |x_i|$ (promotes sparsity in regularized regression)
  - $L_\infty$ Maximum Norm: $\|x\|_\infty = \max_i |x_i|$
  - Dual Norm: $\|u\|_* = \sup \{ u^T v : \|v\| \leq 1 \}$. The dual of $L_p$ is $L_q$ where $\frac{1}{p} + \frac{1}{q} = 1$.

- **Orthogonal Projection**:
  The orthogonal projection of a vector $y \in \mathbb{R}^d$ onto the subspace spanned by the columns of matrix $A \in \mathbb{R}^{d \times k}$ (where $A$ has full column rank $k$) is given by:
  $$ \text{proj}_{\text{col}(A)}(y) = A (A^T A)^{-1} A^T y $$
  The projection matrix $P = A (A^T A)^{-1} A^T$ satisfies:
  1. **Idempotence**: $P^2 = P$
  2. **Symmetry**: $P^T = P$

---

### 1.2 Spectral Theorem and Eigendecomposition

For any real symmetric matrix $A \in \mathbb{R}^{d \times d}$ ($A = A^T$):
1. All eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_d$ are real: $\lambda_i \in \mathbb{R}$.
2. Eigenvectors corresponding to distinct eigenvalues are mutually orthogonal.
3. $A$ can be orthogonally diagonalized:
   $$ A = Q \Lambda Q^T = \sum_{i=1}^d \lambda_i q_i q_i^T $$
   where $Q = [q_1, q_2, \dots, q_d]$ is an orthogonal matrix ($Q^T Q = Q Q^T = I$) containing orthonormal eigenvectors, and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_d)$.

- **Positive Semi-Definite (PSD) Matrices**:
  A symmetric matrix $A \in \mathbb{R}^{d \times d}$ is PSD ($A \succeq 0$) if and only if:
  $$ x^T A x \geq 0 \quad \forall x \in \mathbb{R}^d \iff \lambda_i(A) \geq 0 \quad \forall i \in \{1, \dots, d\} $$
  *ML Application*: Covariance matrices $\Sigma = \frac{1}{N} X^T X$, Hessian matrices at local minima $\nabla^2 \mathcal{L}(\theta) \succeq 0$, and kernel Gram matrices $K_{ij} = k(x_i, x_j)$ are all PSD.

---

### 1.3 Singular Value Decomposition (SVD)

For any arbitrary rectangular matrix $A \in \mathbb{R}^{m \times n}$ with rank $r \leq \min(m, n)$:
$$ A = U \Sigma V^T = \sum_{i=1}^r \sigma_i u_i v_i^T $$

Where:
- $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix of left singular vectors (eigenvectors of $A A^T$).
- $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix of right singular vectors (eigenvectors of $A^T A$).
- $\Sigma \in \mathbb{R}^{m \times n}$ is a rectangular diagonal matrix with non-negative singular values ordered monotonically:
  $$ \sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r > \sigma_{r+1} = \dots = 0 $$
- The relationship with eigenvalues: $\sigma_i(A) = \sqrt{\lambda_i(A^T A)} = \sqrt{\lambda_i(A A^T)}$.

#### Matrix Norms via SVD:
- **Spectral Norm ($L_2$ Operator Norm)**:
  $$ \|A\|_2 = \sup_{x \neq 0} \frac{\|Ax\|_2}{\|x\|_2} = \sigma_{\max}(A) = \sigma_1 $$
- **Frobenius Norm**:
  $$ \|A\|_F = \sqrt{\text{Tr}(A^T A)} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{ij}^2} = \sqrt{\sum_{i=1}^r \sigma_i^2} $$
- **Nuclear Norm (Trace Norm)**:
  $$ \|A\|_* = \text{Tr}\left(\sqrt{A^T A}\right) = \sum_{i=1}^r \sigma_i $$

---

### 1.4 Low-Rank Approximation & The Eckart-Young-Mirsky Theorem

The **Eckart-Young-Mirsky Theorem** provides the theoretical justification for low-rank parameter-efficient fine-tuning (LoRA), KV cache compression, and Multi-Head Latent Attention (MLA).

#### Theorem Statement:
Let $A \in \mathbb{R}^{m \times n}$ have SVD $A = \sum_{i=1}^r \sigma_i u_i v_i^T$. For any $k < r$, the optimal rank-$k$ approximation matrix $A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$ solves the optimization problem:
$$ A_k = \arg\min_{B \in \mathbb{R}^{m \times n}, \text{rank}(B) \leq k} \|A - B\| $$

The approximation errors are given analytically by:
- Under the **Spectral Norm**:
  $$ \min_{\text{rank}(B) \leq k} \|A - B\|_2 = \|A - A_k\|_2 = \sigma_{k+1} $$
- Under the **Frobenius Norm**:
  $$ \min_{\text{rank}(B) \leq k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2} $$

#### Concrete Mathematical Implication for LoRA & MLA:
When fine-tuning a weight matrix $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$, the weight update $\Delta W$ has an intrinsic rank $r \ll \min(d_{out}, d_{in})$. Parameterizing $\Delta W = \frac{\alpha}{r} B A$ (where $B \in \mathbb{R}^{d_{out} \times r}, A \in \mathbb{R}^{r \times d_{in}}$) captures the principal singular directions of the gradient trajectory while reducing parameter complexity from $d_{out} \cdot d_{in}$ to $r \cdot (d_{out} + d_{in})$.

---

## 2. Multivariate Calculus & Computational Graphs

### 2.1 Jacobians, Hessians, and Matrix Derivatives

Let $f: \mathbb{R}^n \to \mathbb{R}^m$ be a differentiable vector-valued function $y = f(x)$.

- **The Jacobian Matrix** $J \in \mathbb{R}^{m \times n}$:
  $$ J_{ij} = \frac{\partial y_i}{\partial x_j} \implies J = \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n} \end{bmatrix} $$

- **The Hessian Matrix** $H \in \mathbb{R}^{n \times n}$ for scalar loss $\mathcal{L}: \mathbb{R}^n \to \mathbb{R}$:
  $$ H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial x_i \partial x_j} \implies \nabla^2 \mathcal{L}(x) $$

- **Essential Matrix Derivative Identities**:
  Let $X \in \mathbb{R}^{m \times n}$, $a \in \mathbb{R}^m$, $b \in \mathbb{R}^n$, $A \in \mathbb{R}^{n \times m}$, $W \in \mathbb{R}^{n \times n}$:
  1. $\frac{\partial (a^T X b)}{\partial X} = a b^T \in \mathbb{R}^{m \times n}$
  2. $\frac{\partial \text{Tr}(A X)}{\partial X} = A^T$
  3. $\frac{\partial \text{Tr}(X^T A X)}{\partial X} = (A + A^T) X$ (for symmetric $A$: $2AX$)
  4. $\frac{\partial \log \det(W)}{\partial W} = W^{-T} = (W^{-1})^T$ (for invertible $W$)
  5. $\frac{\partial \|X\|_F^2}{\partial X} = 2X$
  6. $\frac{\partial (u^T W v)}{\partial W} = u v^T$

---

### 2.2 Vector-Jacobian Products (VJPs) and Automatic Differentiation

In Deep Learning backpropagation (Reverse-Mode Automatic Differentiation), we never materialize the full Jacobian $J \in \mathbb{R}^{m \times n}$ because $m, n \approx 10^7 - 10^{11}$ (e.g. 70B parameter models).

Instead, reverse-mode autodiff evaluates **Vector-Jacobian Products (VJPs)**:
Given an incoming adjoint gradient from the downstream loss $v = \nabla_y \mathcal{L} \in \mathbb{R}^{1 \times m}$:
$$ \nabla_x \mathcal{L} = v \cdot J = v \cdot \frac{\partial y}{\partial x} \in \mathbb{R}^{1 \times n} $$

```
[Forward Pass]:   x (dim n) ----[ f(x) ]----> y (dim m) ----[ Loss L ]----> Scalar Loss
[Backward Pass]:  v·J (dim n) <---[ VJP ]--- v = ∂L/∂y <---[ Adjoint ]---
```

- **Computational Complexity**:
  - Forward Evaluation: $O(T)$ FLOPs where $T$ is the number of operations in the computational graph.
  - VJP Evaluation: Guaranteed by Baur-Strassen Theorem to take at most $c \cdot O(T)$ FLOPs, where constant $c \leq 5$.

---

## 3. Probability & Information Theory

### 3.1 Random Variables, Expectations, and Covariance

Let $X \in \mathbb{R}^d$ be a continuous random vector with probability density function $p(x)$.
- **Expectation**: $\mathbb{E}[X] = \int x p(x) dx \in \mathbb{R}^d$
- **Covariance Matrix**:
  $$ \text{Cov}(X) = \Sigma = \mathbb{E}[(X - \mathbb{E}[X])(X - \mathbb{E}[X])^T] = \mathbb{E}[X X^T] - \mathbb{E}[X]\mathbb{E}[X]^T \in \mathbb{R}^{d \times d} $$
- **Properties of Covariance**:
  1. $\Sigma$ is symmetric: $\Sigma = \Sigma^T$.
  2. $\Sigma$ is positive semi-definite: $v^T \Sigma v = \text{Var}(v^T X) \geq 0$ for all $v \in \mathbb{R}^d$.
  3. Linear transformation: If $Y = A X + b$, then $\text{Cov}(Y) = A \text{Cov}(X) A^T$.

---

### 3.2 Maximum Likelihood Estimation (MLE) vs. Maximum A Posteriori (MAP)

Given an observed dataset $\mathcal{D} = \{x_1, x_2, \dots, x_N\}$ drawn i.i.d. from a parameterized distribution $p(x | \theta)$:

- **Maximum Likelihood Estimation (MLE)**:
  $$ \hat{\theta}_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^N p(x_i | \theta) = \arg\max_\theta \sum_{i=1}^N \log p(x_i | \theta) = \arg\min_\theta -\sum_{i=1}^N \log p(x_i | \theta) $$
  *Equivalence*: Minimizing Negative Log-Likelihood (NLL) is mathematically identical to minimizing the empirical Cross-Entropy between the true empirical data distribution $\hat{p}_{\text{data}}$ and the model distribution $p_\theta$.

- **Maximum A Posteriori (MAP)**:
  Incorporating a prior distribution over parameters $p(\theta)$ via Bayes' Theorem:
  $$ p(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})} \propto p(\mathcal{D} | \theta) p(\theta) $$
  $$ \hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[ \sum_{i=1}^N \log p(x_i | \theta) + \log p(\theta) \right] = \arg\min_\theta \left[ -\sum_{i=1}^N \log p(x_i | \theta) - \log p(\theta) \right] $$

- **Exact Equivalence with Regularization**:
  1. **Gaussian Prior**: If $\theta \sim \mathcal{N}(0, \sigma_0^2 I)$, then $-\log p(\theta) = \frac{1}{2\sigma_0^2} \|\theta\|_2^2 + \text{const} \implies \mathbf{L_2 \text{ Regularization (Weight Decay)}}$.
  2. **Laplace Prior**: If $\theta_j \sim \text{Laplace}(0, b)$, then $-\log p(\theta) = \frac{1}{b} \|\theta\|_1 + \text{const} \implies \mathbf{L_1 \text{ Regularization (Lasso / Sparsity)}}$.

---

### 3.3 Information Measures: Entropy, Cross-Entropy, and KL Divergence

Let $P$ and $Q$ be discrete probability distributions over a discrete support $\mathcal{X}$.

```
               ┌───────────────────────────────────────────────┐
               │         Cross-Entropy H(P, Q)                 │
               │  = - \sum P(x) \log Q(x)                      │
               └───────────────────────┬───────────────────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
    ┌─────────────────────────────┐         ┌─────────────────────────────┐
    │     Shannon Entropy H(P)    │    +    │    KL Divergence D_KL(P||Q) │
    │   = - \sum P(x) \log P(x)   │         │  = \sum P(x) \log(P(x)/Q(x))│
    └─────────────────────────────┘         └─────────────────────────────┘
```

#### Mathematical Definitions:
1. **Shannon Entropy** (Information content / uncertainty of $P$):
   $$ H(P) = -\sum_{x \in \mathcal{X}} P(x) \log_2 P(x) = \mathbb{E}_{x \sim P}\left[\log_2 \frac{1}{P(x)}\right] $$
2. **Kullback-Leibler (KL) Divergence** (Relative entropy / information lost when $Q$ approximates $P$):
   $$ D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \left( \frac{P(x)}{Q(x)} \right) = \mathbb{E}_{x \sim P}\left[ \log \frac{P(x)}{Q(x)} \right] $$
3. **Cross-Entropy**:
   $$ H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \log Q(x) = H(P) + D_{KL}(P \parallel Q) $$

#### Fundamental Properties of KL Divergence:
- **Non-negativity (Gibbs' Inequality)**:
  $$ D_{KL}(P \parallel Q) \geq 0 \quad \text{with equality } \iff P = Q \text{ almost everywhere.} $$
  *Proof via Jensen's Inequality*: Since $-\log(t)$ is strictly convex:
  $$ D_{KL}(P \parallel Q) = \mathbb{E}_{x \sim P}\left[ -\log \frac{Q(x)}{P(x)} \right] \geq -\log \left( \mathbb{E}_{x \sim P}\left[ \frac{Q(x)}{P(x)} \right] \right) = -\log \left( \sum_{x} P(x) \frac{Q(x)}{P(x)} \right) = -\log(1) = 0 $$

- **Asymmetry**: $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$
  - **Forward KL ($D_{KL}(P_{\text{true}} \parallel Q_\theta)$) — Mode Covering / Zero-Avoiding**:
    If $P(x) > 0$, $Q(x)$ must be $> 0$ to avoid infinite penalty. $Q_\theta$ averages over all modes of $P$. (Used in Maximum Likelihood & Standard SFT).
  - **Reverse KL ($D_{KL}(Q_\theta \parallel P_{\text{true}})$) — Mode Seeking / Zero-Forcing**:
    If $P(x) = 0$, $Q_\theta(x)$ is forced to $0$. $Q_\theta$ locks onto a single high-probability mode of $P$ and drops the rest. (Used in RLHF, Policy Gradients, and Distillation).

```
Forward KL:  P(x) [Multi-modal]   --> Q(x) stretches across ALL modes (blurry/safe)
Reverse KL:  P(x) [Multi-modal]   --> Q(x) concentrates on ONE sharp mode
```

---

### 3.4 Perplexity and Sequence Likelihood

For an autoregressive language model generating a token sequence $X = (x_1, x_2, \dots, x_T)$:
The sequence probability factorizes by the probability chain rule:
$$ P_\theta(X) = \prod_{t=1}^T P_\theta(x_t \mid x_{<t}) $$

The **Perplexity (PPL)** of the sequence is the exponentiated average negative log-likelihood per token:
$$ \text{PPL}(X) = \exp \left( -\frac{1}{T} \sum_{t=1}^T \log P_\theta(x_t \mid x_{<t}) \right) = \left( \prod_{t=1}^T \frac{1}{P_\theta(x_t \mid x_{<t})} \right)^{1/T} = 2^{H(\hat{P}_{\text{data}}, P_\theta)} $$

*Interpretation*: A perplexity of $K$ means the model is as uncertain at each step as choosing uniformly among $K$ discrete tokens.

---

## 4. Optimization Theory & Loss Landscapes

### 4.1 Gradient Descent Dynamics & The Conditioning Number

Let $f: \mathbb{R}^d \to \mathbb{R}$ be twice continuously differentiable. The second-order Taylor expansion around $\theta_t$:
$$ f(\theta) \approx f(\theta_t) + \nabla f(\theta_t)^T (\theta - \theta_t) + \frac{1}{2} (\theta - \theta_t)^T H (\theta - \theta_t) $$
where $H = \nabla^2 f(\theta_t)$ is the Hessian matrix.

- **Lipschitz Smoothness**:
  A function $f$ is $L$-smooth if $\|\nabla f(x) - \nabla f(y)\|_2 \leq L \|x - y\|_2$, which implies:
  $$ \lambda_{\max}(\nabla^2 f(x)) \leq L \quad \forall x $$
  Under $L$-smoothness, standard gradient descent $\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$ converges if learning rate $\eta < \frac{2}{L}$.

- **Condition Number of the Hessian**:
  $$ \kappa = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)} $$
  - **Well-conditioned ($\kappa \approx 1$)**: Loss contours are spherical. Gradient points directly toward the global minimum.
  - **Ill-conditioned ($\kappa \gg 1$)**: Loss contours are elongated ravines. Standard gradient descent oscillates violently across the steep direction while making negligible progress along the flat direction.

```
       Ill-Conditioned Landscape (κ >> 1):
       ▲
       │      \      /           Oscillations across steep direction (λ_max)
       │       \    /   ▲  ▲
       │        \  /    │  │     Very slow progress along flat canyon (λ_min)
       │         \/     ▼  ▼
       └────────────────────────►
```

---

### 4.2 First-Order Optimizers: Momentum, Adam, and AdamW

#### 1. Stochastic Gradient Descent with Polyak Momentum:
Accumulates an exponentially decaying moving average of past gradients to cancel orthogonal oscillations:
$$ v_{t+1} = \beta v_t + g_t \quad (\beta \in [0.9, 0.99]) $$
$$ \theta_{t+1} = \theta_t - \eta v_{t+1} $$

#### 2. Adaptive Moment Estimation (Adam):
Maintains running estimates of both the uncentered mean ($m_t$) and uncentered variance ($v_t$) of the gradients:

$$ g_t = \nabla_\theta \mathcal{L}(\theta_t) $$
$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(First Moment: Mean)} $$
$$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(Second Moment: Uncentered Variance)} $$

- **Bias Correction Derivation**:
  Expanding $m_t$ assuming $g_i$ come from a stationary distribution with mean $\mathbb{E}[g]$:
  $$ m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \implies \mathbb{E}[m_t] = \mathbb{E}[g] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} = \mathbb{E}[g] (1 - \beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[g] (1 - \beta_1^t) $$
  Therefore, the unbiased estimators are:
  $$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$

- **Adam Update Rule**:
  $$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t $$

---

### 4.3 The Critical Mathematical Difference: $L_2$ Regularization vs. AdamW (Decoupled Weight Decay)

In standard SGD, $L_2$ regularization $\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|_2^2$ is mathematically equivalent to weight decay:
$$ \nabla_\theta \mathcal{L}_{\text{reg}}(\theta) = \nabla \mathcal{L}(\theta) + \lambda \theta \implies \theta_{t+1} = \theta_t - \eta (\nabla \mathcal{L}(\theta_t) + \lambda \theta_t) = (1 - \eta \lambda)\theta_t - \eta \nabla \mathcal{L}(\theta_t) $$

#### The Failure of $L_2$ Regularization in Adaptive Optimizers (Adam):
When adding $L_2$ regularization to Adam:
$$ g_t^{\text{reg}} = g_t + \lambda \theta_t $$
The gradient $g_t^{\text{reg}}$ enters the second-moment estimator $v_t$:
$$ v_t = \beta_2 v_{t-1} + (1 - \beta_2)(g_t + \lambda \theta_t)^2 $$
The parameter update becomes:
$$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t^{\text{reg}}} + \epsilon} \hat{m}_t^{\text{reg}} \approx \theta_t - \frac{\eta \lambda}{\sqrt{\hat{v}_t^{\text{reg}}}} \theta_t - \frac{\eta}{\sqrt{\hat{v}_t^{\text{reg}}}} \hat{m}_t $$

**The Flaw**: Parameters with massive historical gradients have large $\hat{v}_t$, meaning their effective weight decay rate $\frac{\eta \lambda}{\sqrt{\hat{v}_t}}$ is drastically suppressed. Conversely, parameters with tiny or infrequent gradients receive disproportionately large weight decay.

#### The AdamW Solution (Loshchilov & Hutter):
Decouple weight decay completely from the gradient moment updates:
$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t $$
$$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $$
$$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$
$$ \theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t $$

*Production Implication*: Every modern LLM (LLaMA-3, Mistral, DeepSeek, GPT-4) uses AdamW exclusively.

---

### 4.4 Learning Rate Schedules: Warmup and Cosine Decay

#### Mathematical Formulation:
Given initial learning rate $\eta_{\max}$, minimum learning rate $\eta_{\min}$, warmup steps $T_{\text{warmup}}$, and total training steps $T_{\text{total}}$:

$$ \eta(t) = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t \leq T_{\text{warmup}} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left( \pi \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \right) \right) & \text{if } T_{\text{warmup}} < t \leq T_{\text{total}} \end{cases} $$

#### Why Warmup is Mathematically Mandatory:
At step $t=1$, the second moment vector $v_0 = 0$. In early iterations ($t < 100$), the variance estimate $\hat{v}_t$ is uncalibrated and noisy. Dividing by $\sqrt{\hat{v}_t}$ results in huge, erratic updates in arbitrary directions that destroy pre-trained features or cause loss explosion. Warmup constrains step sizes until the second-moment moving average stabilizes.

---

## 5. Mathematical Summary Cheat Sheet for Interviews

| Mathematical Concept | Core Equation / Formulation | Primary ML Application |
| :--- | :--- | :--- |
| **Eckart-Young Theorem** | $\min_{\text{rank}(B)\leq k} \|A - B\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$ | LoRA, MLA KV-Compression, SVD Pruning |
| **Negative Log-Likelihood** | $\mathcal{L}_{\text{NLL}}(\theta) = -\sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})$ | Autoregressive Next-Token Prediction |
| **KL Divergence** | $D_{KL}(P \parallel Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$ | Distribution Drift, RLHF KL Penalties, DPO |
| **Reverse KL** | $D_{KL}(Q_\theta \parallel P_{\text{true}})$ | Policy Gradient RL, Distillation (Mode Seeking) |
| **Perplexity** | $\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})\right)$ | Model Quality & Validation Metric |
| **AdamW Update** | $\theta_{t+1} = (1 - \eta \lambda)\theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t$ | Standard LLM Pre-training & Fine-tuning Optimizer |
| **Condition Number** | $\kappa = \lambda_{\max}(H) / \lambda_{\min}(H)$ | Loss Landscape Curvature & Convergence Speed |
| **Cosine LR Decay** | $\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})(1 + \cos(\pi \frac{t}{T}))$ | LLM Learning Rate Annealing Schedule |

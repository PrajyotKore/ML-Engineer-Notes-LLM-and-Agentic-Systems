# 02_MACHINE_LEARNING_FOUNDATIONS & 03_DEEP_LEARNING — Technical Reference

## 1. Role Relevance
For an ML Engineer (LLM & Agentic Systems), deep learning foundations are the prerequisite to understanding LLMs. You cannot debug a stalled loss curve, NaNs in training, or catastrophic forgetting in SFT without a mechanistic and mathematical understanding of optimization, loss landscapes, and backpropagation.

## 2. Prerequisites
- Linear Algebra (Vector spaces, Eigenvalues)
- Multivariate Calculus (Gradients, Jacobians, Hessians)
- Basic Probability (Expectation, Variance, Distributions)

## 3. First Principles
Deep learning models are highly parameterized, non-linear function approximators trained via first-order optimization. We define an objective (loss), compute its gradient with respect to parameters (backpropagation), and update parameters to minimize the objective (gradient descent).

## 4. Mechanistic Breakdown
### Forward Pass
Input data flows through a series of linear transformations (matrix multiplications) and non-linear activations (e.g., ReLU, SiLU).
### Backward Pass
The error at the output is propagated backward using the chain rule, calculating the local gradient of each operation and accumulating it.
### Optimizer Update
The optimizer (e.g., AdamW) takes the accumulated gradients, applies momentum and variance estimates, applies weight decay, and updates the weights.

## 5. Mathematical Foundations

### Cross-Entropy Loss
In language modeling (classification over vocabulary), we minimize the negative log-likelihood of the true token.

$$ \mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c}) $$

Where:
- $N$ is the batch size (or sequence length $\times$ batch size).
- $C$ is the vocabulary size.
- $y_{i,c}$ is the indicator variable (1 if class $c$ is the true token for instance $i$, 0 otherwise).
- $\hat{y}_{i,c}$ is the predicted probability (softmax output) for class $c$.

*Intuition*: Cross-entropy measures the distance (KL divergence) between the true empirical distribution and the model's predicted distribution.

### AdamW Optimizer
Adam with decoupled weight decay is the standard for LLM training.

**Given:**
- Step size $\eta$
- Exponential decay rates $\beta_1, \beta_2$
- Weight decay $\lambda$
- Objective function $f(\theta)$

**Update Rule:**
1. Compute gradient: $g_t = \nabla_{\theta} f(\theta_{t-1})$
2. Update biased first moment: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
3. Update biased second raw moment: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
4. Compute bias-corrected moments: $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$
5. Update parameters with weight decay:
   $$ \theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right) $$

*Dimensions*: $m_t$ and $v_t$ have the exact same shape as $\theta$. This means Adam requires $3\times$ the memory of the parameters alone (Params, Gradients, Optimizer States [m, v]).

## 6. Implementation
**Softmax with Numerical Stability (Python/NumPy):**
```python
import numpy as np

def stable_softmax(logits):
    # Subtract max for numerical stability to prevent overflow in exp()
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
```

## 7. Computational Complexity
- **Matrix Multiplication (Linear Layer)**: For input $X \in \mathbb{R}^{B \times d_{in}}$ and weight $W \in \mathbb{R}^{d_{in} \times d_{out}}$, the forward pass requires $2 \cdot B \cdot d_{in} \cdot d_{out}$ FLOPs.
- The backward pass requires approximately $2 \times$ the FLOPs of the forward pass (computing gradient w.r.t weights, and gradient w.r.t inputs).

## 8. Hardware / GPU Behavior
- **Memory Bound vs Compute Bound**: Element-wise operations (like activations, LayerNorm, and optimizer updates) are *memory bound*. Matrix multiplications are *compute bound*.
- **Tensor Cores**: Modern GPUs use Tensor Cores to accelerate FP16/BF16 matrix multiplications. Dimensions must typically be multiples of 8 or 16 for optimal Tensor Core utilization.

## 9. Production Architecture
In production, deep learning fundamentals dictate deployment strategies. Because Adam requires massive memory, production inference drops optimizer states and gradients, relying only on FP16/INT8 quantized weights to maximize batch size.

## 10. Scalability
As models scale, numerical stability degrades. BF16 is preferred over FP16 for training because its wider exponent (8 bits, same as FP32) prevents gradient underflow/overflow, even though it sacrifices precision (7 bits of mantissa).

## 11. Bottlenecks
- **Loss Calculation in LLMs**: Computing Cross-Entropy over a 128k vocabulary for a 4k sequence is massively memory intensive. We use techniques like Flash Cross-Entropy or vocabulary partitioning in parallel to prevent OOM.

## 12. Failure Modes
- **Exploding Gradients**: Unstable loss. Mitigated by gradient clipping (e.g., `torch.nn.utils.clip_grad_norm_`).
- **Loss Spikes**: Often caused by bad data batches (e.g., highly anomalous lengths or corrupted text) or high learning rate during a sharp loss landscape transition.

## 13. Debugging
- **NaNs in Loss**: Usually caused by division by zero, `log(0)`, or exploding FP16 values. Switch to BF16, check input data for anomalies, or add $\epsilon$ to denominators.
- **Silent Degradation**: Loss goes down, but task performance drops. The loss function is a proxy; it does not perfectly map to agentic reasoning capabilities.

## 14. Trade-offs
- **Batch Size vs Learning Rate**: Increasing batch size reduces gradient variance, allowing for a larger learning rate (linear scaling rule), but increases memory footprint.
- **Adam vs SGD**: Adam converges faster and handles sparse gradients well, but requires massive memory for $m_t$ and $v_t$.

## 15. Principal-Level Reasoning
"If my SFT run is suddenly diverging, I don't just blindly lower the learning rate. I first check the gradient norms to see if a specific layer is exploding. I examine the data batch at the iteration of divergence. I ensure that my loss masking for padding tokens is correctly applied, because computing cross-entropy on padded elements will inject massive noise into the gradient."

## 16. Interview Interrogation
- *Level 1*: What is Cross-Entropy?
- *Level 3*: Derive the derivative of Softmax + Cross-Entropy.
- *Level 5*: Implement a numerically stable softmax.
- *Level 7*: Why does Adam need so much memory, and how does ZeRO optimization partition it?
- *Level 9*: Your loss spiked at step 4000. Walk me through exactly what metrics you pull to diagnose the root cause.

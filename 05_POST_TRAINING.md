# 05_POST_TRAINING — Technical Reference

## 1. Role Relevance
For an A1 Technical Lead, post-training (Alignment) is the primary lever to inject agentic capabilities, tool use, and safety into a base LLM. You must architect SFT and RLHF/DPO pipelines, choosing between LoRA for efficiency or full fine-tuning for maximum capability, while ensuring data quality.

## 2. Prerequisites
- Cross-Entropy Loss, Gradient Descent.
- Transformer weights and projections (Q/K/V/O, FFN).
- Instruction following formats (ChatML).

## 3. First Principles
Base models learn the distribution of internet text $P(text)$. Post-training aligns the model to the distribution of useful, safe, instruction-following behavior $P(response | prompt)$.

## 4. Mechanistic Breakdown
### SFT (Supervised Fine-Tuning)
We format data into prompt-response pairs. We apply standard Cross-Entropy loss, but we *mask* the loss on the prompt tokens. The model only receives gradients for the tokens in the target response.

### LoRA (Low-Rank Adaptation)
Instead of updating all weights in a dense matrix $W_0$, we freeze $W_0$ and train a low-rank approximation of the update matrix $\Delta W$.

### DPO (Direct Preference Optimization)
Instead of training a separate Reward Model and using PPO (RLHF), DPO optimizes the LLM directly on preference pairs (chosen vs. rejected) by showing that the optimal policy can be extracted directly from the reward function.

## 5. Mathematical Foundations

### LoRA (Low-Rank Adaptation)
Given a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$. We constrain the update $\Delta W$ by representing it as the product of two smaller matrices:

$$ W = W_0 + \Delta W = W_0 + BA $$

Where:
- $B \in \mathbb{R}^{d \times r}$
- $A \in \mathbb{R}^{r \times k}$
- $r \ll \min(d, k)$ is the rank.

**Initialization**: $A$ is initialized with random Gaussian noise. $B$ is initialized to zero. Therefore, at step 0, $\Delta W = 0$.
**Scaling**: The update is scaled by $\frac{\alpha}{r}$, where $\alpha$ is a constant.

### DPO (Direct Preference Optimization)
Given a prompt $x$, a chosen response $y_w$, and a rejected response $y_l$.
The DPO loss minimizes the negative log-sigmoid of the implicit reward difference.

$$ \mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right] $$

Where:
- $\pi_\theta$ is the model being trained.
- $\pi_{ref}$ is the frozen reference model (typically the SFT model).
- $\beta$ controls the strength of the KL divergence penalty from the reference model.
- $\sigma$ is the sigmoid function.

*Intuition*: Increase the likelihood of the chosen response relative to the reference, and decrease the likelihood of the rejected response, scaled by $\beta$.

## 6. Implementation
**SFT Data Packing:**
To maximize GPU throughput, short conversations are concatenated ("packed") into a single sequence of maximum length (e.g., 4096).
To prevent cross-contamination in attention, we apply **document masking** (modifying the causal mask so tokens from doc B cannot attend to doc A), OR we rely on standard EOS tokens and accept slight noise.

## 7. Computational Complexity
- **LoRA Memory**: If $r=8$, $A$ and $B$ have infinitesimally small parameters compared to $W_0$. Thus, optimizer states (Adam $m_t, v_t$) are only maintained for $A$ and $B$, drastically dropping memory requirements.
- **QLoRA**: Freezes $W_0$ in 4-bit NormalFloat (NF4). During the forward pass, it is dynamically dequantized to BF16, multiplied with inputs, and added to the BF16 LoRA output. This enables fine-tuning a 70B model on a single 80GB GPU.

## 8. Production Architecture
**Serving LoRA in Production (Multi-LoRA Serving):**
Because $BA$ is just an addition, you can serve a single massive base model $W_0$, and batch requests dynamically. For a request requiring Agent A, compute $xW_0 + xB_A A_A$. For a request requiring Agent B, compute $xW_0 + xB_B A_B$.
This allows serving hundreds of specialized agents on a single GPU cluster.

## 9. Scalability & Bottlenecks
- **Catastrophic Forgetting**: Overtraining on SFT data degrades the model's general reasoning capabilities. Mitigated by mixing in "replay" data (a small subset of pre-training data) during SFT.
- **Rank Scaling**: Increasing LoRA rank $r$ past 64 or 128 often yields diminishing returns because the intrinsic dimensionality of the task adaptation is low.

## 10. Failure Modes
- **DPO Reward Hacking**: The model learns to output longer responses because length correlates with human preference, rather than actually being better.
- **SFT Overfitting**: Training loss drops to 0.01, but the model begins repeating itself or outputting EOS immediately in production. The solution is fewer epochs, higher weight decay, or more diverse data.

## 11. Interview Interrogation
- *Level 1*: What is the difference between SFT and Pre-training?
- *Level 4*: Why is the $B$ matrix in LoRA initialized to zero?
- *Level 6*: Explain the implicit reward formulation in DPO. Why do we need the reference model?
- *Level 9*: Your DPO training is failing because the log-probs of the chosen and rejected responses are identical. What metric do you check first? (Answer: check if the dataset has identical chosen/rejected pairs, or if $\beta$ is too high).
- *Level 10*: Architect a multi-tenant platform to train and serve 1,000 personalized agent LoRAs. How do you batch inference?

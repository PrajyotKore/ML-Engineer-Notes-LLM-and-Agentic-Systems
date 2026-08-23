# 04_TRANSFORMERS_AND_LLMS — Mathematical & Architectural Reference

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide an exhaustive, mathematically rigorous masterclass on Transformer mechanics — covering Scaled Dot-Product Attention, Multi-Head Latent Attention (MLA), RoPE/YaRN mathematical proofs, SwiGLU gating, Mixture of Experts (MoE) load balancing, and long-context scaling.

---

## 1. The Core Attention Machinery

### 1.1 Scaled Dot-Product Attention: Variance Preservation Derivation

Let $Q \in \mathbb{R}^{B \times S_q \times d_k}$, $K \in \mathbb{R}^{B \times S_{kv} \times d_k}$, and $V \in \mathbb{R}^{B \times S_{kv} \times d_v}$ denote Query, Key, and Value tensors.

$$ \text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V $$

Where $M \in \mathbb{R}^{S_q \times S_{kv}}$ is the causal attention mask:
$$ M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases} $$

#### Mathematical Proof: Why the $\frac{1}{\sqrt{d_k}}$ Scaling Factor is Mandatory
Consider a single query vector $q \in \mathbb{R}^{d_k}$ and key vector $k \in \mathbb{R}^{d_k}$. Assume components $q_i, k_i$ are independent random variables with zero mean and unit variance:
$$ \mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \quad \text{Var}(q_i) = \text{Var}(k_i) = 1 $$

The dot product is $S = q^T k = \sum_{i=1}^{d_k} q_i k_i$.
1. **Expected Value**:
   $$ \mathbb{E}[S] = \mathbb{E}\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0 $$
2. **Variance**:
   $$ \text{Var}(S) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) $$
   Since $q_i, k_i$ are independent:
   $$ \text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - 0 = \text{Var}(q_i)\text{Var}(k_i) = 1 \cdot 1 = 1 $$
   Therefore:
   $$ \mathbf{\text{Var}(S) = \sum_{i=1}^{d_k} 1 = d_k \implies \text{Std}(S) = \sqrt{d_k}} $$

#### The Failure Mode without Scaling:
For modern head dimensions $d_k = 128$, $\text{Std}(S) = \sqrt{128} \approx 11.31$.  
When logits have standard deviation $\sim 11.3$, the Softmax function pushes outputs to extreme values ($p_{\max} \approx 1, p_i \approx 0$).  
Recall the Softmax Jacobian: $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$.  
When $p_i \in \{0, 1\}$, $p_i(1 - p_i) \to 0$, causing **vanishing gradients** across the entire attention layer during backpropagation.  
Scaling by $\frac{1}{\sqrt{d_k}}$ normalizes $\text{Var}\left(\frac{q^T k}{\sqrt{d_k}}\right) = 1$, keeping gradients within the optimal non-saturating zone.

---

### 1.2 Attention Taxonomy: MHA, MQA, GQA, and MLA

```
┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
│ MHA (Vaswani et al.)   │  │ MQA (Shazeer 2019)     │  │ GQA (Ainslie et al.)   │  │ MLA (DeepSeek-V2/V3)   │
│ N_q = 64, N_kv = 64    │  │ N_q = 64, N_kv = 1     │  │ N_q = 64, N_kv = 8     │  │ Low-Rank Latent Vector │
│ KV Cache: 100% (High)  │  │ KV Cache: 1.56% (Lossy)│  │ KV Cache: 12.5% (Ideal)│  │ KV Cache: ~10% (Exact) │
└────────────────────────┘  └────────────────────────┘  └────────────────────────┘  └────────────────────────┘
```

#### Mathematical Comparison Table:

| Architecture | Query Heads ($N_q$) | Key/Value Heads ($N_{kv}$) | KV Cache Memory per Token/Layer | Quality / Accuracy Retention |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Head Attention (MHA)** | $H$ | $H$ | $2 \cdot H \cdot d_k \cdot 2 \text{ bytes}$ | 100% (Baseline) |
| **Multi-Query Attention (MQA)** | $H$ | $1$ | $2 \cdot 1 \cdot d_k \cdot 2 \text{ bytes}$ | Noticeable quality degradation on complex retrieval |
| **Grouped-Query Attention (GQA)** | $H$ | $G = \frac{H}{\text{group\_size}}$ | $2 \cdot G \cdot d_k \cdot 2 \text{ bytes}$ | $\sim 99\%$ of MHA quality with $\frac{H}{G}\times$ memory savings |
| **Multi-Head Latent Attention (MLA)** | $H$ | Compressed Latent $d_c$ | $(d_c + d_R) \cdot 2 \text{ bytes}$ | Matches or exceeds MHA quality with lower footprint |

---

## 2. Multi-Head Latent Attention (MLA) — DeepSeek Architecture

### 2.1 Theoretical Motivation
In standard GQA (e.g. LLaMA-3 70B with $N_{kv}=8, d_k=128$), the KV cache at 128k context still occupies hundreds of gigabytes across the cluster, severely bounding inference throughput. DeepSeek introduced **Multi-Head Latent Attention (MLA)** to compress Key and Value representations into a low-rank latent space while preserving full expressive power.

### 2.2 Mathematical Formulation of MLA

Let $h_t \in \mathbb{R}^{d_{\text{model}}}$ be the hidden state at token $t$.

```
Hidden State h_t (dim d_model)
   │
   ├───► Down-Projection: c_t^{KV} = W^{DKV} · h_t   [Compressed KV Latent: dim d_c << H · d_h]
   │        │
   │        ├──► Up-Projection Keys:   k_{t, i}^C = W_i^{UK} · c_t^{KV}
   │        └──► Up-Projection Values: v_{t, i}^C = W_i^{UV} · c_t^{KV}
   │
   └───► Decoupled RoPE Key:           k_t^R = RoPE(W^{KR} · h_t)  [dim d_R]
```

#### Step 1: Low-Rank KV Compression (For Cache Storage)
Instead of caching full high-dimensional $K$ and $V$ tensors, MLA projects $h_t$ into a compressed latent vector $c_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c \ll N_h \cdot d_h$):
$$ c_t^{KV} = W^{DKV} h_t \quad \text{where } W^{DKV} \in \mathbb{R}^{d_c \times d_{\text{model}}} $$

#### Step 2: Decoupled Key Generation
Because Rotary Positional Embeddings (RoPE) are position-dependent and cannot be directly multiplied into static low-rank matrices, MLA decouples Keys into **content** and **rotary position** components:
$$ k_{t, i}^C = W_i^{UK} c_t^{KV} \in \mathbb{R}^{d_h} \quad \text{(Content Key for head } i \text{)} $$
$$ k_t^R = \text{RoPE}(W^{KR} h_t, t) \in \mathbb{R}^{d_R} \quad \text{(Decoupled Shared RoPE Key, } d_R \ll d_h \text{)} $$
$$ k_{t, i} = \begin{bmatrix} k_{t, i}^C \\ k_t^R \end{bmatrix} \in \mathbb{R}^{d_h + d_R} $$

#### Step 3: Low-Rank Query Projection & Decoupled RoPE Query
Queries are similarly compressed and decoupled:
$$ c_t^Q = W^{DQ} h_t \in \mathbb{R}^{d_c'} \quad (d_c' \text{ is query compression dimension}) $$
$$ q_{t, i}^C = W_i^{UQ} c_t^Q \in \mathbb{R}^{d_h}, \quad q_{t, i}^R = \text{RoPE}(W_i^{QR} c_t^Q, t) \in \mathbb{R}^{d_R} $$
$$ q_{t, i} = \begin{bmatrix} q_{t, i}^C \\ q_{t, i}^R \end{bmatrix} \in \mathbb{R}^{d_h + d_R} $$

#### Step 4: Value Generation & Attention Computation
$$ v_{t, i}^C = W_i^{UV} c_t^{KV} \in \mathbb{R}^{d_v} $$
$$ A_{ij} = \text{Softmax}\left( \frac{q_i^T k_j}{\sqrt{d_h + d_R}} \right) = \text{Softmax}\left( \frac{(q_i^C)^T k_j^C + (q_i^R)^T k_j^R}{\sqrt{d_h + d_R}} \right) $$
$$ u_{t, i} = \sum_j A_{ij} v_{j, i}^C $$
$$ o_t = W^O \begin{bmatrix} u_{t, 1} \\ \vdots \\ u_{t, N_h} \end{bmatrix} $$

### 2.3 Concrete KV Cache Memory Savings in Production
In DeepSeek-V3 ($d_c = 512, d_R = 64, N_h = 128, d_h = 128$):
- **Standard MHA Cache per Token/Layer**: $2 \times N_h \times d_h \times 2 \text{ bytes} = 2 \times 128 \times 128 \times 2 = \mathbf{65,536 \text{ bytes}}$
- **Standard GQA ($N_{kv}=8$)**: $2 \times 8 \times 128 \times 2 = \mathbf{4,096 \text{ bytes}}$
- **MLA Cache per Token/Layer**: $(d_c + d_R) \times 2 \text{ bytes} = (512 + 64) \times 2 = \mathbf{1,152 \text{ bytes}}$

$$\text{Memory Compression vs MHA} = \frac{65536}{1152} \approx \mathbf{56.9\times}, \quad \text{vs GQA} = \frac{4096}{1152} \approx \mathbf{3.55\times}$$

---

## 3. Rotary Position Embedding (RoPE) & YaRN Scaling

### 3.1 Mathematical Derivation of RoPE (Su et al., 2021)

Given vector $x \in \mathbb{R}^d$ at sequence position $m$, RoPE partitions $x$ into $\frac{d}{2}$ two-dimensional pairs $[x_{2i-1}, x_{2i}]$ and applies a 2D Givens rotation matrix:

$$ R_{\Theta, m}^{(i)} = \begin{bmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{bmatrix} \quad \text{where } \theta_i = \Theta^{-2(i-1)/d}, \quad \Theta = 10000 \text{ (or } 500000 \text{)} $$

In complex notation, let $z_i = x_{2i-1} + i x_{2i} \in \mathbb{C}$:
$$ R_{\Theta, m}(z_i) = z_i \cdot e^{i m \theta_i} $$

#### Inner Product Relative Position Preservation Proof:
Let $q$ be at position $m$ and $k$ be at position $n$:
$$ \tilde{q}_m = q e^{i m \theta}, \quad \tilde{k}_n = k e^{i n \theta} $$
The inner product in the complex plane is given by $\langle \tilde{q}_m, \tilde{k}_n \rangle = \text{Re}(\tilde{q}_m \tilde{k}_n^*)$ (where $*$ denotes the complex conjugate):
$$ \langle \tilde{q}_m, \tilde{k}_n \rangle = \text{Re}\left( (q e^{i m \theta}) (k e^{i n \theta})^* \right) = \text{Re}\left( q k^* e^{i m \theta} e^{-i n \theta} \right) = \text{Re}\left( q k^* e^{i (m - n) \theta} \right) $$

$$\mathbf{\langle R_{\Theta, m} q, R_{\Theta, n} k \rangle = g(q, k, m - n)}$$

**Conclusion**: The attention score between $q$ and $k$ depends **strictly on their relative distance $(m - n)$**, preserving translation invariance across arbitrary sequence lengths.

---

### 3.2 Long-Context RoPE Scaling: YaRN (Yet another RoPE extensioN)

When extending context from $L_{\text{train}} = 4\text{k}$ to $L_{\text{test}} = 128\text{k}$, naive linear interpolation divides frequencies by $s = \frac{L_{\text{test}}}{L_{\text{train}}} = 32$: $\theta_i' = \frac{\theta_i}{s}$.  
*The Flaw*: High-frequency components (small $i$, capturing local grammar) lose resolution, destroying the model's short-context capabilities.

#### The YaRN Solution (Peng et al., 2023):
Partitions frequency spectrum into three distinct regimes based on wavelength $\lambda_i = \frac{2\pi}{\theta_i}$:
1. **High Frequencies ($\lambda_i < r_{\text{low}}$)**: No interpolation ($\theta_i' = \theta_i$). Preserves local token relationships.
2. **Low Frequencies ($\lambda_i > r_{\text{high}}$)**: Full linear interpolation ($\theta_i' = \frac{\theta_i}{s}$). Extrapolates long distances.
3. **Mid Frequencies ($r_{\text{low}} \leq \lambda_i \leq r_{\text{high}}$)**: Smooth ramp interpolation via blending factor $\gamma_i = \frac{\lambda_i - r_{\text{low}}}{r_{\text{high}} - r_{\text{low}}}$:
   $$ \theta_i' = (1 - \gamma_i) \theta_i + \gamma_i \frac{\theta_i}{s} $$

- **Temperature / Entropy Correction Factor ($\sqrt{t}$)**:
  Because longer contexts increase attention entropy ($H(A) \propto \log L$), YaRN scales the query-key dot product by temperature factor $t = 0.1 \ln(s) + 1$:
  $$ \text{Attention} = \text{Softmax}\left( \frac{Q K^T}{\sqrt{t} \sqrt{d_k}} \right) V $$

---

## 4. Feed-Forward Networks (FFN): SwiGLU Mechanics

Standard Transformer FFN (ReLU):
$$ \text{FFN}_{\text{ReLU}}(x) = \max(0, x W_1 + b_1) W_2 + b_2 $$

Modern LLMs (LLaMA-3, Mistral, Gemma) replace ReLU with **SwiGLU (Swish Gated Linear Unit)** (Shazeer, 2020):
$$ \text{Swish}_\beta(z) = z \cdot \sigma(\beta z) = \frac{z}{1 + e^{-\beta z}} \quad (\text{with } \beta = 1 \implies \text{SiLU}) $$
$$ \text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}} \right) W_{\text{down}} $$

```
                       ┌──────────────────────┐
                       │   Hidden Vector x    │
                       └──────────┬───────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
     ┌─────────────────────────┐     ┌─────────────────────────┐
     │  Linear (W_gate) + SiLU │     │     Linear (W_up)       │
     └────────────┬────────────┘     └────────────┬────────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                        Element-wise Multiply ⊙
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │    Linear (W_down)      │
                     └─────────────────────────┘
```

#### Parameter Dimension Matching:
- In standard FFN: $W_1 \in \mathbb{R}^{d \times 4d}, W_2 \in \mathbb{R}^{4d \times d} \implies 8 d^2 \text{ parameters}$.
- In SwiGLU: Three matrices ($W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}}, W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$).
- To maintain exact parameter equivalence with standard $4d$ FFN:
  $$ 3 \cdot d \cdot d_{ff} = 8 d^2 \implies d_{ff} = \frac{8}{3} d \approx 2.667 d $$
  *Example*: In LLaMA-3 8B ($d_{\text{model}} = 4096$), $d_{ff} = 14336 = 3.5 d_{\text{model}}$ (rounded to nearest multiple of 256 for GPU alignment).

---

## 5. Mixture of Experts (MoE) & DeepSeek-V3 Load Balancing

### 5.1 MoE Routing Formulation

An MoE layer replaces the standard dense FFN with $N$ independent expert networks $\{E_1, E_2, \dots, E_N\}$, routing each token to the Top-$K$ experts via a gating network $G(x)$:

$$ y = \sum_{i \in \text{Top}K} g_i(x) E_i(x) $$

- **Standard Top-$K$ Softmax Gating**:
  $$ H(x) = x W_g \in \mathbb{R}^N $$
  $$ \text{Top}K(H(x)) = \text{Indices of } K \text{ largest values in } H(x) $$
  $$ g_i(x) = \begin{cases} \frac{\exp(H(x)_i)}{\sum_{j \in \text{Top}K} \exp(H(x)_j)} & \text{if } i \in \text{Top}K(H(x)) \\ 0 & \text{otherwise} \end{cases} $$

---

### 5.2 The Routing Collapse Problem & Auxiliary Losses

Without constraints, the gating network quickly suffers from **Expert Routing Collapse** — routing all tokens to 2 or 3 favorite experts while the remaining $N-K$ experts starve and receive zero gradient updates.

#### Classical Solution: Auxiliary Load Balancing Loss (Switch Transformer / Mixtral):
$$ \mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^N f_i P_i $$
Where:
- $f_i = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(\text{token } t \text{ routed to expert } i)$ (fraction of tokens dispatched)
- $P_i = \frac{1}{T} \sum_{t=1}^T g_i(x_t)$ (average routing probability)
- $\alpha \approx 0.01$ (hyperparameter).

*The Problem*: $\mathcal{L}_{\text{aux}}$ acts as a regularizer that penalizes routing specialization, degrading the overall model capability.

---

### 5.3 DeepSeek-V3 Auxiliary-Loss-Free Load Balancing

DeepSeek-V3 introduces **bias-driven load balancing** without adding an auxiliary loss term to the objective:

$$ \text{Gating Score: } s_{i, t} = \text{Softmax}(x_t W_g)_i + b_i $$
Where $b_i \in \mathbb{R}$ is a dynamic expert bias updated online during training:
$$ b_i \leftarrow b_i + \gamma \cdot \left( \frac{1}{N} - \frac{\text{Tokens Dispatched to Expert } i}{\text{Total Tokens in Batch}} \right) $$

- If Expert $i$ is overloaded ($f_i > \frac{1}{N}$), $b_i$ is decreased, shifting future tokens away.
- If Expert $i$ is starved ($f_i < \frac{1}{N}$), $b_i$ is increased, naturally pulling tokens in.
- **Advantage**: Gradients directly optimize primary language modeling loss with zero interference from artificial auxiliary balance penalties.

---

## 6. PyTorch Implementation: Multi-Head Latent Attention (MLA)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) as introduced in DeepSeek-V2 / DeepSeek-V3.
    Features:
    - Low-rank KV compression into latent vector c_t^{KV}
    - Decoupled Rotary Positional Embeddings (RoPE)
    - Low-rank Query compression
    """
    def __init__(
        self,
        d_model: int = 2048,
        num_heads: int = 16,
        d_head: int = 128,
        d_c_kv: int = 512,    # KV compression dimension
        d_c_q: int = 512,     # Query compression dimension
        d_rope: int = 64      # Decoupled RoPE dimension
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_c_kv = d_c_kv
        self.d_c_q = d_c_q
        self.d_rope = d_rope
        self.scale = 1.0 / math.sqrt(d_head + d_rope)

        # 1. Query Compression & Projections
        self.w_dq = nn.Linear(d_model, d_c_q, bias=False)
        self.w_uq = nn.Linear(d_c_q, num_heads * d_head, bias=False)
        self.w_qr = nn.Linear(d_c_q, num_heads * d_rope, bias=False)

        # 2. Key-Value Compression & Projections
        self.w_dkv = nn.Linear(d_model, d_c_kv, bias=False)
        self.w_uk = nn.Linear(d_c_kv, num_heads * d_head, bias=False)
        self.w_uv = nn.Linear(d_c_kv, num_heads * d_head, bias=False)
        self.w_kr = nn.Linear(d_model, d_rope, bias=False) # Shared RoPE key

        # 3. Output Projection
        self.w_out = nn.Linear(num_heads * d_head, d_model, bias=False)

    def apply_rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Simple 2D rotary embedding for illustration
        B, S, H, D = x.shape
        half_d = D // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half_d, dtype=torch.float32, device=x.device) / half_d
        )
        angles = pos.unsqueeze(-1) * freqs.unsqueeze(0) # [S, half_d]
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(2) # [1, S, 1, half_d]
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(2) # [1, S, 1, half_d]
        
        x1, x2 = x[..., :half_d], x[..., half_d:]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated

    def forward(self, x: torch.Tensor, kv_cache_latent=None, kv_cache_rope=None):
        B, S, _ = x.shape
        pos = torch.arange(S, device=x.device)

        # --- Query Processing ---
        c_q = self.w_dq(x) # [B, S, d_c_q]
        q_c = self.w_uq(c_q).view(B, S, self.num_heads, self.d_head)
        q_r = self.w_qr(c_q).view(B, S, self.num_heads, self.d_rope)
        q_r = self.apply_rope(q_r, pos)
        q = torch.cat([q_c, q_r], dim=-1) # [B, S, H, d_head + d_rope]

        # --- KV Compression & Latent Generation ---
        c_kv = self.w_dkv(x) # [B, S, d_c_kv] (This is what gets stored in the KV cache!)
        k_r = self.w_kr(x).unsqueeze(2) # [B, S, 1, d_rope]
        k_r = self.apply_rope(k_r, pos).expand(B, S, self.num_heads, self.d_rope)

        # Uncompress K and V on-the-fly
        k_c = self.w_uk(c_kv).view(B, S, self.num_heads, self.d_head)
        v = self.w_uv(c_kv).view(B, S, self.num_heads, self.d_head)
        k = torch.cat([k_c, k_r], dim=-1) # [B, S, H, d_head + d_rope]

        # --- Attention Computation ---
        # Permute for batch matrix multiply: [B, H, S_q, S_k]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply Causal Mask
        mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(1)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v) # [B, H, S, d_head]

        # Concatenate heads and project output
        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.d_head)
        return self.w_out(out), c_kv, k_r

if __name__ == "__main__":
    B, S, D = 2, 64, 2048
    x = torch.randn(B, S, D)
    mla = MultiHeadLatentAttention(d_model=D, num_heads=16, d_head=128, d_c_kv=512, d_rope=64)
    out, latent_kv, rope_k = mla(x)
    assert out.shape == (B, S, D), "MLA output shape mismatch!"
    assert latent_kv.shape == (B, S, 512), "Latent KV shape mismatch!"
    print("Multi-Head Latent Attention (MLA) test passed successfully.")
```

---

## 7. Deep Interview Interrogation Ladder

- **Level 1 (Concept)**: What is the primary operational difference between MHA, GQA, and MLA?
- **Level 3 (Proof)**: Prove mathematically why the variance of Query-Key dot products equals $d_k$, and why Softmax temperature scaling $\frac{1}{\sqrt{d_k}}$ prevents vanishing gradients.
- **Level 5 (Mechanics)**: Why does Rotary Position Embedding (RoPE) preserve translation invariance in the inner product? (Show the complex exponential proof).
- **Level 7 (Architectural Sizing)**: Calculate the exact KV cache memory reduction when switching a 70B parameter model from GQA ($N_{kv}=8$) to MLA ($d_c=512, d_R=64$).
- **Level 9 (MoE Dynamics)**: Explain why standard auxiliary load balancing losses degrade model capability in MoE training, and how DeepSeek-V3's dynamic expert bias term solves this without auxiliary losses.
- **Level 10 (Principal Engineering)**: Walk through how you would optimize the kernel execution of Multi-Head Latent Attention (MLA) during decode. How does matrix associativity allow absorbing the up-projection matrix $W^{UK}$ into the Query vector before attending to the cached latent $c_t^{KV}$?

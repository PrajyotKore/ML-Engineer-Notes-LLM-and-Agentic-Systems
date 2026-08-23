# 06_DATA_AND_SYNTHETIC_DATA — Mathematical & Mechanistic Reference

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide an exhaustive mathematical and algorithmic reference on data engineering for LLMs — covering MinHash LSH deduplication proofs, Model Collapse dynamics, decontamination metrics, synthetic data generation pipelines, and closed-loop production flywheels.

---

## 1. Deduplication Mathematics: MinHash & Locality Sensitive Hashing (LSH)

Comparing all pairs in a dataset of $N = 10^8$ documents requires $\binom{N}{2} = \frac{N(N-1)}{2} \approx 5 \times 10^{15}$ comparisons, which is computationally intractable ($O(N^2)$). MinHash with LSH reduces this to **$O(N)$ linear time**.

### 1.1 Jaccard Similarity on $k$-Shingles
Let document $D$ be represented as a set of $k$-shingles (character or token $k$-grams) $S(D)$.  
The Jaccard Similarity between documents $A$ and $B$ is:

$$ J(A, B) = \frac{|S(A) \cap S(B)|}{|S(A) \cup S(B)|} \in [0, 1] $$

---

### 1.2 The MinHash Theorem

Let $\pi$ be a random permutation of the universal vocabulary of shingles $\mathcal{U}$.  
Define the MinHash function $h_\pi(S) = \min_{s \in S} \pi(s)$.

#### Theorem:
$$ \mathbb{P}\left[ h_\pi(S(A)) = h_\pi(S(B)) \right] = J(A, B) $$

#### Step-by-Step Proof:
Let the universe of shingles $\mathcal{U}$ be partitioned into three disjoint sets:
1. $X$: Shingles present in both $A$ and $B$ ($S(A) \cap S(B)$). $|X| = |S(A) \cap S(B)|$.
2. $Y$: Shingles present in $A$ but not $B$ ($S(A) \setminus S(B)$).
3. $Z$: Shingles present in $B$ but not $A$ ($S(B) \setminus S(A)$).
4. Shingles in neither $A$ nor $B$ do not affect $h_\pi(S(A) \cup S(B))$.

The union size is $|S(A) \cup S(B)| = |X| + |Y| + |Z|$.  
Under a uniform random permutation $\pi$, every element in $S(A) \cup S(B)$ has an equal probability of being mapped to the minimum value.  
The event $h_\pi(S(A)) = h_\pi(S(B))$ occurs if and only if the minimum element in $S(A) \cup S(B)$ falls into region $X$ (the intersection):
$$ \mathbb{P}\left[ h_\pi(S(A)) = h_\pi(S(B)) \right] = \frac{|X|}{|X| + |Y| + |Z|} = \frac{|S(A) \cap S(B)|}{|S(A) \cup S(B)|} = J(A, B) $$

*Implication*: By computing $K$ independent MinHash functions $[h_1(A), \dots, h_K(A)]$, the empirical fraction of matching hash values is an **unbiased estimator** of $J(A, B)$.

---

### 1.3 Locality Sensitive Hashing (LSH) S-Curve Analysis

To avoid all-to-all hash signature comparisons, we divide the $K$ hash values into $b$ bands of $r$ rows ($K = b \cdot r$).

```
Document Signature (K = b · r hashes)
┌──────────────┐ ─── Band 1 (r hashes) ───► Hash to Bucket (exact match required)
├──────────────┤ ─── Band 2 (r hashes) ───► Hash to Bucket
├──────────────┤ ─── ...
└──────────────┘ ─── Band b (r hashes) ───► Hash to Bucket
```

Two documents $A$ and $B$ with Jaccard similarity $s = J(A, B)$ become candidate duplicates if they match in **at least one band**:
1. Probability of all $r$ hashes matching in a single band: $s^r$
2. Probability of not matching in a single band: $1 - s^r$
3. Probability of not matching in all $b$ bands: $(1 - s^r)^b$
4. Probability of matching in at least one band (Candidate Pair):
   $$ \mathbf{P(\text{Candidate Pair}) = 1 - (1 - s^r)^b} $$

```
   P(Candidate)
   1.0 ┼               ┌─────────────
       │              /
   0.5 ┼─────────────┼─ (Threshold s*)
       │            /
   0.0 ┼───────────┘
       └─────────────┼───────────────► Jaccard Similarity s
                    s* = (1/b)^(1/r)
```

- **The Characteristic Threshold $s^*$**:
  $$ 1 - (1 - (s^*)^r)^b = 0.5 \implies (1 - (s^*)^r)^b = 0.5 \implies s^* \approx \left(\frac{1}{b}\right)^{1/r} $$
  *Example*: Setting $b = 20$ bands and $r = 5$ rows ($K = 100$ hashes) yields $s^* = (1/20)^{1/5} \approx 0.549$. Pairs with similarity $> 0.7$ have $> 98\%$ probability of collision, while pairs with $< 0.3$ similarity have $< 0.1\%$ collision rate.

---

## 2. Model Collapse: The Information-Theoretic Hazard

### 2.1 Mathematical Formulation of Model Collapse (Shumailov et al., 2024)

Consider a recursive data flywheel where generation $n+1$ is trained exclusively on synthetic outputs produced by generation $n$:
$$ P_{n+1}(x) = \arg\min_Q \mathbb{D}_{\text{KL}}\left( \hat{P}_n(x) \parallel Q(x) \right) $$

Let the true data distribution be a Gaussian $P_0(x) = \mathcal{N}(\mu_0, \sigma_0^2)$.  
At each generation $n$, the model samples a finite dataset $\mathcal{D}_n = \{x_1, \dots, x_{M_n}\} \sim P_n(x)$ and fits parameters $\mu_{n+1}, \sigma_{n+1}^2$.

#### Mean and Variance Dynamics:
1. **Mean Drift**:
   $$ \mu_{n+1} = \frac{1}{M_n} \sum_{i=1}^{M_n} x_i \implies \mathbb{E}[\mu_{n+1}] = \mu_0, \quad \text{Var}(\mu_n) = \sigma_0^2 \sum_{k=0}^{n-1} \frac{1}{M_k} $$
   As generations progress ($n \to \infty$), the variance of the estimated mean grows linearly with generation count: $\text{Var}(\mu_n) \to \infty$. The model drifts far away from the true distribution mean.

2. **Variance Shrinkage & Tail Loss**:
   $$ \mathbb{E}[\sigma_{n+1}^2] = \sigma_n^2 \left( 1 - \frac{1}{M_n} \right) = \sigma_0^2 \prod_{k=0}^n \left( 1 - \frac{1}{M_k} \right) $$
   As $n \to \infty$:
   $$ \lim_{n \to \infty} \sigma_n^2 = 0 $$

$$\mathbf{\lim_{n \to \infty} P_n(x) = \delta(x - \mu_\infty)}$$

**Conclusion**: The model distribution collapses to a **Dirac delta function (point mass)**. The model loses all linguistic and conceptual diversity, repeating only the most generic, high-probability tokens.

---

### 2.2 Mitigation: Anchored Synthetic Flywheels
To prevent model collapse:
1. **Human Anchor**: Mix synthetic data with a fixed minimum fraction ($\geq 20\%$) of pristine human data.
2. **Rejection Sampling / Verifiers**: Filter synthetic generation using deterministic code execution sandboxes or high-precision unit tests rather than raw unconditional generations.

---

## 3. Benchmark Decontamination Mathematics

If evaluation benchmark questions leak into the pre-training or fine-tuning corpus, evaluation benchmarks measure **memorization** rather than generalization.

### 3.1 $N$-Gram Bloom Filter Decontamination

Given test benchmark documents $\mathcal{D}_{\text{eval}}$, extract all 13-token shingles $\mathcal{S}_{\text{eval}}$.  
Construct a **Bloom Filter** of $m$ bits and $k$ independent hash functions:
$$ k = \frac{m}{|S_{\text{eval}}|} \ln 2 $$
The theoretical False Positive Rate is:
$$ p_{\text{FP}} = \left( 1 - e^{-k |S_{\text{eval}}| / m} \right)^k $$

A training document $D_{\text{train}}$ is flagged and removed if:
$$ \frac{|\{s \in S(D_{\text{train}}) : s \in \text{BloomFilter}(\mathcal{S}_{\text{eval}})\}|}{|S(D_{\text{train}})|} > \tau_{\text{overlap}} \quad (\tau_{\text{overlap}} = 0.05) $$

---

## 4. PyTorch & Python Implementation: MinHash LSH and Bloom Filter

```python
import hashlib
import numpy as np

class MinHashLSH:
    """
    MinHash with Locality Sensitive Hashing for document deduplication.
    """
    def __init__(self, num_hashes: int = 100, num_bands: int = 20):
        assert num_hashes % num_bands == 0
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        
        # Random hash coefficients: (a * x + b) % prime
        np.random.seed(42)
        self.a = np.random.randint(1, 2**31 - 1, size=num_hashes, dtype=np.int64)
        self.b = np.random.randint(0, 2**31 - 1, size=num_hashes, dtype=np.int64)
        self.prime = 2147483647 # 2^31 - 1 (Mersenne prime)

    def _get_shingle_hashes(self, text: str, k: int = 5) -> np.ndarray:
        words = text.lower().split()
        shingles = [" ".join(words[i:i+k]) for i in range(max(1, len(words) - k + 1))]
        hashes = np.array([int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16) for s in shingles], dtype=np.int64)
        return hashes

    def compute_minhash(self, text: str) -> np.ndarray:
        shingle_hashes = self._get_shingle_hashes(text)
        if len(shingle_hashes) == 0:
            return np.zeros(self.num_hashes, dtype=np.int64)
        
        # Matrix multiply: (num_hashes, len(shingles))
        all_hashes = (np.outer(self.a, shingle_hashes) + self.b[:, None]) % self.prime
        return np.min(all_hashes, axis=1)

    def get_band_buckets(self, signature: np.ndarray) -> list:
        buckets = []
        for i in range(self.num_bands):
            start = i * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = signature[start:end]
            band_hash = hash(tuple(band_slice))
            buckets.append((i, band_hash))
        return buckets

if __name__ == "__main__":
    lsh = MinHashLSH(num_hashes=100, num_bands=20)
    doc1 = "The Transformer attention mechanism uses scaled dot product attention for language modeling."
    doc2 = "The Transformer attention mechanism uses scaled dot product attention for sequence modeling."
    doc3 = "Convolutional neural networks apply kernel filters over spatial image grids for computer vision."
    
    sig1 = lsh.compute_minhash(doc1)
    sig2 = lsh.compute_minhash(doc2)
    sig3 = lsh.compute_minhash(doc3)
    
    jaccard_1_2 = np.mean(sig1 == sig2)
    jaccard_1_3 = np.mean(sig1 == sig3)
    
    print(f"Estimated Jaccard(Doc1, Doc2): {jaccard_1_2:.2f} (Near Duplicate)")
    print(f"Estimated Jaccard(Doc1, Doc3): {jaccard_1_3:.2f} (Dissimilar)")
```

---

## 5. Deep Interview Interrogation Ladder

- **Level 1 (Concept)**: What is the Jaccard similarity between two sets of text shingles?
- **Level 3 (Proof)**: Prove mathematically why the probability of two documents having identical MinHash values equals their exact Jaccard similarity.
- **Level 5 (LSH Derivation)**: Derive the characteristic S-curve threshold $s^* = (1/b)^{1/r}$ in Locality Sensitive Hashing.
- **Level 7 (Model Collapse)**: Explain mathematically why recursive training on synthetic data causes the variance of the learned distribution to converge to zero.
- **Level 9 (Decontamination)**: Design a petabyte-scale deduplication and benchmark decontamination pipeline using PySpark, MinHash LSH, and Bloom Filters for a 15-trillion token pre-training run.
- **Level 10 (Principal Engineering)**: You observe that after 3 iterations of fine-tuning an agent on its own tool-use trajectories, the model's tool error recovery rate drops from 85% to 20%. Diagnose the mathematical root cause and architect a closed-loop data flywheel that fixes it.

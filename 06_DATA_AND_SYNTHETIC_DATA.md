# 06_DATA_AND_SYNTHETIC_DATA — Technical Reference

## 1. Role Relevance
For an ML Engineer (LLM & Agentic Systems), data is the highest-leverage lever to improve model and agent performance. Algorithmic tweaks rarely beat high-quality data. You must know how to build a data flywheel, generate synthetic data correctly without model collapse, and rigorously deduplicate/decontaminate datasets.

## 2. Prerequisites
- Supervised Fine-Tuning (SFT) objective.
- Embeddings and cosine similarity.
- MinHash / LSH (Locality Sensitive Hashing).

## 3. First Principles
Models learn exactly what they are trained on. "Garbage in, garbage out" scales exponentially with LLMs. The quality of pre-training and post-training data fundamentally dictates the model's ceiling. Data engineering for ML requires statistical rigor to prevent biases and leakage.

## 4. Mechanistic Breakdown
### Data Pipeline Stages
1. **Collection**: Scraping, human annotation (RLHF), telemetry.
2. **Filtering**: Removing HTML, short lines, low-quality documents (using heuristics or classifier models).
3. **Deduplication**: Removing exact and near-duplicate documents.
4. **Decontamination**: Ensuring no overlap between training data and evaluation benchmarks.
5. **Mixing**: Setting the correct ratios of Code vs. Math vs. General Text.

### Synthetic Data Generation
Using a larger "teacher" model (e.g., GPT-4) to generate training data for a smaller "student" model (e.g., LLaMA 8B).
**Methods**:
- **Rejection Sampling**: Generate $N$ responses, use a reward model to pick the best 1, and use that for SFT.
- **Instruction Evolution**: Prompting the teacher to rewrite a simple prompt into a highly complex, multi-constraint prompt.

## 5. Mathematical Foundations
### MinHash for Near-Deduplication
Comparing all $N$ documents to each other is $O(N^2)$, which is impossible for a trillion-token corpus.
MinHash estimates the Jaccard Similarity:
$$ J(A, B) = \frac{|A \cap B|}{|A \cup B|} $$
By hashing the n-grams of documents and keeping the minimum hash values, we can group similar documents in $O(N)$ time using Locality Sensitive Hashing (LSH).

### Model Collapse
If a model trains recursively on its own synthetic outputs over generations $t$:
$$ P_t(x) = \text{Model trained on data from } P_{t-1}(x) $$
The tails of the distribution disappear. The model converges to a point mass of highly generic, repetitive text. Thus, synthetic data *must* be anchored by fresh human data or highly curated, diverse seed prompts.

## 6. Implementation
**Data Leakage Prevention:**
If the test set (e.g., HumanEval) leaks into the training set, the model will achieve 100% accuracy but fail in production.
```python
def decontaminate(train_dataset, eval_dataset, n_gram_size=13):
    eval_ngrams = build_ngram_index(eval_dataset, n_gram_size)
    clean_train = []
    for doc in train_dataset:
        if not contains_overlap(doc, eval_ngrams):
            clean_train.append(doc)
    return clean_train
```

## 7. Computational Complexity
- **Deduplication**: MinHash LSH requires massive distributed CPU clusters (e.g., PySpark) to process petabytes of text.
- **Synthetic Generation**: Highly compute-intensive on GPUs. Generating 1 million synthetic conversations requires massive inference budgets.

## 8. Hardware / GPU Behavior
- Data loading during training must not bottleneck the GPU. If `num_workers` in PyTorch's DataLoader is too low, the GPU utilization drops to 0% while waiting for CPU RAM to feed the next batch over the PCIe bus.

## 9. Production Architecture
**The Production Data Flywheel:**
1. Agent deployed to production.
2. User provides explicit feedback (thumbs up/down) or implicit feedback (user abandoned the workflow).
3. Trajectory is logged to the Data Lake.
4. Offline pipeline scores the trajectory.
5. High-quality trajectories (where the agent self-corrected and succeeded) are added to the continuous SFT dataset.

## 10. Scalability & Bottlenecks
- **Human Annotation Bottleneck**: Human labelers are slow and expensive. Synthetic data is fast but suffers from quality degradation. The scalable solution is LLM-as-a-Judge to score synthetic generation.

## 11. Failure Modes
- **Formatting Overfitting**: If all synthetic data ends with "Is there anything else I can help you with?", the model will learn this exact string and append it to every response in production.
- **Contamination**: A user pastes an open-source evaluation benchmark into a GitHub issue, your scraper picks it up, and your model "memorizes" the test set.

## 12. Debugging
- **Model degradation after SFT**: Often caused by a sudden shift in data mixtures. If you add 10x more coding data, the model's conversational ability will mathematically drop due to catastrophic forgetting. Use perplexity checks on held-out diverse datasets to catch this.

## 13. Principal-Level Reasoning
"In this role, I would not blindly generate 10 million synthetic tool-use examples. I would carefully design the seed prompts to cover the long tail of edge cases (e.g., API timeouts, schema mismatches). I would enforce a strict decontamination pipeline before every SFT run, using a 13-gram MinHash filter against all our production evaluation sets to ensure our progress is real, not memorized."

## 14. Interview Interrogation
- *Level 2*: What is data leakage?
- *Level 4*: Why do we use MinHash instead of direct string comparison for deduplication?
- *Level 7*: Mathematically, why does Model Collapse happen when training exclusively on synthetic data?
- *Level 9*: Your model's performance on the internal agent benchmark skyrocketed from 40% to 90%, but user satisfaction dropped. Walk me through your data pipeline investigation.
- *Level 10*: Architect a closed-loop data flywheel that automatically turns production failures into synthetic SFT data for the next model version.

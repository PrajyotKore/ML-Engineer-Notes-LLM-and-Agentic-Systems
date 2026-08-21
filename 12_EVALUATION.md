# 12_EVALUATION — Technical Reference

## 1. Role Relevance
For an A1 Technical Lead, evaluation is the absolute truth of the system. Without a rigorous, statistically sound evaluation framework, you are flying blind. You cannot improve what you cannot measure. You must distinguish between Model Quality, System Quality, and Product Quality.

## 2. Prerequisites
- Statistics (p-values, Confidence Intervals, A/B Testing).
- SFT and RLHF objective functions.
- RAG (Retrieval-Augmented Generation) metrics.

## 3. First Principles
Evaluation must answer: "Did this change actually make the product better?"
Because LLMs are generative and open-ended, exact-match metrics (like BLEU or ROUGE) are useless. We must use probabilistic evaluation (LLM-as-a-judge) backed by human baselines, followed by strict online A/B testing.

## 4. Mechanistic Breakdown
### The Three Layers of Evaluation
1. **Model Evaluation (Offline)**: Perplexity, HumanEval, MMLU. (Did the base weights get smarter?)
2. **System/Agent Evaluation (Offline)**: Execution success rate, tool-calling accuracy, retrieval precision (RAG). (Did the ReAct loop work?)
3. **Product Evaluation (Online)**: Task completion rate, user retention, latency. (Did the user get what they wanted?)

### LLM-as-a-Judge
Using a strong model (GPT-4) to evaluate the outputs of the production model.
**Techniques**:
- **Pairwise Comparison**: "Which response is better, A or B?" (Reduces calibration issues).
- **Point-wise Scoring**: "Score this response 1-5 based on helpfulness."

## 5. Mathematical Foundations
### Statistical Significance in A/B Testing
When comparing the success rate of Model A ($p_a$) vs Model B ($p_b$), we use a two-proportion z-test.

$$ Z = \frac{p_b - p_a}{\sqrt{p(1-p)(\frac{1}{n_a} + \frac{1}{n_b})}} $$
Where $p$ is the pooled proportion.

*Implication*: If the baseline success rate is 90%, and you want to detect a 1% improvement with 95% confidence, you mathematically need thousands of samples. You cannot evaluate a model on 50 prompts and declare victory.

### Precision vs Recall in Agent Tool Use
- **Precision**: When the agent called `delete_file`, was it actually supposed to? (Safety).
- **Recall**: When the user asked to delete a file, did the agent actually call the tool? (Helpfulness).
$$ F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$

## 6. Implementation
**Pairwise LLM-as-Judge Prompt:**
```text
[System] You are an impartial judge. Evaluate Model A and Model B based on the user's request.
Consider tool accuracy, hallucination, and conciseness.
[User Request]: {prompt}
[Model A Trajectory]: {traj_A}
[Model B Trajectory]: {traj_B}
[Output Format]: Output only 'A', 'B', or 'TIE'.
```
*Crucial*: You must swap the order (evaluate A vs B, then B vs A) to mitigate the LLM's **Position Bias** (models often favor the first option).

## 7. Computational Complexity
- **Cost of Offline Evals**: Running a 10,000-prompt agentic benchmark requires 10,000 full workflow executions. If each workflow takes 5 steps, that is 50,000 LLM calls. This is incredibly expensive and slow.

## 8. Hardware / GPU Behavior
- **Evaluation Clusters**: Evals require massive parallel throughput. You typically spin up ephemeral inference clusters specifically for the nightly evaluation run to maximize batch size.

## 9. Production Architecture
**The Evaluation Pipeline:**
1. **Nightly Regression Run**: Every night, the `main` branch model is evaluated against the "Golden Dataset" of 1,000 complex historical prompts.
2. **Canary Deployment**: If offline evals pass, deploy the new model to 1% of live traffic.
3. **Shadow Mode**: The new model runs in parallel with the old model on live traffic. Its outputs are logged but not shown to the user. Evaluators compare the shadow outputs offline.

## 10. Scalability & Bottlenecks
- **Golden Dataset Degradation**: Over time, engineers inadvertently optimize the system to pass the Golden Dataset, causing "eval overfitting". The dataset must be continuously refreshed from live production distributions.

## 11. Failure Modes
- **Judge Hallucination**: The LLM-as-a-judge hallucinates a reason for penalizing a perfectly good response. (Mitigated by asking the judge to output a "Chain of Thought" before outputting the final score).
- **Length Bias**: LLM judges almost universally prefer longer responses, even if the user wanted a concise answer. (Mitigated by explicitly penalizing verbosity in the judge's prompt).

## 12. Debugging
- **Metric Mismatch**: Offline evals say the new model is 10% better, but online A/B tests show a 5% drop in user engagement. *Root cause*: Your offline dataset no longer represents what users are actually asking in production, or the offline eval did not measure latency, which caused users to abandon the workflow.

## 13. Principal-Level Reasoning
"I never trust a single metric. If a researcher tells me the new LoRA model improved agent success by 5%, my first question is: 'What was the confidence interval?'. My second question is: 'Did we measure position bias in the LLM judge?'. At A1, I would architect a three-tiered evaluation system: fast deterministic unit tests for tool parsing, heavy LLM-as-judge for reasoning quality, and strict online shadow-mode testing before any rollout."

## 14. Interview Interrogation
- *Level 2*: What is the difference between an offline evaluation and an A/B test?
- *Level 4*: Why do we swap the order of Model A and Model B in an LLM-as-judge prompt?
- *Level 6*: How do you calculate the statistical significance of a 2% improvement in task completion rate?
- *Level 8*: Why does an agent's offline success rate often fail to correlate with online user retention?
- *Level 10*: Design the complete evaluation infrastructure for A1's continuous delivery pipeline, ensuring no degraded agent is ever deployed to production.

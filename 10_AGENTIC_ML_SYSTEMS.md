# 10_AGENTIC_ML_SYSTEMS — Technical Reference

## 1. Role Relevance
A1 is explicitly building a proactive AI assistant. This requires moving beyond stateless LLM chat into Agentic systems. An ML Technical Lead must understand how to architect systems that reason, plan, execute real-world APIs, maintain persistent memory, and recover from inevitable probabilistic failures. This is a P0 track.

## 2. Prerequisites
- Autoregressive generation.
- Post-training (SFT/RLHF) for instruction following.
- JSON schemas and external API concepts.

## 3. First Principles
An LLM is a probabilistic text generator. An **Agent** is a system that wraps an LLM in a control loop, granting it access to tools (APIs), memory, and the ability to affect the real world. The core challenge is bridging the non-deterministic output of the LLM with the deterministic requirements of software execution.

## 4. Mechanistic Breakdown
### The Agent Loop (ReAct / Plan-and-Solve)
1. **Perception**: System prompt + User input + Retrieved memory + Current state.
2. **Reasoning (Thought)**: The LLM generates a "chain of thought" to plan the next step.
3. **Action (Tool Use)**: The LLM outputs a structured request (e.g., JSON) to invoke a specific tool.
4. **Execution**: The deterministic system parses the JSON, executes the API, and captures the result.
5. **Observation**: The API result is appended to the prompt, and the loop repeats until the LLM emits a `[FINAL_ANSWER]` token.

### Memory Architecture
- **Working Memory**: The current context window (conversation history, recent tool outputs). Extremely fast, but bounded by KV Cache memory constraints.
- **Episodic Memory**: Vector database containing specific past interactions. Retrieved via cosine similarity.
- **Semantic Memory**: Knowledge graphs or structured databases summarizing learned facts about the user.

## 5. Mathematical Foundations
### The Reliability Problem of Chained Reasoning
In an agentic workflow, a task requires $N$ sequential tool calls.
Let $p$ be the probability of the LLM successfully generating the correct tool call and formatting it properly.
The probability of the entire workflow succeeding without intervention is:

$$ P(\text{Success}) = p^N $$

*Implication*: If the LLM has a 90% success rate per step ($p = 0.9$), a 10-step workflow has a $0.9^{10} \approx 34.8\%$ chance of success.
This mathematical reality dictates that we cannot just make the LLM "smarter" to achieve five-nines (99.999%) reliability. We *must* build deterministic recovery systems (Reflection, Retries, Validation).

## 6. Implementation
**Tool Validation Boundary:**
```python
def execute_agent_step(llm_output):
    try:
        # 1. Parse JSON strictly
        tool_call = parse_strict_json(llm_output)
        
        # 2. Validate against JSON Schema (Deterministic)
        validate_schema(tool_call, schema)
        
        # 3. Execute Tool
        return invoke_api(tool_call)
    except SchemaValidationError as e:
        # REFLECTION: Feed the exact error back to the LLM to self-correct
        return f"System Error: {e}. Please correct your JSON and try again."
```

## 7. Computational Complexity
- **Latency Multiplication**: An agent step requires a full prefill and decode. A 5-step workflow incurs $5\times$ TTFT and $5\times$ TPOT. This severely limits synchronous real-time use cases.
- **Context Growth**: As the agent iterates, the context grows linearly. Because attention compute scales $O(L^2)$, the 5th step is much more computationally expensive than the 1st step.

## 8. Hardware / GPU Behavior
- **KV Cache Trashing**: Long-running agents rapidly fill the KV cache. If an agent goes to sleep waiting for an external API (e.g., waiting 5 minutes for an Uber), its KV cache should be offloaded from VRAM to CPU RAM, and restored upon API completion to free up the GPU for other agents.

## 9. Production Architecture
**Model Routing:**
Not all steps require a massive 70B parameter model.
- **Routing Node**: A fast, cheap 8B model acts as a triage router.
- **Execution Node**: For simple API formatting, use the 8B model.
- **Complex Planning**: If the task requires deep reasoning, the router escalates the prompt to the 70B model.
This optimizes cost and latency across the fleet.

## 10. Scalability & Bottlenecks
- **Context Limit Walls**: Even with 128k context windows, dumping raw API responses (e.g., full HTML of a webpage) will overwhelm the context. Information must be aggressively summarized or filtered by intermediate scripts before being presented to the agent's observation window.

## 11. Failure Modes
- **Hallucinated Tools**: The LLM tries to call a tool that does not exist in its prompt.
- **Infinite Loops**: The LLM makes a mistake, the system returns an error, the LLM makes the exact same mistake. It loops until the context limit is hit. (Requires a hard `max_retries` circuit breaker).
- **Tool Misuse**: The LLM calls a destructive tool (e.g., `delete_email`) with hallucinated arguments.

## 12. Debugging
- **Trajectory Analysis**: You cannot debug an agent by just looking at the final output. You must have an observability platform (like LangSmith or Phoenix) that captures the exact prompt, thought, action, and observation at every step $t$.

## 13. Trade-offs
- **ReAct vs Direct Tool Calling**: ReAct (forcing the LLM to output a "Thought" string before the "Action") drastically improves reasoning quality but increases latency and token cost.

## 14. Principal-Level Reasoning
"At A1, the ML execution layer must assume the LLM will fail. I treat the LLM as an unreliable heuristic engine inside a highly reliable state machine. If an agent loops on a tool error, I don't just prompt-engineer; I architect a validation layer that intercepts the failure, logs the trajectory for SFT data collection (so we can fine-tune the failure out), and uses a cheaper model to summarize the error before returning it to the main reasoning loop."

## 15. Interview Interrogation
- *Level 2*: What is the difference between Working Memory and Semantic Memory?
- *Level 4*: Why is JSON schema validation critical at the system boundary?
- *Level 7*: Mathematically prove why long workflows fail, and explain how to mitigate it.
- *Level 9*: Your agent is stuck in an infinite loop of failing to parse a complex API response. How do you architect a system to automatically break the loop and recover?
- *Level 10*: Design a multi-agent architecture for A1 where a "Planner Agent" delegates to "Execution Agents", including state handoffs, error bubbling, and KV cache optimization.

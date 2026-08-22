# 17_PYTHON_AND_CODING — Technical Reference

## 1. Role Relevance
An ML Engineer (LLM & Agentic Systems) must code. You are expected to write clean, concurrent Python and understand PyTorch tensor operations perfectly. Interviewers will ask you to implement core ML algorithms from scratch and solve production concurrency problems.

## 2. Core Python Implementations (PyTorch / NumPy)

### A. Numerically Stable Softmax
```python
import torch

def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Subtract max for numerical stability (prevents e^1000 overflow)
    max_vals, _ = torch.max(logits, dim=dim, keepdim=True)
    shifted_logits = logits - max_vals
    
    exp_logits = torch.exp(shifted_logits)
    sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)
    
    return exp_logits / sum_exp
```

### B. Scaled Dot-Product Attention
```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q, K, V shape: (batch_size, num_heads, seq_len, head_dim)
    d_k = Q.size(-1)
    
    # Compute attention scores: Q * K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        # Apply causal mask (set masked positions to -inf)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
    attention_weights = F.softmax(scores, dim=-1)
    
    # Multiply by V
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

### C. LoRA Parameter Update
```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.r = r
        self.alpha = alpha
        
        # Frozen base weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        
        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r)) # B initialized to 0
        
        self.scaling = self.alpha / self.r

    def forward(self, x):
        # Y = XW_0 + XA^T B^T * scaling
        base_output = F.linear(x, self.weight)
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_output + lora_output
```

## 3. Core Python Concurrency (Agent Runtimes)
Agent workflows require calling multiple tools in parallel (e.g., searching Google and searching Wikipedia simultaneously) and gathering the results.

### A. Async API Aggregation
```python
import asyncio
import aiohttp

async def call_tool(session, tool_name, payload):
    async with session.post(f"https://api.internal/{tool_name}", json=payload) as response:
        return await response.json()

async def parallel_tool_execution(tool_requests):
    """
    Executes multiple tools concurrently and waits for all to finish.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for req in tool_requests:
            tasks.append(call_tool(session, req['name'], req['payload']))
        
        # Gather executes all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

### B. Exponential Backoff Decorator
Critical for durable agent loops interacting with flaky APIs.
```python
import asyncio
import random

def retry_with_backoff(retries=3, base_delay=1.0):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise e
                    # Exponential backoff with jitter
                    sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 0.1)
                    await asyncio.sleep(sleep_time)
        return wrapper
    return decorator
```

## 4. Interview Strategy
- Always write types (`-> torch.Tensor`, `-> list`).
- State the tensor dimensions in comments above matrix multiplications.
- Handle edge cases (e.g., `return_exceptions=True` in `asyncio.gather` so one failed API call doesn't crash the parallel batch).

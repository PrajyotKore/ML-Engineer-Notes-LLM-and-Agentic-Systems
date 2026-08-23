# 19_LEADERSHIP_AND_TECHNICAL_JUDGMENT — Staff & Principal Reference

> **Audience**: ML Engineers, LLM Systems Engineers, and AI Researchers preparing for senior/principal technical interviews.  
> **Core Objective**: Provide structured frameworks for navigating multi-million dollar compute trade-offs, leading critical production incidents, managing technical debt, and defending high-stakes ML engineering decisions.

---

## 1. The Staff/Principal Trade-Off Matrix

```
                                  ┌───────────────────────────────┐
                                  │      The Architectural Triad  │
                                  │   Quality vs Latency vs Cost  │
                                  └───────────────┬───────────────┘
                                                  │
                 ┌────────────────────────────────┼────────────────────────────────┐
                 ▼                                ▼                                ▼
      [ Data vs Algorithm ]             [ Build vs Buy ]             [ Model Size vs Systems ]
      - Data curation (High ROI)        - Use Postgres / Temporal    - 8B + Speculative Decode vs
      - Custom loss (High Risk)         - Build custom Core IP       - 70B Dense Model
```

### 1.1 SFT vs. LoRA vs. In-Context Prompting
- **In-Context Prompting**: Fast prototyping, zero training compute, but high inference token cost and long TTFT.
- **LoRA / QLoRA**: Ideal when task adaptation requires structured format compliance, fast domain adaptation, and modular adapter switching on shared base weights.
- **Full SFT / Pre-training**: Mandatory only when shifting fundamental linguistic/reasoning representations or pre-training on domain-specific corpora (medical, legal, code).

---

## 2. Production Incident Leadership Protocol

When a critical production incident occurs (e.g. *"Autonomous Agent enters infinite loop ordering duplicate financial trades"*):

```
Step 1: Triage & Halt Bleed (Revert Canary / Disable Tool globally via feature flag)
   │ (Target: < 2 minutes | Do NOT attempt deep debugging yet!)
   ▼
Step 2: Stakeholder Communication (Quantify blast radius and declare mitigation)
   │
   ▼
Step 3: Root Cause Telemetry (Pull OpenTelemetry Traces, examine prompt/logits)
   │
   ▼
Step 4: Regression Hardening (Add failing trajectory to Golden Evaluation Suite)
   │
   ▼
Step 5: Blameless Post-Mortem & Architectural Remediation
```

---

## 3. Deep Interview Interrogation Ladder

- **Level 1 (Concept)**: When should an ML engineer say "NO" to fine-tuning a model?
- **Level 5 (Cost Optimization)**: You are tasked with cutting a $200k/month inference bill in half without degrading user retention. Walk through your hierarchical optimization plan.
- **Level 8 (Technical Strategy)**: The research team wants to implement a novel research activation function. The systems team wants to stick to standard SwiGLU. How do you arbitrate the decision?
- **Level 10 (Principal Engineering)**: You have a team of 4 engineers and 6 months of compute runway. Design the product and infrastructure roadmap to take a prototype agent from 60% reliability to a 99.5% enterprise SLA.

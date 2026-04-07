# Paper Comparison: RL for LLM Abstention and Uncertainty Management

**Papers compared:** 2
**Date:** 2026-04-07
**Category:** Verification / Abstention / Calibration

## Papers

1. **TruthRL** -- Wei et al., Meta, 2025. "TruthRL: Truthful and Reliable LLMs through Reinforcement Learning"
2. **Rewarding Doubt** -- Shayovitz & Rabinovich, ICLR 2026. "Rewarding Doubt: A Reinforcement Learning Approach to Confidence Calibration of LLMs"

---

## Comparison Table

| Dimension | TruthRL (Wei et al., 2025) | Rewarding Doubt (Shayovitz & Rabinovich, 2026) |
|-----------|---------------------------|------------------------------------------------|
| **Core Problem** | Reduce hallucinations by training LLMs to abstain when uncertain | Produce calibrated confidence scores so LLMs can selectively abstain |
| **Theoretical Basis** | Empirical reward engineering; ternary signal design | Proper scoring rules from calibration theory (log scoring rule) |
| **Reward Function** | Discrete ternary: +1 (correct), 0 (abstain), -1 (hallucinate). OOK variant: +1 for "I don't know" on unanswerable questions. | Continuous: R = log(p) if correct, R = log(1-p) if incorrect, where p = model's stated confidence. Theoretically optimal strategy is truthful confidence reporting. |
| **RL Algorithm** | GRPO (Group Relative Policy Optimization) | PPO (Proximal Policy Optimization) |
| **Abstention Mechanism** | Explicit generation of "I don't know" as a trained behavior | Threshold-based: model outputs confidence p; abstains if p < tau |
| **Uncertainty Expression** | Binary: answer or "I don't know" | Graded: continuous confidence score in [0, 1] |
| **Verifier / Judge** | LLM-as-a-judge (Llama-3.3-70B-Instruct) checks correctness against gold references | Binary correctness check (exact match or similar) feeds into scoring rule |
| **Training Data** | CRAG dataset. OOK labels obtained by sampling 256 responses (optional, for knowledge-enhanced variant). Default ternary reward does not require OOK labels. | QA datasets without explicit knowledge boundary labels |
| **Base Model** | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct (also tested on 3B and 32B scales) | Various LLMs (paper reports on multiple scales) |
| **Dataset / Evaluation** | CRAG, NaturalQuestions, HotpotQA, MuSiQue (4 benchmarks, retrieval and non-retrieval settings). T = Acc - Hall. LLM-as-a-judge evaluation. | Factoid QA benchmarks; ECE, selective prediction AUC, reliability diagrams, accuracy at coverage levels |
| **Key Result** | Avg T=25.6 (vs. -17.8 SFT, +4.5 binary RL), H=18.8% (vs. 58.9% SFT), across 4 benchmarks (CRAG, NQ, HotpotQA, MuSiQue). On CRAG alone: T=37.2. On hallucination-baiting: T=52.4, H=16.5%. | Significant ECE reduction; improved accuracy-coverage trade-offs vs. vanilla, temperature scaling, verbalized confidence |
| **Strengths** | Simple reward design; dramatic hallucination reduction; practical and deployable; failure mode analysis (Table 5) | Theoretically principled; graded uncertainty; no need for knowledge boundary labels; generalizable framework |
| **Limitations** | Binary abstention only; needs OOK labels; relies on LLM judge quality; single benchmark | Log scoring rule instability near extremes; PPO training cost; requires format changes for confidence output; threshold is a deployment hyperparameter |

---

## TruthRL Quantitative Results (from paper tables)

Since TruthRL provides extensive quantitative results while Rewarding Doubt focuses more on calibration metrics that are not directly comparable, the key TruthRL numbers are summarized here for reference:

| Setting | Method | T (Truthfulness) | H (Hallucination %) | A (Accuracy %) |
|---------|--------|-------------------|---------------------|-----------------|
| Avg 4 benchmarks, Llama-3.1-8B | SFT | -17.8 | 58.9 | 41.1 |
| Avg 4 benchmarks, Llama-3.1-8B | Binary RL | +4.5 | 47.7 | 52.2 |
| Avg 4 benchmarks, Llama-3.1-8B | **TruthRL (ternary)** | **+25.6** | **18.8** | 44.4 |
| CRAG only, Llama-3.1-8B | TruthRL | +37.2 | 19.4 | -- |
| CRAG hallucination-baiting | TruthRL | +52.4 | 16.5 | -- |
| CRAG, rule-based verifier | TruthRL | -3.6 | 3.6 | -- |
| Scalability, Qwen2.5-32B | TruthRL | +40.0 | 18.2 | -- |
| Difficult questions (OOK) | SFT | -- | 100% | -- |
| Difficult questions (OOK) | TruthRL | -- | 15.5% | -- |

The most striking result: on questions where almost no method produces correct answers, SFT hallucinates 100% of the time, while TruthRL abstains 84.5% and hallucinates only 15.5%.

---

## Detailed Analysis

### 1. Reward Design Philosophy

The fundamental difference is in how uncertainty is formalized as a reward signal.

**TruthRL** takes a pragmatic, engineering-driven approach: the reward is a discrete ternary signal where abstention is treated as a neutral action (reward = 0) that is strictly better than hallucination (reward = -1) but worse than being correct (reward = +1). This creates a clear incentive gradient: when the model is unsure, abstention is the safe default. For out-of-knowledge questions, abstention is actively rewarded (+1), creating an even stronger signal.

**Rewarding Doubt** takes a theoretically grounded approach rooted in decision theory. A proper scoring rule has the mathematical property that the expected reward is maximized *if and only if* the model reports its true belief probability. This means the reward function itself encodes the correct incentive structure -- no manual reward engineering is needed. The model is not told "abstain or don't abstain"; instead, it learns to report calibrated confidence, and abstention emerges as a downstream consequence of low confidence.

**Implication for TruthRL-personalized:** The ternary reward is simpler to implement and debug, but it forces a hard binary decision (answer vs. abstain). A hybrid approach could combine TruthRL's ternary outcome reward with a Rewarding Doubt-style confidence calibration component, producing models that both abstain when necessary and express graded uncertainty when answering.

### 2. Abstention Granularity

TruthRL's abstention is **all-or-nothing**: the model either answers or says "I don't know." There is no middle ground -- no "I am somewhat confident" or "I think the answer might be X but I'm not sure."

Rewarding Doubt's abstention is **graded**: the model produces a confidence score, and abstention is a continuous decision (any threshold can be applied). This is strictly more informative -- a confidence of 0.3 tells the user more than "I don't know." It also allows deployment-time flexibility: a safety-critical application can set a high threshold (tau = 0.9), while a casual assistant can use a lower one (tau = 0.5).

However, graded confidence comes at a cost: the model must learn to generate meaningful probability tokens, the output format is more complex, and miscalibrated confidence can be worse than binary abstention (a model that says "95% confident" and is wrong is more harmful than one that says "I don't know").

### 3. Data Requirements

TruthRL's default ternary reward (+1/0/-1) does **not** require OOK labels -- it works with any QA data where correctness can be verified. However, the knowledge-enhanced variant (which gives +1 for abstention on OOK questions) does require identifying which questions are outside the model's knowledge. TruthRL obtains these labels by sampling 256 responses per question and marking questions with zero correct responses as OOK -- a computationally expensive probing step. Notably, the ablation (Table 3) shows the knowledge-enhanced variant does not consistently improve over plain ternary, so OOK labels may be unnecessary in practice.

Rewarding Doubt requires **only correctness labels**: the scoring rule computes reward from (correctness, confidence) pairs without needing to know whether a question is "supposed to be" answerable. The model discovers its own knowledge boundary through the calibration process. This is more scalable but less controllable -- you cannot explicitly tell the model "these questions are outside your knowledge."

### 4. Training Stability

TruthRL uses GRPO, which is generally more stable than PPO for language model fine-tuning (no critic network, simpler optimization landscape). The ternary reward has bounded magnitude (-1 to +1), preventing gradient explosions.

Rewarding Doubt uses PPO with a log scoring rule that has **unbounded negative rewards**: log(p) approaches -infinity as p approaches 0. If the model is very confident (p near 1) and wrong, R = log(1-p) = log(nearly 0), producing an extremely large negative reward. This requires careful reward clipping or bounded confidence intervals to prevent training instability.

### 5. Evaluation Scope

Both papers are primarily evaluated on **factoid QA**, which is a narrow slice of LLM use cases. Neither demonstrates generalization to:
- Multi-step mathematical reasoning
- Code generation
- Long-form text generation
- Multi-turn dialogue
- Creative or subjective tasks

This is a shared limitation and an open research gap.

---

## Agreement Points

1. **RL is effective for teaching abstention/calibration.** Both papers convincingly demonstrate that RL fine-tuning can improve uncertainty management compared to SFT-only or post-hoc methods.
2. **The reward signal matters enormously.** Both papers show that naive approaches fail -- TruthRL's Table 5 shows rule-based verifier collapse, and Rewarding Doubt shows that uncalibrated baselines (vanilla, temperature scaling, verbalized confidence) are clearly inferior.
3. **There is a trade-off between accuracy and coverage.** Both papers operate on the same fundamental accuracy-coverage curve, just with different mechanisms for navigating it.

---

## Key Differences Summary

| Aspect | TruthRL | Rewarding Doubt |
|--------|---------|-----------------|
| Reward | Discrete ternary | Continuous proper scoring rule |
| Uncertainty | Binary (answer / refuse) | Graded (confidence score) |
| Theory | Empirical | Information-theoretic |
| RL Algorithm | GRPO | PPO |
| Data needs | Correctness labels (OOK labels optional, for knowledge-enhanced variant) | Correctness labels only |
| Stability | Bounded rewards, stable | Unbounded log penalties, needs clipping |
| Flexibility | Fixed abstention behavior | Tunable threshold at inference |

---

## Synthesis

These two papers represent complementary approaches to the same fundamental problem: teaching LLMs to manage uncertainty honestly rather than hallucinate.

**TruthRL** is the more practical, deployment-ready approach. Its ternary reward is simple to implement, GRPO is stable to train, and the resulting model behavior (answer or "I don't know") is easy for end users to understand. Its main weakness is the bluntness of binary abstention and the dependency on OOK labels and LLM judge quality.

**Rewarding Doubt** is the more theoretically principled approach. Proper scoring rules provide a formal guarantee that truthful confidence reporting is optimal, and graded confidence is strictly more informative than binary abstention. Its main weaknesses are training complexity (PPO + unbounded log rewards) and the requirement that models learn to generate meaningful confidence probabilities.

**The gap that remains** is the bridge between these approaches: a system that combines TruthRL's practical simplicity with Rewarding Doubt's theoretical grounding. Specifically:

1. **Hybrid reward:** Could the ternary reward be augmented with a calibration component? For example: ternary outcome reward + a scoring rule-based bonus/penalty for confidence expression.
2. **Graded abstention in the TruthRL framework:** Instead of "I don't know," the model could express "I am X% confident that..." while still receiving TruthRL's ternary outcome reward.
3. **Symbolic confidence indicators:** Could the symbolic process reward (as being developed in TruthRL-personalized) detect reasoning patterns that correlate with confidence, providing a formal bridge between the two approaches?
4. **Scaling to complex tasks:** Both approaches need evaluation beyond factoid QA -- particularly on reasoning tasks where uncertainty is more nuanced and harder to define.

---

## Relevance to TruthRL-personalized Project

This comparison directly informs the project's direction:

- The **symbolic process reward** being developed here occupies a unique position: it operates on reasoning structure (like Rewarding Doubt's focus on calibration signals) but uses formally decidable properties (unlike either paper's reward mechanisms). It could serve as a bridge, providing graded signals about reasoning quality that complement TruthRL's ternary outcome reward.
- Rewarding Doubt's **proper scoring rule framework** could be adapted for the symbolic process reward: instead of log(p), use a scoring function over symbolic reasoning properties (phase transitions, echo detection, n-gram degeneration) as a quasi-calibration signal.
- The project's advantage over both papers: **no LLM judge dependency** for the process reward component, and **formally verifiable** reasoning quality assessment.

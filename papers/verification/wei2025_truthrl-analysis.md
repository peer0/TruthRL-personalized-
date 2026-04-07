# Paper Analysis: TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning

**Authors:** Zhepei Wei, Xiao Yang, Kai Sun, Jiaqi Wang, Rulin Shao, Sean Chen, Mohammad Kachuee, Teja Gollapudi, Tony Liao, Nicolas Scheffer, Rakesh Wanga, Anuj Kumar, Yu Meng, Wen-tau Yih, Xin Luna Dong
**Year:** 2025
**Source:** arXiv:2509.25760v1 [cs.CL] (Meta Reality Labs / UVA / UW / FAIR)
**PDF path:** /Users/jay/projects/papers/abstention/Meta2025_TruthRL.pdf
**Analyzed:** 2026-04-07

## Key Claims

1. Accuracy-driven training methods (vanilla SFT, vanilla RL with binary reward) inherently incentivize LLMs to guess rather than abstain when uncertain, which amplifies hallucinations and compromises truthfulness.
2. A simple ternary reward scheme (+1 correct, 0 abstention, -1 hallucination) used with GRPO is sufficient to teach LLMs to balance accuracy and calibrated abstention, reducing hallucinations by up to 28.9% and improving truthfulness by up to 21.1% over vanilla RL.
3. The improvement arises from genuine knowledge boundary recognition, not over-conservatism: TruthRL abstains mostly on questions where the model genuinely lacks knowledge, while baselines hallucinate heavily on those same questions.
4. More complicated reward designs (knowledge-enhanced, reasoning-enhanced) generally do not outperform the simple ternary reward, suggesting that the key insight is the three-way distinction itself rather than additional signal complexity.
5. An LLM-based verifier is critical for stable RL training; rule-based (string matching) verifiers cause model collapse into over-conservative abstention (T=-3.6) because predicted answers rarely match reference answers in exact string form.

## Method

**Core framework:** TruthRL implements GRPO (Group Relative Policy Optimization) with a ternary reward function that explicitly distinguishes three response types:
- **Correct answer** -> reward +1
- **Abstention ("I don't know")** -> reward 0 (neutral)
- **Hallucination (incorrect answer)** -> reward -1

**Why ternary matters over binary:** Under binary reward (correct=+1, else=-1), abstentions and hallucinations receive the same -1 reward, yielding zero relative advantage between them in GRPO's group-normalized advantage computation. Under ternary reward, abstention (0) gets a higher advantage than hallucination (-1), explicitly incentivizing the model to abstain rather than guess when uncertain.

**GRPO objective:** For each question x, G responses are sampled from the old policy. The advantage for response y_i is computed as:

    A_hat_i = (r(x, y_i) - mean({r(x, y_j)})) / std({r(x, y_j)})

The clipped surrogate objective with KL regularization is then optimized:

    L_GRPO = -E[min(w * A_hat, clip(w, 1-eps, 1+eps) * A_hat) - beta * D_KL(pi_theta || pi_ref)]

**Knowledge boundary probing:** For each training question, 256 responses are sampled; a question is marked as out-of-knowledge (OOK) if none are correct. OOK questions are relabeled with "I don't know" as the target for SFT baselines (R-Tuning, RFT). For TruthRL, OOK labels are optionally used in a knowledge-enhanced reward variant where abstention on OOK questions gets +1 instead of 0.

**Reward variants explored:**
1. **Binary:** +1 correct, -1 otherwise (recovers vanilla RL)
2. **Ternary:** +1 correct, 0 abstain, -1 incorrect (TruthRL default)
3. **Knowledge-enhanced:** OOK questions get +1 for abstention, -1 for any answer; non-OOK uses base scheme
4. **Reasoning-enhanced:** Adds reasoning quality reward (r_reason) on top of outcome reward via multiplicative, additive, or conditional strategies

**Training setup:** 8x NVIDIA H100 GPUs, full-parameter fine-tuning via VeRL framework. GRPO with learning rate 1e-6, batch size 64, KL coefficient beta=0.001, clip ratio epsilon=0.2, max context 16,384 tokens, max generation 2,048 tokens. vLLM for rollouts (tensor parallel=2, GPU utilization=0.8). Rollout sampling: temperature=1.0, top-p=1.0. SFT/DPO baselines trained via Open-R1 with DeepSpeed ZeRO-3.

**Evaluation:** LLM-as-a-judge (Llama3.3-70B-Instruct) for answer correctness. Truthfulness score = w1*Acc + w2*Unc - w3*Hall, with w1=1, w2=0, w3=1 (so T = Acc - Hall). Greedy decoding for evaluation. Evaluated on CRAG, NaturalQuestions, HotpotQA, MuSiQue under both retrieval and non-retrieval settings.

**Answer format:** Reasoning in `<think></think>` tags, final answer in `\boxed{}`.

## Results

### Main Results (Table 1, Average across 4 benchmarks, With Retrieval)

| Metric | Prompting | SFT | RFT | R-Tuning | TruthRL_Binary | TruthRL |
|--------|-----------|-----|-----|----------|----------------|---------|
| **Llama3.1-8B-Inst** | | | | | | |
| Truthfulness (T) | -16.4 | -17.8 | -15.8 | -8.7 | 4.5 | **25.6** |
| Hallucination (H) | 54.1 | 58.9 | 53.5 | 48.8 | 47.7 | **18.8** |
| Accuracy (A) | 37.7 | 41.1 | 37.7 | 40.1 | **52.2** | 44.4 |
| **Qwen2.5-7B-Inst** | | | | | | |
| Truthfulness (T) | -7.9 | -18.2 | 11.0 | 2.1 | -2.3 | **23.1** |
| Hallucination (H) | 46.5 | 59.1 | 30.2 | 35.6 | 50.1 | **14.6** |
| Accuracy (A) | 38.6 | 40.9 | 41.1 | 37.7 | **47.9** | 37.6 |

### Ablation: Reward Design (Table 3, Llama3.1-8B, With Retrieval, Average)

| Reward Design | Avg T | Avg H |
|---------------|-------|-------|
| Binary | 4.5 | 47.7 |
| Binary + knowledge-enhanced | 11.7 | 38.3 |
| **Ternary** | **25.6** | **18.8** |
| Ternary + knowledge-enhanced | 23.2 | 18.9 |

### Online vs Offline RL (Table 4, Llama3.1-8B, Average)

| Method | Avg T | Avg H |
|--------|-------|-------|
| DPO (offline) | -10.1 | 51.1 |
| Iterative DPO (best: Iter 3) | 12.6 | 31.7 |
| **TruthRL (online GRPO)** | **25.6** | **18.8** |

### Verifier Type (Table 5, CRAG)

| Verifier | T | H |
|----------|---|---|
| Rule-based (string matching) | -3.6 | 3.6 |
| **LLM-based** | **37.2** | **19.4** |

### Scalability (Table 7, CRAG with Retrieval)

| Backbone | Prompting T | TruthRL T | Prompting H | TruthRL H |
|----------|-------------|-----------|-------------|-----------|
| Llama3.2-3B | 1.9 | 27.4 | 45.1 | 21.5 |
| Qwen2.5-3B | -0.3 | 21.9 | 45.4 | 16.2 |
| Qwen2.5-7B | 10.6 | 33.1 | 38.4 | 17.3 |
| Llama3.1-8B | 5.3 | 37.2 | 43.5 | 19.4 |
| Qwen2.5-32B | 29.1 | 40.0 | 27.1 | 18.2 |

### Robustness to LLM Judges (Table 6, CRAG)

| Judge | Prompting T / H | TruthRL T / H |
|-------|-----------------|---------------|
| Llama3.3-70B | 5.3 / 43.5 | 37.2 / 19.4 |
| Qwen2.5-72B | 1.9 / 45.3 | 35.6 / 20.2 |
| Gemma3-27B | 6.5 / 42.9 | 39.7 / 18.2 |

### Hallucination-Baiting (Table 2, Comparison-type CRAG questions)

| Method | T | H |
|--------|---|---|
| Prompting | 9.7 | 39.8 |
| SFT | 3.0 | 48.5 |
| RFT | 12.7 | 38.8 |
| R-Tuning | 6.8 | 43.7 |
| **TruthRL** | **52.4** | **16.5** |

### Reasoning Reward (Table 8, CRAG)

| Method | Outcome T | Outcome H | Reasoning Score |
|--------|-----------|-----------|-----------------|
| Prompting | 5.3 | 43.5 | 50.2 |
| TruthRL (outcome only) | **37.2** | 19.4 | 56.6 |
| + multiplicative r_reason | 37.0 | 19.4 | 54.7 |
| + additive r_reason | 36.1 | **19.1** | **59.1** |
| + conditional r_reason | 35.6 | 19.3 | 55.1 |

### Knowledge Boundary Recognition (Figure 3, CRAG with Retrieval, Llama3.1-8B)

On **difficult questions** (where almost no method provides correct answers):
- SFT: 100% hallucination
- TruthRL_Binary: 99.1% hallucination
- TruthRL: 15.5% hallucination, 84.5% abstention

## Limitations

- **Single training dataset:** All models are trained on CRAG only and evaluated on four benchmarks. Generalization to fundamentally different domains (code, math, creative writing) is untested.
- **Dependence on LLM judge:** The reward signal relies on an LLM verifier (Llama3.3-70B), which introduces its own biases and failure modes. The paper shows robustness across three judges but all are large instruction-tuned models from similar training paradigms.
- **Truthfulness metric ignores uncertainty rate:** With w2=0 in the default metric (T = Acc - Hall), the model gets no explicit credit for appropriate abstention in the evaluation metric, even though the ternary reward does reward it during training. This creates a disconnect between training signal and evaluation.
- **Knowledge boundary probing is expensive:** Sampling 256 responses per question to identify OOK questions is computationally costly and model-specific, limiting practical applicability.
- **No analysis of reasoning quality improvement mechanism:** Section 4.6 shows outcome-only reward implicitly improves reasoning score (50.2 -> 56.6), but the mechanism is unexplored. Explicit reasoning rewards showed mixed results and were deferred to future work.
- **Rule-based verifier collapse not deeply analyzed:** Table 5 is a critical finding (T=-3.6 with string matching) but the paper does not explore middle-ground solutions (e.g., fuzzy matching, embedding similarity) between pure string matching and full LLM judge.
- **Abstention detection method not detailed:** The paper does not fully specify the exact mechanism for classifying a response as "uncertain" vs. "incorrect" during reward assignment -- this classification is critical for the ternary reward to work correctly.
- **No comparison with recent calibration-focused methods:** Missing comparisons with RLCR (Brier score rewards), Rewarding Doubt (log scoring rules), or CDA (contrastive decoding with abstention).
- **Binary abstention only:** Abstention is all-or-nothing ("I don't know") -- no graded confidence or calibrated probability expression.

## Connections to Current Research

- **Direct foundation of the TruthRL-personalized- project:** This is the paper the repository is based on. The ternary reward scheme (+1/0/-1) is implemented in `training/verl/verl/utils/reward_score/truthrl_qa.py`. The OOK variant (_OOK functions) maps to the knowledge-enhanced reward variant.
- **Symbolic process reward (project Direction 3):** The paper's Section 4.6 and Table 8 show that naive reasoning reward integration yields mixed results -- outcome-only reward implicitly improves reasoning, while explicit reasoning rewards require "non-trivial design." This validates the project's approach of targeting decidable structural properties (echo detection, phase transitions, repetition) rather than LLM-judged reasoning quality.
- **Table 5 finding as project constraint:** Rule-based verifier collapse (T=-3.6) is already logged in the project's cerebrum.md Do-Not-Repeat section. Any symbolic verifier must handle semantic equivalence, not just string matching. The project's hybrid approach (symbolic process reward + LLM outcome reward) is designed to avoid this failure mode.
- **GRPO advantage mechanism:** The paper's explanation of why ternary beats binary (group-normalized advantage computation giving abstention higher advantage than hallucination when rewards are 0 vs. -1) is key for understanding how to compose additional reward signals without disrupting the ternary signal's effectiveness.
- **Connection to formal verification:** The ternary reward is a simple formal structure (three discrete values), but the verifier that produces it is an LLM, not a formal system. The project's symbolic process reward adds formally verifiable signals (decidable properties of reasoning text) alongside the LLM-based outcome reward.
- **AbstentionBench (Meta2025) finding:** Reasoning fine-tuning degrades abstention by 24%, consistent with TruthRL's finding that vanilla SFT suppresses uncertainty to near-zero. TruthRL's ternary reward is one solution; the project explores whether symbolic process rewards can provide additional training signal without this degradation.
- **RLCR and Rewarding Doubt:** These ICLR 2026 papers take calibration-theoretic approaches (Brier score, log scoring rules) to the same core problem. TruthRL's ternary reward is simpler but less theoretically grounded in proper scoring rules.

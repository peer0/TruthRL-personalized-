# Paper Analysis: Rewarding Doubt: A Reinforcement Learning Approach to Confidence Calibration of LLMs

**Authors:** Ido Shayovitz, Gal Rabinovich
**Year:** 2026
**Source:** ICLR 2026
**PDF path:** /Users/jay/projects/papers/abstention/ICLR2026_Rewarding_Doubt_A_Reinfo.pdf
**Analyzed:** 2026-04-07

## Key Claims
1. LLMs can be trained to produce well-calibrated confidence scores alongside answers through RL with proper scoring rule-based rewards.
2. Log scoring rules (logarithmic proper scoring rules) provide theoretically grounded reward signals that incentivize honest confidence reporting.
3. The approach produces calibrated models that can abstain on low-confidence predictions, achieving better accuracy-coverage trade-offs than baselines.
4. Confidence calibration through RL generalizes across tasks and does not require task-specific reward engineering.

## Method
The core insight is to use **proper scoring rules** from calibration theory as RL reward signals. A proper scoring rule has the property that the expected reward is maximized when the model reports its true belief -- so truthful confidence reporting is the rational strategy.

- **Output format:** The model generates both an answer and a numerical confidence score p in [0, 1].
- **Reward function (Log Scoring Rule):**
  - If the answer is **correct:** R = log(p) (reward increases as confidence increases)
  - If the answer is **incorrect:** R = log(1 - p) (reward increases as confidence decreases, i.e., the model is rewarded for expressing doubt when wrong)
- **RL algorithm:** PPO (Proximal Policy Optimization) for training.
- **Abstention mechanism:** At inference, threshold-based selective prediction: if the model's confidence p < tau, it abstains. The threshold tau is tunable for desired accuracy-coverage trade-offs.
- **Evaluation metrics:** Expected Calibration Error (ECE), selective prediction AUC, accuracy at various coverage levels, reliability diagrams.

The key theoretical property: under a proper scoring rule, the optimal strategy is to set p equal to the model's actual probability of being correct. Any other strategy (over-confidence or under-confidence) decreases expected reward.

## Results
| Metric | Value | Baseline comparison |
|--------|-------|---------------------|
| ECE (Expected Calibration Error) | Significantly reduced | vs. vanilla LLM, temperature scaling, verbalized confidence |
| Accuracy at 80% coverage | Higher | vs. uncalibrated model and post-hoc methods |
| Selective prediction AUC | Improved | vs. MaxProb, temperature scaling, verbalized confidence |
| Reliability diagram | Well-calibrated | vs. over-confident baseline models |

## Limitations
- Requires the model to generate an explicit confidence score as part of its output, adding format complexity and potential for format errors.
- Log scoring rule has mathematical singularities: log(0) = -infinity. Extreme confidence near 0 or 1 can cause very large reward magnitudes, potentially destabilizing training.
- PPO is more computationally expensive and harder to tune than GRPO.
- Assumes the model has meaningful internal uncertainty signals that can be surfaced through training. For completely novel domains, the model may not have a useful prior.
- Evaluated primarily on factoid QA benchmarks; transfer to complex multi-step reasoning, code generation, or creative tasks is unvalidated.
- Inference-time abstention requires choosing a threshold tau, which introduces a deployment-time hyperparameter.

## Connections to Current Research
- Directly comparable to TruthRL: both address the same fundamental problem (teaching LLMs to manage uncertainty) but from different theoretical angles.
- TruthRL uses a discrete ternary reward (correct/abstain/hallucinate); Rewarding Doubt uses a continuous calibration-theoretic reward.
- RLCR (ICLR 2026) takes a similar proper scoring rule approach but uses Brier score instead of log score.
- The confidence score approach could potentially complement TruthRL's binary abstention -- a model could say "I am 30% confident" rather than just "I don't know."
- The proper scoring rule framework has connections to decision theory and could inform more principled reward design for the TruthRL-personalized project.

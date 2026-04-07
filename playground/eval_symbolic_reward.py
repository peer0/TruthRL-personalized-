"""
Offline evaluation of the Symbolic Process Reward on TruthRL-CRAG.

Generates model outputs with <think>...</think>\\boxed{...} format,
scores them with the symbolic process reward, and analyzes correlation
between structural quality and answer correctness.

Usage:
    # Quick test (5 examples)
    python playground/eval_symbolic_reward.py --n_examples 5

    # Full evaluation (all 643 test examples)
    python playground/eval_symbolic_reward.py --n_examples 0

    # Custom model
    python playground/eval_symbolic_reward.py --model mlx-community/Qwen2.5-3B-Instruct-4bit

    # Resume from saved generations (skip generation, just re-score)
    python playground/eval_symbolic_reward.py --load_from playground/results/generations.json

Requirements:
    pip install mlx-lm datasets
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add verl to path for symbolic reward imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "verl"))

from datasets import load_dataset
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


# ---------------------------------------------------------------------------
# Symbolic Process Reward (inline import to avoid verl __init__.py deps)
# ---------------------------------------------------------------------------
def _import_symbolic_reward():
    """Import SymbolicProcessReward without triggering verl's heavy __init__.py."""
    import importlib.util

    base = Path(__file__).resolve().parent.parent / "training" / "verl" / "verl" / "utils" / "reward_score" / "symbolic_process_reward"

    def _load_module(name, filepath):
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load_module("verl.utils.reward_score.symbolic_process_reward.format_gate", base / "format_gate.py")
    _load_module("verl.utils.reward_score.symbolic_process_reward.phase_scorer", base / "phase_scorer.py")
    _load_module("verl.utils.reward_score.symbolic_process_reward.ngram_scorer", base / "ngram_scorer.py")
    _load_module("verl.utils.reward_score.symbolic_process_reward.echo_detector", base / "echo_detector.py")
    composer_mod = _load_module("verl.utils.reward_score.symbolic_process_reward.composer", base / "composer.py")

    return composer_mod.SymbolicProcessReward


# ---------------------------------------------------------------------------
# Answer evaluation (simplified — no LLM judge, EM only)
# ---------------------------------------------------------------------------
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_boxed(text):
    match = re.search(r"\\boxed{(.*?)}", text, re.DOTALL)
    return match.group(1).strip() if match else None


def classify_output(prediction, ground_truths):
    """Classify output as correct / abstain / hallucinate."""
    if prediction is None:
        return "no_boxed"

    normalized = normalize_answer(prediction)

    if "i dont know" in normalized or "i don't know" in normalized.replace("'", ""):
        return "abstain"

    if "invalid question" in normalized:
        return "invalid_question"

    for gt in ground_truths:
        if normalize_answer(gt) == normalized:
            return "correct"

    # No exact match — could be correct via LLM judge, but we classify as wrong for EM
    return "wrong"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate symbolic process reward on TruthRL-CRAG")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                        help="MLX model to use for generation")
    parser.add_argument("--n_examples", type=int, default=5,
                        help="Number of examples to evaluate (0 = all)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens per generation")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--load_from", default=None,
                        help="Load pre-generated outputs from JSON instead of generating")
    parser.add_argument("--save_dir", default="playground/results",
                        help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load symbolic reward
    SymbolicProcessReward = _import_symbolic_reward()
    reward = SymbolicProcessReward()
    print("[OK] Symbolic process reward loaded")

    if args.load_from:
        # Load pre-generated outputs
        with open(args.load_from) as f:
            results = json.load(f)
        print(f"[OK] Loaded {len(results)} pre-generated outputs from {args.load_from}")
    else:
        # Generate outputs
        print(f"[..] Loading model: {args.model}")
        model, tokenizer = load(args.model)
        sampler = make_sampler(temp=0.0)
        print("[OK] Model loaded")

        ds = load_dataset("weizhepei/TruthRL-CRAG", split=args.split)
        n = len(ds) if args.n_examples == 0 else min(args.n_examples, len(ds))
        print(f"[..] Generating {n} outputs...")

        results = []
        for i in range(n):
            ex = ds[i]
            prompt_content = ex["prompt"][0]["content"]
            messages = [{"role": "user", "content": prompt_content}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            t0 = time.time()
            response = generate(model, tokenizer, prompt=prompt_text,
                                max_tokens=args.max_tokens, sampler=sampler)
            elapsed = time.time() - t0

            gt = ex["reward_model"]["ground_truth"]
            targets = gt["target"] if gt["target"] else [ex["answer"]]

            results.append({
                "index": i,
                "query": ex["query"],
                "domain": ex["domain"],
                "question_type": ex["question_type"],
                "ground_truth": targets,
                "out_of_knowledge": gt.get("out_of_knowledge"),
                "prompt": prompt_content,
                "response": response,
                "generation_time": round(elapsed, 2),
            })

            prediction = extract_boxed(response)
            label = classify_output(prediction, targets)
            print(f"  [{i+1}/{n}] {label:15s} | {elapsed:.1f}s | {ex['query'][:60]}")

        # Save generations
        gen_path = os.path.join(args.save_dir, "generations.json")
        with open(gen_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[OK] Generations saved to {gen_path}")

    # -----------------------------------------------------------------------
    # Score all outputs with symbolic reward
    # -----------------------------------------------------------------------
    print(f"\n[..] Scoring {len(results)} outputs with symbolic process reward...")

    scored = []
    for r in results:
        response = r["response"]
        prompt = r["prompt"]
        prediction = extract_boxed(response)
        label = classify_output(prediction, r["ground_truth"])

        process_result = reward.score(response, prompt=prompt)

        scored.append({
            **r,
            "prediction": prediction,
            "label": label,
            "process_score": process_result["process_score"],
            "gate_pass": process_result["gate_pass"],
            "phase_score": process_result["phase_score"],
            "ngram_penalty": process_result["ngram_penalty"],
            "echo_penalty": process_result["echo_penalty"],
        })

    # Save scored results
    scored_path = os.path.join(args.save_dir, "scored.json")
    with open(scored_path, "w") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)
    print(f"[OK] Scored results saved to {scored_path}")

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYSIS: Symbolic Process Reward vs Outcome")
    print(f"{'='*60}")

    # Group by label
    groups = defaultdict(list)
    for s in scored:
        groups[s["label"]].append(s)

    print(f"\n--- Distribution ---")
    for label in ["correct", "wrong", "abstain", "invalid_question", "no_boxed"]:
        count = len(groups[label])
        pct = count / len(scored) * 100 if scored else 0
        print(f"  {label:20s}: {count:4d} ({pct:5.1f}%)")

    print(f"\n--- Process Score by Outcome ---")
    print(f"  {'Label':20s} {'Count':>6s} {'Mean':>8s} {'Std':>8s} {'Gate%':>8s}")
    print(f"  {'-'*52}")
    for label in ["correct", "wrong", "abstain", "invalid_question", "no_boxed"]:
        items = groups[label]
        if not items:
            continue
        scores = [x["process_score"] for x in items]
        gate_pass = sum(1 for x in items if x["gate_pass"]) / len(items) * 100
        mean = sum(scores) / len(scores)
        std = (sum((s - mean)**2 for s in scores) / len(scores)) ** 0.5
        print(f"  {label:20s} {len(items):6d} {mean:8.4f} {std:8.4f} {gate_pass:7.1f}%")

    print(f"\n--- Component Breakdown (gate-passed only) ---")
    passed = [s for s in scored if s["gate_pass"]]
    if passed:
        for label in ["correct", "wrong", "abstain"]:
            items = [s for s in passed if s["label"] == label]
            if not items:
                continue
            phase_avg = sum(x["phase_score"] for x in items) / len(items)
            ngram_avg = sum(x["ngram_penalty"] for x in items) / len(items)
            echo_avg = sum(x["echo_penalty"] for x in items) / len(items)
            print(f"  {label:15s} (n={len(items):3d}): phase={phase_avg:.4f}  ngram={ngram_avg:.4f}  echo={echo_avg:.4f}")

    print(f"\n--- Key Question ---")
    correct_scores = [x["process_score"] for x in groups["correct"]]
    wrong_scores = [x["process_score"] for x in groups["wrong"]]
    if correct_scores and wrong_scores:
        c_mean = sum(correct_scores) / len(correct_scores)
        w_mean = sum(wrong_scores) / len(wrong_scores)
        print(f"  Correct mean process_score:  {c_mean:.4f}")
        print(f"  Wrong mean process_score:    {w_mean:.4f}")
        print(f"  Difference (correct - wrong): {c_mean - w_mean:.4f}")
        if c_mean > w_mean:
            print(f"  >> Symbolic reward discriminates correctly!")
        else:
            print(f"  >> Symbolic reward does NOT discriminate — needs tuning")
    else:
        print(f"  Not enough data to compare (correct={len(correct_scores)}, wrong={len(wrong_scores)})")

    print(f"\n[DONE] Full results at {scored_path}")


if __name__ == "__main__":
    main()

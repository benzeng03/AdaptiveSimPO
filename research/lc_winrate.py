"""Length-controlled (LC) win rate for a research/llm_judge.py comparison, after AlpacaEval 2.

AlpacaEval 2 (Dubois et al., 2024) fits a logistic model of the per-prompt preference
    logit P(variant wins) = a + b * tanh(d / sd(d)) [+ instruction-difficulty term],
where d is the variant's minus the baseline's answer length, and reports sigma(a), the win rate
with the length term set to zero. The instruction-difficulty term needs judgements of many models
on the same prompts, so it is left out here. The target is the per-prompt score from llm_judge.py
(1 win, 0.5 tie, 0 loss), fitted by maximum likelihood with fractional targets (Newton steps).
Lengths are answer lengths in characters, as in AlpacaEval. The 95% interval is a prompt bootstrap.

    python research/lc_winrate.py GEN/llm-judge-gpt-6.2-sol-medium-calibrated-batch-vs-gpt4-reference
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_lengths(gen: Path, name: str):
    out = {}
    for f in sorted(gen.glob(f"[0-9][0-9][0-9]-{name}.json")):
        for g in json.loads(f.read_text(encoding="utf-8")):
            out[g["id"]] = len(g["output"])
    return out


def fit(y, x, iters=100):
    """MLE of logit p = a + b x with fractional y. Returns (a, b); a = -inf / +inf if all y are 0 / 1."""
    if y.max() == 0:
        return -np.inf, 0.0
    if y.min() == 1:
        return np.inf, 0.0
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        w = np.clip(p * (1 - p), 1e-10, None)
        step = np.linalg.solve(X.T @ (w[:, None] * X) + 1e-9 * np.eye(2), X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta[0], beta[1]


def lc(y, d):
    sd = d.std()
    x = np.tanh(d / sd) if sd > 0 else np.zeros_like(d)
    a, b = fit(y, x)
    return 1 / (1 + np.exp(-a)), b


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("judge_dir", type=Path, help="GEN/llm-judge-<model>-<effort>-<variant>-vs-<baseline>")
    p.add_argument("--resamples", type=int, default=2000)
    a = p.parse_args(argv)
    summary = json.loads(next(a.judge_dir.glob("summary_*.json")).read_text(encoding="utf-8"))
    gen = a.judge_dir.parent
    lv, lb = load_lengths(gen, summary["variant"]), load_lengths(gen, summary["baseline"])
    rows = summary["per_prompt"]
    y = np.array([r["score"] for r in rows], dtype=float)
    d = np.array([lv[r["id"]] - lb[r["id"]] for r in rows], dtype=float)
    est, b = lc(y, d)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(a.resamples):
        i = rng.integers(0, len(y), len(y))
        boot.append(lc(y[i], d[i])[0])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    out = dict(variant=summary["variant"], baseline=summary["baseline"], n=len(y), win_rate=float(y.mean()),
               lc_win_rate=float(est), lc_ci95=[float(lo), float(hi)], length_coef=float(b),
               mean_chars_variant=float(np.mean([lv[r["id"]] for r in rows])),
               mean_chars_baseline=float(np.mean([lb[r["id"]] for r in rows])))
    (a.judge_dir / f"lc_{summary['variant']}_vs_{summary['baseline']}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{out['variant']} vs {out['baseline']}: WR {100 * out['win_rate']:.2f}%, LC WR {100 * est:.2f}% "
          f"(95% CI {100 * lo:.2f}-{100 * hi:.2f}), length coef {b:+.3f}, mean chars "
          f"{out['mean_chars_variant']:.0f} / {out['mean_chars_baseline']:.0f}")


if __name__ == "__main__":
    main()

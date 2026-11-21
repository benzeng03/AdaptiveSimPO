"""Collect training margin statistics, dev/test scores and win rates of a run into RUN/results.md.

Test comparisons are paired against the dev-selected fixed baseline on the same
pairs: the accuracy and log-loss differences come with 95% bootstrap intervals over
pairs (2000 resamples, seed 0), and the accuracy difference with an exact McNemar test.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}%"


def num(x, digits=4):
    return "—" if x is None else f"{x:.{digits}f}"


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def margin_stats(adapter: Path):
    """margin_stats.json of an adapter; the mean calibration slope is filled in from
    metrics.jsonl for adapters trained before it was stored."""
    s = load(adapter / "margin_stats.json")
    metrics = adapter / "metrics.jsonl"
    if "calib_k_mean" not in s and metrics.exists():
        ks = [json.loads(line).get("calib_k") for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
        ks = [k for k in ks if k is not None]
        if ks:
            s["calib_k_mean"] = sum(ks) / len(ks)
    return s


def mcnemar(a_correct, b_correct):
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    n01 = int((a_correct & ~b_correct).sum())
    n10 = int((~a_correct & b_correct).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail), n01, n10


def paired(gaps, base_gaps, resamples=2000):
    a, b = torch.tensor(gaps, dtype=torch.float64), torch.tensor(base_gaps, dtype=torch.float64)
    acc_d = (a > 0).double() - (b > 0).double()
    ll_d = F.softplus(-a) - F.softplus(-b)
    idx = torch.randint(0, len(a), (resamples, len(a)), generator=torch.Generator().manual_seed(0))
    ci = lambda d: tuple(torch.quantile(d[idx].mean(1), torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist())
    p, n01, n10 = mcnemar(a > 0, b > 0)
    return dict(acc_diff=acc_d.mean().item(), acc_ci=ci(acc_d), logloss_diff=ll_d.mean().item(),
                logloss_ci=ci(ll_d), mcnemar_p=p, only_variant=n01, only_baseline=n10)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args(argv)
    run = args.run
    dev = {r["adapter"]: r for r in load(run / "dev_scores.json") or []}
    test = {r["adapter"]: r for r in load(run / "test_scores.json") or []}
    stats = {d.name: margin_stats(d) for d in sorted(run.iterdir()) if (d / "margin_stats.json").exists()}
    # GPT-judge summaries written by research/llm_judge.py (one per comparison).
    judged = [load(f) for f in sorted(run.glob("generations-*/llm-judge-*/summary_*.json"))
              if "-limit" not in f.parent.name]
    selection = load(run / "selection.json") or {}
    baseline = selection.get("best_fixed")
    names = list(dev) if dev else list(stats)  # dev_scores lists exactly the scored variants

    lines = [f"# Results: {run.name}", "",
             f"Dev-selected fixed baseline: `{baseline}` ({selection.get('rule', '')})", "",
             "## Training and dev", "",
             "| Variant | Train mean γ | Train γ > 0 | Mean k̂ | Mean grad weight | Grad share on misranked | "
             "Dev acc | Dev acc (untied) | Dev logloss |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        s, d = stats.get(n) or {}, dev.get(n, {})
        lines.append(f"| {n} | {num(s.get('gamma_mean'))} | {pct(s.get('active'))} | {num(s.get('calib_k_mean'), 3)} | "
                     f"{num(s.get('weight_mean'))} | {pct(s.get('weight_share_misranked'))} | {pct(d.get('accuracy'))} | "
                     f"{pct(d.get('accuracy_untied'))} | {num(d.get('logloss'))} |")
    lines += ["", "k̂ is the calibration slope of the calibrated margin. Grad weights are sigmoid(γ_i − r_i); "
              "both are averaged over training steps."]

    if test:
        lines += ["", "## Test pairs (reporting only)", "",
                  "| Variant | Test acc | Test acc (untied) | Test logloss | Δacc vs baseline (95% CI) | "
                  "McNemar p [only variant / only baseline] | Δlogloss vs baseline (95% CI) |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        base = test.get(baseline)
        for n in names:
            t = test.get(n)
            if t is None:
                continue
            cmp_acc = cmp_p = cmp_ll = "—"
            if base is not None and n == baseline:
                cmp_acc = cmp_p = cmp_ll = "baseline"
            elif base is not None:
                c = paired(t["gaps"], base["gaps"])
                cmp_acc = f"{100 * c['acc_diff']:+.2f} ({100 * c['acc_ci'][0]:+.2f}, {100 * c['acc_ci'][1]:+.2f})"
                cmp_p = f"{c['mcnemar_p']:.3f} [{c['only_variant']}/{c['only_baseline']}]"
                cmp_ll = f"{c['logloss_diff']:+.4f} ({c['logloss_ci'][0]:+.4f}, {c['logloss_ci'][1]:+.4f})"
            lines.append(f"| {n} | {pct(t['accuracy'])} | {pct(t.get('accuracy_untied'))} | "
                         f"{num(t['logloss'])} | {cmp_acc} | {cmp_p} | {cmp_ll} |")
        lines += ["", f"Test pairs: {next(iter(test.values()))['pairs']}; differences are in "
                  "percentage points (accuracy) and nats (log-loss), variant minus baseline."]

    if judged:
        lines += ["", "## Generation (GPT judge)", "",
                  "| Comparison | Judge | Win rate (95% CI) [W/T/L] | Order-consistent | Mean tokens (variant / baseline) |",
                  "|---|---|---:|---:|---:|"]
        for s in judged:
            rows = s["per_prompt"]
            vt = sum(r["variant_tokens"] for r in rows) / len(rows)
            bt = sum(r["baseline_tokens"] for r in rows) / len(rows)
            lo, hi = s["ci95"]
            lines.append(f"| {s['variant']} vs {s['baseline']} | {s['judge']} | {pct(s['win_rate'])} "
                         f"({pct(lo)}–{pct(hi)}) [{s['wins']}/{s['ties']}/{s['losses']}] | "
                         f"{pct(s['order_consistency'])} | {vt:.0f} / {bt:.0f} |")
        lines += ["", "Each prompt is judged in both answer orders; a win needs both orders to agree, "
                  "anything else is a tie."]
    (run / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

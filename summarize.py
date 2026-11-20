"""Collect training margin statistics, dev scores and win rates of a run into RUN/results.md."""
import argparse
import json
import sys
from pathlib import Path


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}%"


def num(x, digits=4):
    return "—" if x is None else f"{x:.{digits}f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args(argv)
    run = args.run
    dev = {r["adapter"]: r for r in json.loads((run / "dev_scores.json").read_text(encoding="utf-8"))} \
        if (run / "dev_scores.json").exists() else {}
    stats = {d.name: json.loads((d / "margin_stats.json").read_text(encoding="utf-8"))
             for d in sorted(run.iterdir()) if (d / "margin_stats.json").exists()}
    wins = None
    for gen in sorted(run.glob("generations-*")):
        if (gen / "win_rates.json").exists():
            wins = json.loads((gen / "win_rates.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8")) \
        if (run / "selection.json").exists() else {}

    lines = [f"# Results: {run.name}", "",
             f"Dev-selected fixed baseline: `{selection.get('best_fixed')}` ({selection.get('rule', '')})", "",
             "| Variant | Train mean γ | Normal-ref mean γ | Train active | Normal-ref active | "
             "Dev acc | Dev logloss | Dev mean r | H2H vs baseline (95% CI) | vs GPT-4 ref | Mean tokens |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names = list(dev) if dev else list(stats)  # dev_scores lists exactly the scored variants
    for n in names:
        s, d = stats.get(n, {}), dev.get(n, {})
        h2h = ref = tokens = "—"
        if wins:
            if n in wins["head_to_head"]:
                w = wins["head_to_head"][n]
                lo, hi = w["bootstrap_ci95"]
                h2h = f"{pct(w['win_rate'])} ({pct(lo)}–{pct(hi)}) [{w['wins']}/{w['ties']}/{w['losses']}]"
            elif n == wins["baseline"]:
                h2h = "baseline"
            if n in wins["vs_reference"]:
                ref = pct(wins["vs_reference"][n]["win_rate"])
                tokens = f"{wins['mean_tokens'][n]:.0f}"
        lines.append(f"| {n} | {num(s.get('gamma_mean'))} | {num(s.get('normal_ref_gamma_mean'))} | "
                     f"{pct(s.get('active'))} | {pct(s.get('normal_ref_active'))} | {pct(d.get('accuracy'))} | "
                     f"{num(d.get('logloss'))} | {num(d.get('reward_gap'))} | {h2h} | {ref} | {tokens} |")
    lines += ["", "H2H counts are wins/ties/losses from the variant's perspective; "
              "ties include order-inconsistent PairRM judgments."]
    (run / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

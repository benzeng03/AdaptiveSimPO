"""Accuracy on the merged dev+test set (data/uf_devtest1986.jsonl: the 500 dev pairs, then the
1486 test pairs; the two splits share no prompt).

The per-pair reward gaps already written by score_pairs.py (RUN/dev_scores.json and
RUN/test_scores.json) are concatenated in that order, so the result is the same as scoring the
merged file and stays consistent with the reported dev and test numbers. Each adapter is taken
from the first run directory that has both splits for it. Writes OUT (default
outputs/devtest1986_scores.json) with accuracy per adapter and paired comparisons with the
baseline (bootstrap CI + exact McNemar, as in summarize.py).

    python research/devtest_scores.py outputs/run-A outputs/run-B --baseline fixed-g2

With --dev-data none there is no dev split: the test scores alone are used (the on-policy test
set, every test_prefs prompt).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from adaptive_simpo.modeling import read_jsonl  # noqa: E402
from summarize import paired  # noqa: E402


def load(run: Path, split: str):
    path = run / f"{split}_scores.json"
    return {r["adapter"]: r for r in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--baseline", default="fixed-g2")
    p.add_argument("--adapters", nargs="+", help="only these adapters (default: all with dev and test scores)")
    p.add_argument("--dev-data", type=Path, default=HERE / "data/uf_dev500.jsonl")
    p.add_argument("--test-data", type=Path, default=HERE / "data/uf_test1486.jsonl")
    p.add_argument("--out", type=Path, default=HERE / "outputs/devtest1986_scores.json")
    a = p.parse_args(argv)

    no_dev = str(a.dev_data) == "none"
    dev_rows, test_rows = ([] if no_dev else read_jsonl(a.dev_data)), read_jsonl(a.test_data)
    rows = dev_rows + test_rows
    strict = torch.tensor([r.get("score_chosen", 1.0) != r.get("score_rejected", 0.0) for r in rows])
    gaps, source = {}, {}
    for run in a.runs:
        dev, test = ({} if no_dev else load(run, "dev")), load(run, "test")
        for name in (test if no_dev else dev):
            if name in test and name not in gaps and (not a.adapters or name in a.adapters):
                d, t = ([] if no_dev else dev[name]["gaps"]), test[name]["gaps"]
                if len(d) != len(dev_rows) or len(t) != len(test_rows):
                    raise SystemExit(f"{run}/{name}: scored on different dev/test files")
                gaps[name], source[name] = d + t, str(run)
    if a.adapters and set(a.adapters) - set(gaps):
        raise SystemExit(f"no dev+test scores for {sorted(set(a.adapters) - set(gaps))}")
    if a.baseline not in gaps:
        raise SystemExit(f"baseline {a.baseline} has no dev+test scores in {a.runs}")

    base = gaps[a.baseline]
    results = []
    for name, g in gaps.items():
        x = torch.tensor(g)
        res = dict(adapter=name, run=source[name], pairs=len(g), correct=int((x > 0).sum()),
                   accuracy=(x > 0).float().mean().item(),
                   accuracy_untied=(x[strict] > 0).float().mean().item(), untied_pairs=int(strict.sum()))
        if name != a.baseline:
            res["vs_baseline"] = paired(g, base)
        results.append(res)
    def gamma(name):  # fixed-g2 or fixed-g2-lr1e-05
        m = re.match(r"fixed-g([0-9.]+)", name)
        return float(m.group(1)) if m else 0.0

    order = lambda r: (r["adapter"] != "base", not r["adapter"].startswith("fixed-g"),
                       gamma(r["adapter"]), r["adapter"])
    results.sort(key=order)
    a.out.write_text(json.dumps(dict(baseline=a.baseline, data=f"test set {a.test_data.name}" if no_dev else "dev500 + test1486", results=results), indent=2),
                     encoding="utf-8")
    for r in results:
        line = f"{r['adapter']:18s} {r['correct']}/{r['pairs']}  {100 * r['accuracy']:.2f}%  untied {100 * r['accuracy_untied']:.2f}%"
        if "vs_baseline" in r:
            v = r["vs_baseline"]
            line += (f"  vs {a.baseline} {100 * v['acc_diff']:+.2f} (CI {100 * v['acc_ci'][0]:+.2f} to "
                     f"{100 * v['acc_ci'][1]:+.2f}, McNemar p={v['mcnemar_p']:.3f}, {v['only_variant']}/{v['only_baseline']})")
        print(line)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

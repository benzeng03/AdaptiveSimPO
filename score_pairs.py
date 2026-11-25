"""Score trained adapters on held-out preference pairs and select the fixed-margin baseline.

For each adapter under RUN (and the untrained base model) this reports pairwise
accuracy (r > 0), accuracy on pairs whose annotation scores are not tied, mean
log-loss softplus(-r) and the mean reward gap r. Writes RUN/<split>_scores.json and
RUN/<split>_scores.csv. With --split dev (the default) the fixed-margin adapter with
the highest accuracy is selected, ties broken by lower log-loss and then smaller
gamma, and written to RUN/selection.json. --split test and --split train (the training pairs,
for training accuracy) only report.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from adaptive_simpo.modeling import Policies, batch_logps, is_policy_dir, read_jsonl

HERE = Path(__file__).resolve().parent


def select_fixed(results):
    fixed = [r for r in results if r["margin"] == "fixed"]
    if not fixed:
        return None
    return min(fixed, key=lambda r: (-r["accuracy"], r["logloss"], r["gamma"]))["adapter"]


@torch.inference_mode()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--split", choices=("dev", "test", "train"), default="dev")
    p.add_argument("--data", type=Path, help="default: data/uf_dev500.jsonl, data/uf_test1486.jsonl or data/uf_train4096.jsonl")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-base", action="store_true", help="skip scoring the untrained base model")
    p.add_argument("--adapters", nargs="+", help="only score these adapter directory names")
    args = p.parse_args(argv)
    if args.data is None:
        args.data = HERE / {"dev": "data/uf_dev500.jsonl", "test": "data/uf_test1486.jsonl",
                             "train": "data/uf_train4096.jsonl"}[args.split]
    rows = read_jsonl(args.data)
    if not rows:
        p.error(f"empty {args.split} dataset")
    strict = torch.tensor([r.get("score_chosen", 1.0) != r.get("score_rejected", 0.0) for r in rows])
    dirs = sorted(d for d in args.run.iterdir() if d.is_dir() and is_policy_dir(d)
                  and (not args.adapters or d.name in args.adapters))
    if args.adapters and len(dirs) != len(set(args.adapters)):
        p.error(f"missing adapters: {sorted(set(args.adapters) - {d.name for d in dirs})}")
    if not dirs:
        p.error(f"no trained adapters under {args.run}")
    cfgs = {d.name: json.loads((d / "training_args.json").read_text(encoding="utf-8")) for d in dirs}
    first = cfgs[dirs[0].name]
    policies = Policies(first["model"], {d.name: d for d in dirs}, first["device"], first["bf16"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(first["model"])
    names = ([] if args.no_base else ["base"]) + [d.name for d in dirs]
    results = []
    for name in names:
        cfg = first if name == "base" else cfgs[name]
        gaps = []
        with policies.use(name) as model:
            for i in range(0, len(rows), args.batch_size):
                c, r = batch_logps(model, rows[i:i + args.batch_size], tok, cfg["max_length"],
                                   cfg["max_prompt"], cfg["device"], cfg.get("chat_prompt", True),
                                   truncation=cfg.get("truncation", "always"))
                gaps.append((cfg["beta"] * (c.float() - r.float())).cpu())
        gap = torch.cat(gaps)
        kind = "base" if name == "base" else "dpo" if cfg.get("loss") == "dpo" else cfg["margin"]
        res = dict(adapter=name, margin=kind,
                   gamma=cfg["gamma"] if kind == "fixed" else None,
                   pairs=len(rows), correct=int((gap > 0).sum()),
                   accuracy=(gap > 0).float().mean().item(),
                   accuracy_untied=(gap[strict] > 0).float().mean().item() if strict.any() else None,
                   untied_pairs=int(strict.sum()),
                   logloss=F.softplus(-gap).mean().item(), reward_gap=gap.mean().item(),
                   gaps=gap.tolist())
        results.append(res)
        print(f"[{args.split}] {name:24s} acc {res['correct']}/{len(rows)}  logloss {res['logloss']:.4f}  "
              f"mean r {res['reward_gap']:+.4f}", flush=True)
    (args.run / f"{args.split}_scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (args.run / f"{args.split}_scores.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [k for k in results[0] if k != "gaps"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    if args.split != "dev":
        return
    best = select_fixed(results)
    (args.run / "selection.json").write_text(json.dumps(dict(
        best_fixed=best, rule="max dev accuracy, then min dev logloss, then min gamma",
        data=str(args.data)), indent=2), encoding="utf-8")
    print(f"[dev] selected fixed baseline: {best}")


if __name__ == "__main__":
    main()

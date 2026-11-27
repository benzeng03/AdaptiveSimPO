"""Score trained adapters on held-out preference pairs and select the fixed-margin baseline.

For each adapter under RUN (and the untrained base model) this reports pairwise
accuracy (r > 0), accuracy on pairs whose annotation scores are not tied, mean
log-loss softplus(-r) and the mean reward gap r. r is the reward the adapter was trained on
(training_args.json "reward": simpo, dpo, dpo-noref, r-dpo or orpo; DPO and R-DPO against the base
model as reference); the base model is scored with the SimPO reward. Adapters with another reward
also report accuracy_mean_logp, the accuracy of the length-normalised gap, which is comparable
across rewards. Writes RUN/<split>_scores.json and RUN/<split>_scores.csv. With --split dev (the
default) the fixed-margin SimPO adapter with the highest accuracy is selected, ties broken by
lower log-loss and then smaller gamma, and written to RUN/selection.json. --split test and --split train (the training pairs,
for training accuracy) only report. A model trained with train.py --eval-data on the same file
(same content and lengths) is not loaded again: its cached log-probabilities are used.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from adaptive_simpo import evalcache
from adaptive_simpo.modeling import Policies, batch_logp_sums, is_policy_dir, read_jsonl
from adaptive_simpo.rewards import NEEDS_REFERENCE, reward_gap

HERE = Path(__file__).resolve().parent


def reward_of(cfg):
    """Reward an adapter was trained on; adapters from before --reward used --loss dpo or SimPO."""
    return "dpo" if cfg.get("loss") == "dpo" else cfg.get("reward", "simpo")


def select_fixed(results):
    fixed = [r for r in results if r["margin"] == "fixed" and r.get("reward", "simpo") == "simpo"]
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
    tables = {}  # (policy, encoding) -> summed log-probabilities and lengths of every pair

    def table(name, cfg):
        enc = (cfg["max_length"], cfg["max_prompt"], cfg.get("chat_prompt", True), cfg.get("truncation", "always"))
        if (name, enc) not in tables and name != "base":
            cached = evalcache.load(args.run / name, args.data, enc)  # written by train.py --eval-data
            if cached is not None:
                print(f"[{args.split}] {name}: log-probabilities cached at the end of training", flush=True)
                tables[(name, enc)] = cached
        if (name, enc) not in tables:
            parts = []
            with policies.use(name) as model:
                for i in range(0, len(rows), args.batch_size):
                    parts.append([t.float().cpu() for t in batch_logp_sums(
                        model, rows[i:i + args.batch_size], tok, enc[0], enc[1], cfg["device"], enc[2],
                        truncation=enc[3])])
            tables[(name, enc)] = [torch.cat(col) for col in zip(*parts)]
        return tables[(name, enc)]

    results = []
    for name in names:
        cfg = first if name == "base" else cfgs[name]
        reward = "simpo" if name == "base" else reward_of(cfg)
        cs, rs, cl, rl = table(name, cfg)
        ref = {}
        if reward in NEEDS_REFERENCE:
            ref_cs, ref_rs, _, _ = table("base", cfg)
            ref = dict(ref_chosen_sum=ref_cs, ref_rejected_sum=ref_rs)
        gap = reward_gap(reward, cfg["beta"], cs, rs, cl, rl, alpha=cfg.get("rdpo_alpha", 0.0), **ref)
        kind = "base" if name == "base" else "dpo" if cfg.get("loss") == "dpo" else cfg["margin"]
        res = dict(adapter=name, margin=kind, reward=reward,
                   gamma=cfg["gamma"] if kind == "fixed" else None,
                   pairs=len(rows), correct=int((gap > 0).sum()),
                   accuracy=(gap > 0).float().mean().item(),
                   accuracy_untied=(gap[strict] > 0).float().mean().item() if strict.any() else None,
                   untied_pairs=int(strict.sum()),
                   logloss=F.softplus(-gap).mean().item(), reward_gap=gap.mean().item(),
                   gaps=gap.tolist())
        if reward != "simpo":
            res["accuracy_mean_logp"] = (cs / cl > rs / rl).float().mean().item()
        results.append(res)
        extra = f"  (mean-logp acc {100 * res['accuracy_mean_logp']:.2f}%)" if "accuracy_mean_logp" in res else ""
        print(f"[{args.split}] {name:24s} {reward:9s} acc {res['correct']}/{len(rows)}  logloss {res['logloss']:.4f}  "
              f"mean r {res['reward_gap']:+.4f}{extra}", flush=True)
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

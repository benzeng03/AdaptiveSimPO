"""Score trained adapters on held-out preference pairs and select the fixed-margin baseline.

For each adapter under RUN (and the untrained base model) this reports pairwise
accuracy (r > 0), mean log-loss softplus(-r) and the mean reward gap r on the
development pairs. The fixed-margin adapter with the highest accuracy is selected,
ties broken by lower log-loss and then smaller gamma. Writes RUN/dev_scores.json,
RUN/dev_scores.csv and RUN/selection.json.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from adaptive_simpo.modeling import adapter_key, batch_logps, is_adapter_dir, load_policy, read_jsonl

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
    p.add_argument("--data", type=Path, default=HERE / "data/uf_original_test.jsonl")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-base", action="store_true", help="skip scoring the untrained base model")
    args = p.parse_args(argv)
    rows = read_jsonl(args.data)
    if not rows:
        p.error("empty development dataset")
    dirs = sorted(d for d in args.run.iterdir() if d.is_dir() and is_adapter_dir(d))
    if not dirs:
        p.error(f"no trained adapters under {args.run}")
    cfgs = {d.name: json.loads((d / "training_args.json").read_text(encoding="utf-8")) for d in dirs}
    first = cfgs[dirs[0].name]
    model = load_policy(first["model"], {d.name: d for d in dirs}, first["device"], first["bf16"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(first["model"])
    names = ([] if args.no_base else ["base"]) + [d.name for d in dirs]
    results = []
    for name in names:
        cfg = first if name == "base" else cfgs[name]
        gaps = []
        ctx = model.disable_adapter() if name == "base" else None
        if ctx is None:
            model.set_adapter(adapter_key(name))
        else:
            ctx.__enter__()
        try:
            for i in range(0, len(rows), args.batch_size):
                c, r = batch_logps(model, rows[i:i + args.batch_size], tok, cfg["max_length"],
                                   cfg["max_prompt"], cfg["device"], cfg.get("chat_prompt", True))
                gaps.append((cfg["beta"] * (c.float() - r.float())).cpu())
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        gap = torch.cat(gaps)
        res = dict(adapter=name, margin="base" if name == "base" else cfg["margin"],
                   gamma=cfg["gamma"] if name != "base" and cfg["margin"] == "fixed" else None,
                   pairs=len(rows), correct=int((gap > 0).sum()),
                   accuracy=(gap > 0).float().mean().item(),
                   logloss=F.softplus(-gap).mean().item(), reward_gap=gap.mean().item(),
                   gaps=gap.tolist())
        results.append(res)
        print(f"[dev] {name:24s} acc {res['correct']}/{len(rows)}  logloss {res['logloss']:.4f}  "
              f"mean r {res['reward_gap']:+.4f}", flush=True)
    (args.run / "dev_scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (args.run / "dev_scores.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [k for k in results[0] if k != "gaps"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    best = select_fixed(results)
    (args.run / "selection.json").write_text(json.dumps(dict(
        best_fixed=best, rule="max dev accuracy, then min dev logloss, then min gamma",
        data=str(args.data)), indent=2), encoding="utf-8")
    print(f"[dev] selected fixed baseline: {best}")


if __name__ == "__main__":
    main()

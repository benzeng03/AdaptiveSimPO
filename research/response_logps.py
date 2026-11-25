"""Per-response average log-probabilities and answer lengths on the merged test set.

score_pairs.py keeps only the reward gap r = beta * (logp_w - logp_l) of each pair. The
SimPO-paper style figures (research/plot_simpo_figures.py) also need the two terms separately
(e.g. the log-likelihood of chosen responses under each margin) and the answer lengths, so this
script recomputes them with the same encoding and truncation as training and scoring.

For the base model and every adapter it writes chosen/rejected mean per-token log-probabilities,
plus the number of answer tokens that count in that mean (after truncation, EOS included).
plot_simpo_figures.py checks that beta * (chosen - rejected) matches the stored gaps.

    python research/response_logps.py [--out outputs/response_logps_devtest1986.json]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from adaptive_simpo.modeling import adapter_key, batch_logps, encode_pair, load_policy, read_jsonl  # noqa: E402

ADAPTERS = ["fixed-g0", "fixed-g0.5", "fixed-g1", "fixed-g1.5", "fixed-g2", "fixed-g2.5", "calibrated-batch"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devtest", type=Path, default=HERE / "outputs/devtest1986_scores.json")
    p.add_argument("--data", nargs=2, type=Path, default=[HERE / "data/uf_dev500.jsonl", HERE / "data/uf_test1486.jsonl"])
    p.add_argument("--adapters", nargs="+", default=ADAPTERS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", type=Path, default=HERE / "outputs/response_logps_devtest1986.json")
    a = p.parse_args(argv)

    rows = read_jsonl(a.data[0]) + read_jsonl(a.data[1])
    runs = {r["adapter"]: HERE / r["run"].replace("\\", "/") for r in json.loads(a.devtest.read_text(encoding="utf-8"))["results"]}
    dirs = {n: runs[n] / n for n in a.adapters}
    cfgs = {n: json.loads((d / "training_args.json").read_text(encoding="utf-8")) for n, d in dirs.items()}
    first = cfgs[a.adapters[0]]
    for n, c in cfgs.items():
        for k in ("model", "max_length", "max_prompt", "beta"):
            if c[k] != first[k]:
                raise SystemExit(f"{n}: {k} differs from {a.adapters[0]}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(first["model"])
    model = load_policy(first["model"], dirs, first["device"], first["bf16"])

    lengths = [[len([t for t in labels if t != -100]) for _, labels in encode_pair(r, tok, first["max_length"], first["max_prompt"],
                                                                      truncation=first.get("truncation", "always"))]
               for r in rows]
    out = dict(data=[str(x) for x in a.data], beta=first["beta"], pairs=len(rows),
               chosen_len=[l[0] for l in lengths], rejected_len=[l[1] for l in lengths],
               score_chosen=[r.get("score_chosen") for r in rows], score_rejected=[r.get("score_rejected") for r in rows],
               models={})
    for name in ["base"] + a.adapters:
        ctx = model.disable_adapter() if name == "base" else None
        if ctx is None:
            model.set_adapter(adapter_key(name))
        else:
            ctx.__enter__()
        cs, rs = [], []
        try:
            with torch.no_grad():
                for i in range(0, len(rows), a.batch_size):
                    c, r = batch_logps(model, rows[i:i + a.batch_size], tok, first["max_length"], first["max_prompt"],
                                       first["device"], first.get("chat_prompt", True),
                                       truncation=first.get("truncation", "always"))
                    cs.append(c.float().cpu()); rs.append(r.float().cpu())
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        c, r = torch.cat(cs), torch.cat(rs)
        out["models"][name] = dict(chosen=c.tolist(), rejected=r.tolist())
        gap = first["beta"] * (c - r)
        print(f"{name:18s} acc {(gap > 0).float().mean():.4f}  mean logp chosen {c.mean():+.3f} rejected {r.mean():+.3f}",
              flush=True)
    a.out.write_text(json.dumps(out), encoding="utf-8")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

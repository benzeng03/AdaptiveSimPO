"""Blind pairwise PairRM judging of generated answers.

Compares every policy against --baseline (head to head) and against the stored
GPT-4 reference answers. Each comparison is run in both answer orders: the
candidate wins only if both orders agree, loses if both disagree, and otherwise
ties. Win rate = (W + 0.5 T) / N with a 95% prompt-bootstrap interval.

Writes GEN/judgments.json and GEN/win_rates.json.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SPECIAL = ("<|source|>", "<|candidate1|>", "<|candidate2|>", "<|candidate|>", "[SEP]")


def load_judge(folder: Path, device: str):
    from safetensors.torch import load_model
    from tokenizers import Tokenizer
    from transformers import DebertaV2Config
    spec = importlib.util.spec_from_file_location("local_pairrm", HERE / "third_party/llm_blender/pairrm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    arch = DebertaV2Config(
        vocab_size=tok.get_vocab_size(), hidden_size=1024, num_hidden_layers=24,
        num_attention_heads=16, intermediate_size=4096, relative_attention=True,
        position_buckets=256, norm_rel_ebd="layer_norm", share_att_key=True,
        pos_att_type=["p2c", "c2p"], layer_norm_eps=1e-7, position_biased_input=False,
        type_vocab_size=0, out_hidden_state_size=1024, sep_token_id=2,
        source_prefix_id=128001, cand1_prefix_id=128002, cand2_prefix_id=128003,
        cand_prefix_id=128004, n_tasks=1, drop_out=0.05)
    model = module.DebertaV2PairRM(arch)
    load_model(model, str(folder / "model.safetensors"), strict=True)
    model = model.eval().to(device)
    if device == "cuda":
        model.half()
    return tok, model


def compare(tok, model, prompt, left, right):
    """Return score in {1, 0.5, 0} for `left` versus `right`, judged in both orders."""
    def segment(marker, text, limit):
        for s in SPECIAL:
            text = text.replace(s, " ")
        seq = tok.encode(marker + text).ids
        return seq[:limit - 1] + [2] if len(seq) > limit else seq
    source = segment("<|source|>", prompt, 1224)
    cap = 412 + (1224 - len(source)) // 2
    seqs = [source + segment("<|candidate1|>", a, cap) + segment("<|candidate2|>", b, cap)
            for a, b in ((left, right), (right, left))]
    width = max(map(len, seqs))
    ids = torch.tensor([s + [0] * (width - len(s)) for s in seqs], device=next(model.parameters()).device)
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=ids.device)
    a, b = model(input_ids=ids, attention_mask=mask).logits.float().tolist()
    if not all(map(lambda v: v == v and abs(v) != float("inf"), (a, b))):
        raise ValueError("non-finite judge logits")
    score = 1.0 if a > 0 and b < 0 else 0.0 if a < 0 and b > 0 else 0.5
    return dict(score=score, forward=a, reverse=b)


def summarize(scores, seed=42, draws=50000):
    s = torch.tensor(scores, dtype=torch.float64)
    boot = s[torch.randint(len(s), (draws, len(s)), generator=torch.Generator().manual_seed(seed))].mean(1)
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return dict(wins=int((s == 1).sum()), ties=int((s == 0.5).sum()), losses=int((s == 0).sum()),
                n=len(s), win_rate=s.mean().item(), bootstrap_ci95=[lo, hi])


def load_generations(gen: Path):
    meta = json.loads((gen / "prompts.json").read_text(encoding="utf-8"))
    out = {}
    for name in meta["policies"]:
        files = sorted(gen.glob(f"[0-9][0-9][0-9]-{name}.json"))
        rows = [r for f in files for r in json.loads(f.read_text(encoding="utf-8"))]
        if [r["id"] for r in rows] != meta["indices"]:
            raise SystemExit(f"incomplete generations for {name} in {gen}")
        out[name] = rows
    return meta, out


@torch.inference_mode()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("generations", type=Path)
    p.add_argument("--baseline", required=True, help="policy name used for head-to-head comparisons")
    p.add_argument("--judge", type=Path, default=HERE / "models/PairRM")
    p.add_argument("--eval-data", type=Path, default=HERE / "data/alpaca_eval_gpt4_baseline.json")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    meta, gens = load_generations(args.generations)
    if args.baseline not in gens:
        p.error(f"baseline {args.baseline!r} not among {list(gens)}")
    reference = json.loads(args.eval_data.read_text(encoding="utf-8"))
    tok, model = load_judge(args.judge, args.device)
    others = [n for n in gens if n != args.baseline]
    judgments = []
    for j, idx in enumerate(meta["indices"]):
        ref = reference[idx]
        row = dict(id=idx, instruction=ref["instruction"], reference=ref["output"],
                   answers={n: gens[n][j]["output"] for n in gens}, vs_baseline={}, vs_reference={})
        for n in gens:
            assert gens[n][j]["instruction"] == ref["instruction"]
            row["vs_reference"][n] = compare(tok, model, ref["instruction"], row["answers"][n], ref["output"])
        for n in others:
            row["vs_baseline"][n] = compare(tok, model, ref["instruction"], row["answers"][n],
                                            row["answers"][args.baseline])
        judgments.append(row)
        print(f"[judge] {j + 1}/{len(meta['indices'])}", flush=True)
    summary = dict(baseline=args.baseline, judge="PairRM (both orders)", head_to_head={}, vs_reference={},
                   mean_tokens={n: sum(r["tokens"] for r in gens[n]) / len(gens[n]) for n in gens},
                   unfinished={n: sum(not r["ended"] for r in gens[n]) for n in gens})
    for n in others:
        summary["head_to_head"][n] = summarize([r["vs_baseline"][n]["score"] for r in judgments])
    for n in gens:
        summary["vs_reference"][n] = summarize([r["vs_reference"][n]["score"] for r in judgments])
    (args.generations / "judgments.json").write_text(json.dumps(judgments, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    (args.generations / "win_rates.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key in ("head_to_head", "vs_reference"):
        for n, s in summary[key].items():
            print(f"{key:12s} {n:24s} {s['wins']:3d}/{s['ties']:3d}/{s['losses']:3d}  "
                  f"WR {s['win_rate']:.2%}  CI [{s['bootstrap_ci95'][0]:.2%}, {s['bootstrap_ci95'][1]:.2%}]")


if __name__ == "__main__":
    main()

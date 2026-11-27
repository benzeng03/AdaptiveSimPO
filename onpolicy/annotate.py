"""Turn on-policy samples into preference pairs, as SimPO's on_policy_data_gen post_process.py and
reward_model_annotate.py: combine the per-seed files of onpolicy/decode.py, drop prompts whose
answers are identical across all seeds, score every answer with a reward model (SimPO:
RLHFlow/ArmoRM-Llama3-8B-v0.1, one conversation at a time through its chat template), and keep the
highest-scoring answer as chosen and the lowest-scoring one as rejected (first index on ties).

Writes OUT (JSONL, the training format: prompt, chosen, rejected, score_chosen, score_rejected, plus
all_generated_responses and all_rm_scores). Scores are cached in OUT.scores.jsonl, so a rerun
continues after an interruption.

    python onpolicy/annotate.py --input-dir data/onpolicy/train4096 --output data/uf_onpolicy_train4096.jsonl
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def combine(input_dir):
    """[{prompt, all_generated_responses}] over the seed files (sorted by seed), minus prompts
    whose answers are all identical; returns (records, number dropped)."""
    files = sorted(Path(input_dir).glob("output_*.jsonl"), key=lambda f: int(f.stem.split("_")[1]))
    if len(files) < 2:
        raise SystemExit(f"{input_dir}: need at least two output_<seed>.jsonl files, found {len(files)}")
    runs = [[json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] for f in files]
    if len({len(r) for r in runs}) != 1:
        raise SystemExit(f"{input_dir}: seed files have different lengths")
    records, identical = [], 0
    for rows in zip(*runs):
        if len({r["prompt"] for r in rows}) != 1:
            raise SystemExit(f"{input_dir}: seed files list the prompts in different orders")
        texts = [r["generated_text"] for r in rows]
        if len(set(texts)) == 1:
            identical += 1
            continue
        records.append(dict(prompt=rows[0]["prompt"], all_generated_responses=texts))
    return records, identical


def binarize(record, scores):
    """SimPO's rule: chosen = argmax, rejected = argmin of the reward-model scores."""
    hi = max(range(len(scores)), key=lambda i: (scores[i], -i))
    lo = min(range(len(scores)), key=lambda i: (scores[i], i))
    resp = record["all_generated_responses"]
    return dict(prompt=record["prompt"], chosen=resp[hi], rejected=resp[lo],
                score_chosen=scores[hi], score_rejected=scores[lo],
                all_generated_responses=resp, all_rm_scores=scores)


def armorm_scorer(path, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(path, device_map=device, trust_remote_code=True,
                                                               torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)

    @torch.no_grad()
    def score(prompt, answer):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt},
                                       {"role": "assistant", "content": answer}], return_tensors="pt")
        if not torch.is_tensor(ids):  # newer transformers return a BatchEncoding
            ids = ids["input_ids"]
        return model(ids.to(device)).score.float().item()
    return score


def annotate(records, score, cache):
    """Score every answer (reusing cached prompts) and binarize."""
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["prompt"]] = r["scores"]
    out = []
    with cache.open("a", encoding="utf-8") as f:
        for n, rec in enumerate(records, 1):
            scores = done.get(rec["prompt"])
            if scores is None or len(scores) != len(rec["all_generated_responses"]):
                scores = [score(rec["prompt"], a) for a in rec["all_generated_responses"]]
                f.write(json.dumps(dict(prompt=rec["prompt"], scores=scores), ensure_ascii=False) + "\n")
                f.flush()
            out.append(binarize(rec, scores))
            if n % 200 == 0:
                print(f"[annotate] {n}/{len(records)}", flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True, help="output_<seed>.jsonl files of decode.py")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reward-model", default=str(HERE / "models/ArmoRM-Llama3-8B-v0.1"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    records, identical = combine(args.input_dir)
    print(f"[annotate] {len(records)} prompts ({identical} dropped: all answers identical)", flush=True)
    pairs = annotate(records, armorm_scorer(args.reward_model, args.device),
                     args.output.with_name(args.output.name + ".scores.jsonl"))
    same = sum(r["chosen"] == r["rejected"] for r in pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs), encoding="utf-8")
    tmp.replace(args.output)
    gap = sum(r["score_chosen"] - r["score_rejected"] for r in pairs) / max(len(pairs), 1)
    print(f"[annotate] wrote {len(pairs)} pairs to {args.output} (mean score gap {gap:.4f}; "
          f"{same} with identical chosen and rejected text)", flush=True)


if __name__ == "__main__":
    main()

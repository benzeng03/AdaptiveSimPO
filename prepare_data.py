"""Build larger, prompt-disjoint train/dev/test subsets of UltraFeedback-binarized.

The existing pilot files (256 train pairs, 64 dev pairs) are kept as prefixes of the
new train and dev files, so earlier runs remain comparable. The rest is sampled with
a fixed seed:

    train  <- train_prefs  (default 4096 pairs)
    dev    <- test_prefs   (default 500 pairs; selects the fixed-margin baseline)
    test   <- test_prefs   (every remaining eligible pair; reporting only)

No prompt appears in more than one split. Pairs with an empty or identical chosen and
rejected answer are skipped. Each JSONL row is
{"prompt", "chosen", "rejected", "score_chosen", "score_rejected"}, where chosen and
rejected are the final assistant messages and the prompt is whitespace-stripped.
"""
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
UF = HERE / "data/ultrafeedback_binarized"


def load_split(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["prompt", "chosen", "rejected", "score_chosen", "score_rejected"])
    rows = []
    for r in table.to_pylist():
        prompt = (r["prompt"] or "").strip()
        chosen, rejected = r["chosen"][-1], r["rejected"][-1]
        if chosen["role"] != "assistant" or rejected["role"] != "assistant":
            continue
        rows.append(dict(prompt=prompt, chosen=chosen["content"], rejected=rejected["content"],
                         score_chosen=r["score_chosen"], score_rejected=r["score_rejected"]))
    return rows


def usable(row):
    return bool(row["prompt"] and row["chosen"].strip() and row["rejected"].strip()
                and row["chosen"] != row["rejected"])


def attach_scores(pilot, official, name):
    index = {(r["prompt"], r["chosen"], r["rejected"]): r for r in official}
    out = []
    for row in pilot:
        match = index.get((row["prompt"], row["chosen"], row["rejected"]))
        if match is None:
            raise SystemExit(f"{name}: pilot pair not found in the official split: {row['prompt'][:80]!r}")
        out.append(match)
    return out


def sample(official, n, used, rng):
    eligible, seen = [], set()
    for r in official:
        if usable(r) and r["prompt"] not in used and r["prompt"] not in seen:
            seen.add(r["prompt"])
            eligible.append(r)
    rng.shuffle(eligible)
    return eligible if n is None else eligible[:n]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows, overwrite):
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists (pass --overwrite)")
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", type=Path, default=UF / "train_prefs-00000-of-00001.parquet")
    p.add_argument("--test-parquet", type=Path, default=UF / "test_prefs-00000-of-00001.parquet")
    p.add_argument("--pilot-train", type=Path, default=HERE / "data/uf_original_train.jsonl")
    p.add_argument("--pilot-dev", type=Path, default=HERE / "data/uf_original_test.jsonl")
    p.add_argument("--train-size", type=int, default=4096)
    p.add_argument("--dev-size", type=int, default=500)
    p.add_argument("--test-size", type=int, default=0, help="0 = every remaining eligible pair")
    p.add_argument("--out-dir", type=Path, default=HERE / "data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    train_off, test_off = load_split(args.train_parquet), load_split(args.test_parquet)
    pilot_train = attach_scores(read_jsonl(args.pilot_train), train_off, "train")
    pilot_dev = attach_scores(read_jsonl(args.pilot_dev), test_off, "dev")
    if args.train_size < len(pilot_train) or args.dev_size < len(pilot_dev):
        p.error("train/dev sizes must be at least the pilot sizes")

    used = {r["prompt"] for r in pilot_train + pilot_dev}
    dev = pilot_dev + sample(test_off, args.dev_size - len(pilot_dev), used, rng)
    used |= {r["prompt"] for r in dev}
    test = sample(test_off, args.test_size or None, used, rng)
    used |= {r["prompt"] for r in test}
    # Train prompts must also avoid every test_prefs prompt that ended up in dev or test.
    train = pilot_train + sample(train_off, args.train_size - len(pilot_train), used, rng)
    if len(dev) < args.dev_size or len(train) < args.train_size or (args.test_size and len(test) < args.test_size):
        p.error("not enough eligible pairs for the requested sizes")
    splits = {"train": train, "dev": dev, "test": test}
    prompts = [{r["prompt"] for r in rows} for rows in splits.values()]
    assert all(len(rows) == len(ps) for rows, ps in zip(splits.values(), prompts)), "duplicate prompts"
    assert not (prompts[0] & prompts[1] or prompts[0] & prompts[2] or prompts[1] & prompts[2])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = args.out_dir / f"uf_{name}{len(rows)}.jsonl"
        write_jsonl(path, rows, args.overwrite)
        ties = sum(r["score_chosen"] == r["score_rejected"] for r in rows)
        print(f"[data] {name}: {len(rows)} pairs ({ties} score ties) -> {path}")


if __name__ == "__main__":
    main()

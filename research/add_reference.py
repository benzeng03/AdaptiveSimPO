"""Add the reference answers of an evaluation set to a generation directory as one more policy.

generate.py writes GEN/prompts.json (the sampled prompt indices) and GEN/NNN-<name>.json groups.
This script writes GEN/NNN-<name>.json groups with the same prompts and grouping, filled with the
"output" field of the evaluation data (for AlpacaEval: the GPT-4 reference answers), so that
research/llm_judge.py can compare any policy with the reference. Token counts use the policy
tokenizer, for comparability with the generated answers.

    python research/add_reference.py outputs/run/generations-ae128-seed42-g16 --group 16
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gen", type=Path)
    p.add_argument("--eval-data", type=Path, default=HERE / "data/alpaca_eval_gpt4_baseline.json")
    p.add_argument("--name", default="gpt4-reference")
    p.add_argument("--group", type=int, required=True, help="group size used by generate.py for GEN")
    p.add_argument("--tokenizer", default=str(HERE / "models/Qwen3-0.6B"))
    a = p.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    data = json.loads(a.eval_data.read_text(encoding="utf-8"))
    indices = json.loads((a.gen / "prompts.json").read_text(encoding="utf-8"))["indices"]
    for g in range(0, len(indices), a.group):
        existing = sorted(a.gen.glob(f"{g:03d}-*.json"))
        if not existing:
            raise SystemExit(f"no generation group {g:03d} in {a.gen}; is --group right?")
        ids = [r["id"] for r in json.loads(existing[0].read_text(encoding="utf-8"))]
        if ids != indices[g:g + a.group]:
            raise SystemExit(f"group {g:03d} does not match prompts.json; is --group right?")
        records = []
        for i in ids:
            out = data[i]["output"]
            records.append(dict(id=i, instruction=data[i]["instruction"], output=out,
                                tokens=len(tok(out, add_special_tokens=False).input_ids), ended=True))
        (a.gen / f"{g:03d}-{a.name}.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"[reference] wrote {len(indices)} answers as '{a.name}' to {a.gen}")


if __name__ == "__main__":
    main()

"""Sample answers from several adapters on the same seeded evaluation prompts (AlpacaEval by
default; --eval-data data/arena-hard-v0.1/question.jsonl for Arena-Hard).

Each policy answers the same prompts with the same per-group sampling seed, so
policies differ only in their weights. Prompts that exactly match a training
prompt are excluded. Completed groups are cached as OUT/<group>-<name>.json and
reused on rerun.

    python generate.py --output outputs/run/generations \
        --adapter fixed-g2=outputs/run/fixed-g2 --adapter calibrated=outputs/run/calibrated [--base]
"""
import argparse
import json
import random
from pathlib import Path

import torch

from adaptive_simpo.modeling import Policies, read_jsonl

HERE = Path(__file__).resolve().parent


def load_eval_data(path: Path):
    """A JSON list (AlpacaEval) or JSONL file (Arena-Hard question.jsonl) of prompts.
    The prompt is read from "instruction" or, failing that, "prompt"."""
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if path.suffix == ".json" else [json.loads(l) for l in text.splitlines() if l.strip()]
    for r in rows:
        r.setdefault("instruction", r.get("prompt"))
        if not isinstance(r["instruction"], str):
            raise SystemExit(f"{path}: every row needs a string 'instruction' or 'prompt'")
    return rows


def eval_indices(data, train_rows, n, seed):
    train = {r["prompt"] for r in train_rows if isinstance(r["prompt"], str)}
    eligible = [i for i, row in enumerate(data) if row["instruction"] not in train]
    return random.Random(seed).sample(eligible, min(n, len(eligible)))


def parse_adapters(items):
    adapters = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or name in adapters or name == "base":
            raise SystemExit(f"bad --adapter {item!r}; use unique NAME=PATH (not 'base')")
        adapters[name] = Path(path)
    return adapters


@torch.inference_mode()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--base", action="store_true", help="also generate with the untrained model")
    p.add_argument("--model", default=str(HERE / "models/Qwen3-0.6B"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eval-data", type=Path, default=HERE / "data/alpaca_eval_gpt4_baseline.json")
    p.add_argument("--train-data", type=Path, default=HERE / "data/uf_train4096.jsonl")
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-mem-fraction", type=float, default=0.9,
                   help="cap on the CUDA caching allocator (fraction of device memory); the growing KV "
                        "cache fragments memory, and without a cap Windows spills it into slow shared "
                        "memory instead of reusing cached blocks. 0 disables the cap")
    p.add_argument("--max-new", type=int, default=0,
                   help="stop after writing this many new group files (0 = no limit); used to rerun only "
                        "the group that ran out of memory under the cap, without the cap")
    args = p.parse_args(argv)
    if args.device.startswith("cuda") and args.gpu_mem_fraction > 0 and torch.cuda.is_available():
        dev = torch.device(args.device)
        torch.cuda.set_per_process_memory_fraction(
            args.gpu_mem_fraction, dev.index if dev.index is not None else torch.cuda.current_device())
    adapters = parse_adapters(args.adapter)
    names = (["base"] if args.base else []) + list(adapters)
    if not names:
        p.error("give at least one --adapter or --base")

    from transformers import AutoTokenizer
    data = load_eval_data(args.eval_data)
    indices = eval_indices(data, read_jsonl(args.train_data), args.num_prompts, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "prompts.json").write_text(json.dumps(dict(
        indices=indices, seed=args.seed, temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens, policies={n: str(adapters.get(n, "base")) for n in names},
    ), indent=2), encoding="utf-8")

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    # every full model stays loaded (a few are compared at a time; LoRA adapters share the base)
    policies = Policies(args.model, adapters, args.device, bf16=True, keep_full=len(adapters))
    eos = policies.shared.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    total, done, new = len(indices) * len(names), 0, 0
    for g in range(0, len(indices), args.group):
        if args.max_new and new >= args.max_new:
            print(f"[generate] stopping after {new} new group file(s) (--max-new)")
            return
        chosen = indices[g:g + args.group]
        texts = [tok.apply_chat_template([{"role": "user", "content": data[i]["instruction"]}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for i in chosen]
        batch = tok(texts, padding=True, add_special_tokens=False, return_tensors="pt").to(args.device)
        for name in names:
            dest = args.output / f"{g:03d}-{name}.json"
            if dest.exists():
                done += len(chosen)
                continue
            if args.max_new and new >= args.max_new:
                break
            torch.manual_seed(args.seed + g)
            gen_kwargs = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=0,
                              max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
            with policies.use(name) as model:
                seq = model.generate(**batch, **gen_kwargs)
            records = []
            for i, tail in zip(chosen, seq[:, batch.input_ids.size(1):].tolist()):
                stop = next((n for n, t in enumerate(tail) if t in eos), len(tail))
                records.append(dict(id=i, instruction=data[i]["instruction"],
                                    output=tok.decode(tail[:stop], skip_special_tokens=True),
                                    tokens=stop, ended=stop < len(tail)))
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            tmp.replace(dest)
            del seq
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            done += len(chosen)
            new += 1
            print(f"[generate] {name} prompts {g + 1}-{g + len(chosen)} | {done}/{total}", flush=True)
    print(f"[generate] done -> {args.output}")


if __name__ == "__main__":
    main()

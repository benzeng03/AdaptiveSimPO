"""Sample on-policy answers, as SimPO's on_policy_data_gen/decode.py: every prompt of a JSONL file
(field "prompt") is answered once per seed by the policy itself, through its chat template with
thinking off (the same prompt format train.py uses). Defaults are SimPO's: seeds 13 21 42 79 100,
temperature 0.8, top-p 0.95, no top-k, at most 4096 new tokens (cloud/onpolicy.sh passes the
Qwen3 model-card sampling instead: temperature 0.7, top-p 0.8, top-k 20).

Writes OUT/output_<seed>.jsonl (one line per prompt: prompt, generated_text, tokens, finish_reason);
a seed whose file already exists is skipped, so a rerun continues after an interruption.

    python onpolicy/decode.py --data data/uf_train4096.jsonl --output-dir data/onpolicy/train4096

--engine vllm (default, as SimPO) needs the vllm package; --engine hf uses transformers.generate
(slow, for tests and machines without vLLM).
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SEEDS = (13, 21, 42, 79, 100)


def read_prompts(path):
    """Unique prompts in order of first appearance."""
    seen, prompts = set(), []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)["prompt"]
            if p not in seen:
                seen.add(p)
                prompts.append(p)
    return prompts


def chat_texts(tokenizer, prompts):
    return [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False) for p in prompts]


def write_jsonl(path, rows):
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def decode_vllm(args, prompts, seeds):
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              dtype="bfloat16", seed=0)
    texts = chat_texts(llm.get_tokenizer(), prompts)
    for seed in seeds:
        params = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k or -1,
                                max_tokens=args.max_tokens, seed=seed)
        outs = llm.generate(texts, params)
        yield seed, [dict(prompt=p, generated_text=o.outputs[0].text, tokens=len(o.outputs[0].token_ids),
                          finish_reason=o.outputs[0].finish_reason) for p, o in zip(prompts, outs)]


def decode_hf(args, prompts, seeds):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    texts = chat_texts(tok, prompts)
    for seed in seeds:
        rows = []
        for b in range(0, len(texts), args.batch_size):
            torch.manual_seed(seed * 100003 + b)
            batch = tok(texts[b:b + args.batch_size], padding=True, add_special_tokens=False,
                        return_tensors="pt").to(args.device)
            with torch.inference_mode():
                seq = model.generate(**batch, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                     top_k=args.top_k, max_new_tokens=args.max_tokens, pad_token_id=tok.pad_token_id)
            for p, tail in zip(prompts[b:b + args.batch_size], seq[:, batch.input_ids.size(1):].tolist()):
                stop = next((n for n, t in enumerate(tail) if t in eos), len(tail))
                rows.append(dict(prompt=p, generated_text=tok.decode(tail[:stop], skip_special_tokens=True),
                                 tokens=stop, finish_reason="stop" if stop < len(tail) else "length"))
        yield seed, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(HERE / "models/Qwen3-0.6B"))
    p.add_argument("--data", required=True, help="JSONL with a 'prompt' field")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=0, help="0 = no top-k filtering")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--engine", choices=("vllm", "hf"), default="vllm")
    p.add_argument("--max-model-len", type=int, default=8192, help="vllm: prompt + answer token budget")
    p.add_argument("--gpu-mem", type=float, default=0.9, help="vllm: fraction of GPU memory to use")
    p.add_argument("--batch-size", type=int, default=16, help="hf engine only")
    p.add_argument("--device", default="cuda", help="hf engine only")
    p.add_argument("--limit", type=int, default=0, help="only the first N prompts (0 = all; smoke tests)")
    args = p.parse_args(argv)

    prompts = read_prompts(args.data)
    if args.limit:
        prompts = prompts[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(model=args.model, data=str(args.data), prompts=len(prompts), seeds=args.seeds,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens,
                engine=args.engine)
    meta_file = args.output_dir / "decode_args.json"
    if meta_file.exists():
        old = json.loads(meta_file.read_text(encoding="utf-8"))
        diff = {k: (old.get(k), v) for k, v in meta.items() if k not in ("seeds", "model") and old.get(k) != v}
        if diff:
            raise SystemExit(f"{args.output_dir} was decoded with other settings {diff}; use another --output-dir")
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    todo = [s for s in args.seeds if not (args.output_dir / f"output_{s}.jsonl").exists()]
    print(f"[decode] {len(prompts)} prompts, seeds {args.seeds} ({len(args.seeds) - len(todo)} already done), "
          f"{args.engine}, T={args.temperature} top-p={args.top_p} top-k={args.top_k or 'off'} "
          f"max {args.max_tokens} tokens", flush=True)
    if not todo:
        return
    engine = decode_vllm if args.engine == "vllm" else decode_hf
    for seed, rows in engine(args, prompts, todo):
        write_jsonl(args.output_dir / f"output_{seed}.jsonl", rows)
        cut = sum(r["finish_reason"] == "length" for r in rows)
        mean = sum(r["tokens"] for r in rows) / max(len(rows), 1)
        print(f"[decode] seed {seed}: {len(rows)} answers, mean {mean:.0f} tokens, {cut} hit the limit", flush=True)


if __name__ == "__main__":
    main()

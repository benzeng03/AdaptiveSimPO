"""Train one SimPO policy (fixed or calibrated margin) on JSONL pairs.

Every variant shares the model, data, shuffling order, LoRA initialisation and
optimiser settings for a given --seed, so only the margin rule differs.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from adaptive_simpo.margins import KINDS, MarginConfig, simpo_loss
from adaptive_simpo.modeling import batch_logps, read_jsonl

HERE = Path(__file__).resolve().parent


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(HERE / "models/Qwen3-0.6B"))
    p.add_argument("--data", default=str(HERE / "data/uf_train4096.jsonl"))
    p.add_argument("--output", required=True)
    p.add_argument("--margin", choices=KINDS, default="calibrated")
    p.add_argument("--gamma", type=float, default=0.5, help="fixed margin (reward units)")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accumulate", type=int, default=1,
                   help="micro-batches per update; margins are computed per micro-batch")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--schedule", choices=("constant", "cosine"), default="constant",
                   help="cosine: 10%% warmup, cosine decay, grad-norm clip 1 (SimPO recipe)")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--max-prompt", type=int, default=192)
    p.add_argument("--no-chat-prompt", dest="chat_prompt", action="store_false")
    p.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def margin_config(args):
    return MarginConfig(kind=args.margin, gamma=args.gamma)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 < args.max_prompt < args.max_length:
        parser.error("require 0 < max-prompt < max-length")
    if min(args.epochs, args.batch_size, args.accumulate) < 1:
        parser.error("epochs, batch-size and accumulate must be positive")
    cfg = margin_config(args)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    rows = read_jsonl(args.data)
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=list,
                        generator=torch.Generator().manual_seed(args.seed))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"]))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.lora:
            model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    updates = math.ceil(len(loader) / args.accumulate) * args.epochs
    scheduler = None
    if args.schedule == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(0.1 * updates), updates)

    print(f"[train] {cfg.name}: {len(rows)} pairs, {len(loader)} batches/epoch, {updates} updates",
          flush=True)
    log = (output / "metrics.jsonl").open("w", encoding="utf-8")
    totals = dict(gamma=0.0, active=0.0, n=0, steps=0, share=0.0, weight=0.0, k=0.0)
    step, start = 0, time.time()
    done = False
    for epoch in range(args.epochs):
        for index, pairs in enumerate(loader):
            if index % args.accumulate == 0:
                optimizer.zero_grad(set_to_none=True)
                window = min(len(rows) - index * args.batch_size, args.batch_size * args.accumulate)
            chosen, rejected = batch_logps(model, pairs, tokenizer, args.max_length, args.max_prompt,
                                           args.device, args.chat_prompt)
            loss, scores, gamma, stats = simpo_loss(chosen, rejected, args.beta, cfg)
            (loss * (len(pairs) / window)).backward()
            if (index + 1) % args.accumulate == 0 or index + 1 == len(loader):
                if scheduler is not None:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            step += 1
            totals["gamma"] += gamma.sum().item()
            totals["active"] += (gamma > 0).sum().item()
            totals["n"] += gamma.numel()
            record = dict(step=step, epoch=epoch, loss=loss.item(), reward_gap=scores.mean().item(),
                          accuracy=(scores > 0).float().mean().item(), **stats)
            totals["share"] += stats["weight_share_misranked"]
            totals["weight"] += stats["weight_mean"]
            totals["k"] += stats.get("calib_k", 0.0)
            totals["steps"] += 1
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                              for k, v in record.items()}), flush=True)
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        if done:
            break
    log.close()

    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    summary = dict(variant=cfg.name, margin=cfg.to_dict(), steps=step, pairs_seen=totals["n"],
                   seconds=round(time.time() - start, 1),
                   gamma_mean=totals["gamma"] / totals["n"],
                   active=totals["active"] / totals["n"])
    if totals["steps"]:
        summary.update(weight_mean=totals["weight"] / totals["steps"],
                       weight_share_misranked=totals["share"] / totals["steps"])
        if cfg.kind == "calibrated":
            summary.update(calib_k_mean=totals["k"] / totals["steps"])
    (output / "margin_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "training_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"[train] saved {output} | {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()

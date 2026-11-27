"""Train one preference-optimisation policy on JSONL pairs: a margin rule (--margin fixed or
calibrated) applied to a reward gap (--reward simpo, dpo, dpo-noref, r-dpo or orpo; see
adaptive_simpo/rewards.py). The loss is softplus(gamma_i - r_i), plus the chosen answer's NLL for ORPO.

DPO and R-DPO use summed log-probabilities and the frozen base model as reference: with LoRA the
reference is the same model with the adapter disabled, so no second copy is loaded; with full
fine-tuning a frozen copy of the base model is kept on the device.

Every variant shares the model, data, shuffling order, LoRA initialisation and
optimiser settings for a given --seed, so only the margin rule differs.

With --checkpoint-every N the trainable weights, optimiser state and running statistics are
saved to OUTPUT/checkpoint every N steps; rerunning the same command with --resume continues
from there (the data order is fixed by --seed, so the batches already trained on are skipped).
The checkpoint is removed once the run finishes.
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from adaptive_simpo.margins import FAMILIES, KINDS, MarginConfig, MomentEMA, margin_loss
from adaptive_simpo.modeling import TRUNCATIONS, batch_logp_sums, read_jsonl
from adaptive_simpo.rewards import DEFAULT_BETA, NEEDS_REFERENCE, REWARDS, reward_gap
from adaptive_simpo.optim import MasterAdamW, make_optimizer

HERE = Path(__file__).resolve().parent


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(HERE / "models/Qwen3-0.6B"))
    p.add_argument("--data", default=str(HERE / "data/uf_train4096.jsonl"))
    p.add_argument("--output", required=True)
    p.add_argument("--reward", choices=REWARDS, default="simpo",
                   help="reward gap the margin acts on (adaptive_simpo/rewards.py)")
    p.add_argument("--margin", choices=KINDS, default="calibrated", help="margin rule")
    p.add_argument("--gamma", type=float, default=0.5, help="fixed margin (reward units)")
    p.add_argument("--k-ema", type=float, default=0.0,
                   help="calibrated margin: estimate k = 2 mean / var from exponential moving averages "
                        "(this decay, bias-corrected, one update per micro-batch) of the gap mean and "
                        "second moment instead of the current batch alone; 0 = current batch only")
    p.add_argument("--k-family", choices=FAMILIES, default="normal",
                   help="calibrated margin: reference distribution of the gaps, estimated from their mean "
                        "and variance. normal: linear logit 2 mean r / var; logistic: logistic location "
                        "model, a logit that rises with slope (2/s) tanh(mean/2s) and saturates at "
                        "2 mean / s, s = sqrt(3 var) / pi (adaptive_simpo/margins.py)")
    p.add_argument("--beta", type=float, default=None,
                   help="reward scale; default 2.0 for simpo, 0.1 for dpo / dpo-noref / r-dpo, 1.0 for orpo")
    p.add_argument("--rdpo-alpha", type=float, default=0.05,
                   help="r-dpo: r = DPO gap + alpha * (|y_w| - |y_l|) in answer tokens")
    p.add_argument("--orpo-lambda", type=float, default=0.1,
                   help="orpo: loss = NLL(y_w) + lambda * preference loss")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accumulate", type=int, default=1,
                   help="micro-batches per update; margins are computed per micro-batch")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--schedule", choices=("constant", "cosine"), default="constant",
                   help="cosine: 10%% warmup, cosine decay, grad-norm clip 1 (SimPO recipe)")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--max-prompt", type=int, default=192)
    p.add_argument("--truncation", choices=TRUNCATIONS, default="always",
                   help="always: cut every prompt to --max-prompt; simpo: cut only pairs longer than "
                        "--max-length, as SimPO/TRL do (see modeling.encode_pair)")
    p.add_argument("--no-chat-prompt", dest="chat_prompt", action="store_false")
    p.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True,
                   help="--no-lora trains every weight (full fine-tuning)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp32-master", choices=("auto", "gpu", "cpu", "off"), default="auto",
                   help="AdamW on fp32 copies of bf16 weights (auto: whenever a trainable weight is "
                        "below fp32, i.e. full fine-tuning in bf16; cpu: keep the copies and moments "
                        "in CPU memory)")
    p.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    p.add_argument("--checkpoint-every", type=int, default=32,
                   help="save a resumable checkpoint every N steps (at optimiser updates); 0 = never")
    p.add_argument("--resume", action="store_true",
                   help="continue from OUTPUT/checkpoint if there is one, else start over in OUTPUT")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def margin_config(args):
    return MarginConfig(kind=args.margin, gamma=args.gamma, k_ema=args.k_ema, family=args.k_family)


def variant_name(args, cfg):
    """fixed-g2 / calibrated for SimPO; otherwise <reward>[-a<alpha>|-l<lambda>][-b<beta>]-<margin>,
    with beta only when it is not the reward's default (e.g. dpo-fixed-g0, rdpo-a0.05-calibrated)."""
    if args.reward == "simpo" and args.beta == DEFAULT_BETA["simpo"]:
        return cfg.name
    tag = {"r-dpo": f"rdpo-a{args.rdpo_alpha:g}", "orpo": f"orpo-l{args.orpo_lambda:g}"}.get(args.reward, args.reward)
    if args.beta != DEFAULT_BETA[args.reward]:
        tag += f"-b{args.beta:g}"
    return f"{tag}-{cfg.name}"


# Arguments that may differ between an interrupted run and its resumption.
RESUME_FREE = {"output", "resume", "checkpoint_every", "device", "max_steps"}


def save_checkpoint(output, model, optimizer, scheduler, state):
    """Write OUTPUT/checkpoint atomically: build checkpoint.tmp, then swap it in."""
    tmp, final, old = output / "checkpoint.tmp", output / "checkpoint", output / "checkpoint.old"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    if not isinstance(optimizer, MasterAdamW):  # with fp32 masters the weights are in the optimiser
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                   tmp / "trainable.pt")
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), tmp / "scheduler.pt")
    (tmp / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    if final.exists():
        shutil.rmtree(old, ignore_errors=True)
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)


def find_checkpoint(output):
    for name in ("checkpoint", "checkpoint.old"):  # .old survives a crash in the middle of the swap
        if (output / name / "state.json").exists():
            return output / name
    return None


def load_checkpoint(ckpt, model, optimizer, scheduler):
    if (ckpt / "trainable.pt").exists():
        trainable = torch.load(ckpt / "trainable.pt", map_location="cpu")
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if set(trainable) != set(params):
            raise SystemExit(f"checkpoint {ckpt} does not match the model's trainable parameters")
        with torch.no_grad():
            for name, value in trainable.items():
                params[name].copy_(value)
    optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location="cpu"))
    if scheduler is not None:
        scheduler.load_state_dict(torch.load(ckpt / "scheduler.pt", map_location="cpu"))
    return json.loads((ckpt / "state.json").read_text(encoding="utf-8"))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 < args.max_prompt < args.max_length:
        parser.error("require 0 < max-prompt < max-length")
    if min(args.epochs, args.batch_size, args.accumulate) < 1:
        parser.error("epochs, batch-size and accumulate must be positive")
    if args.beta is None:
        args.beta = DEFAULT_BETA[args.reward]
    if not (math.isfinite(args.beta) and args.beta > 0):
        parser.error("beta must be positive and finite")
    if not math.isfinite(args.rdpo_alpha) or not (math.isfinite(args.orpo_lambda) and args.orpo_lambda >= 0):
        parser.error("rdpo-alpha must be finite and orpo-lambda finite and >= 0")
    if args.k_ema and args.margin != "calibrated":
        parser.error("--k-ema applies only to --margin calibrated")
    if args.k_family != "normal" and args.margin != "calibrated":
        parser.error("--k-family applies only to --margin calibrated")
    try:
        cfg = margin_config(args)
    except ValueError as err:
        parser.error(str(err))
    ema = MomentEMA(cfg.k_ema) if cfg.kind == "calibrated" and cfg.k_ema else None
    variant = variant_name(args, cfg)
    output = Path(args.output)
    if args.checkpoint_every < 0:
        parser.error("checkpoint-every must be >= 0")
    ckpt = find_checkpoint(output) if args.resume and output.exists() else None
    if ckpt is None and output.exists() and any(output.iterdir()):
        if not args.resume:
            parser.error(f"output directory is not empty: {output}")
        if (output / "margin_stats.json").exists():
            parser.error(f"{output} already holds a finished run")
        print(f"[train] no checkpoint in {output}: starting over", flush=True)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if ckpt is not None:
        saved = json.loads((ckpt / "state.json").read_text(encoding="utf-8"))["args"]
        # an option added after the checkpoint was written counts as its default there
        changed = {k: (saved.get(k, parser.get_default(k)), v) for k, v in vars(args).items()
                   if k not in RESUME_FREE and saved.get(k, parser.get_default(k)) != v}
        if changed:
            parser.error(f"--resume with different arguments than the checkpoint (old, new): {changed}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    rows = read_jsonl(args.data)
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=list,
                        generator=torch.Generator().manual_seed(args.seed))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    reference = None  # frozen base model for DPO / R-DPO under full fine-tuning
    if args.reward in NEEDS_REFERENCE and not args.lora:
        reference = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
        reference.requires_grad_(False)
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
    optimizer = make_optimizer(params, args.lr, args.fp32_master)
    updates = math.ceil(len(loader) / args.accumulate) * args.epochs
    scheduler = None
    if args.schedule == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(getattr(optimizer, "inner", optimizer),
                                                    math.ceil(0.1 * updates), updates)

    print(f"[train] {variant}: {len(rows)} pairs, {len(loader)} batches/epoch, {updates} updates, "
          f"{'LoRA' if args.lora else 'full'} ({sum(p.numel() for p in params):,} trainable), "
          f"{type(optimizer).__name__}{' on CPU' if args.fp32_master == 'cpu' else ''}", flush=True)
    totals = dict(gamma=0.0, active=0.0, n=0, steps=0, share=0.0, weight=0.0, k=0.0)
    step, start, elapsed = 0, time.time(), 0.0
    next_batch = (0, 0)  # (epoch, index) of the first batch still to train on
    metrics = output / "metrics.jsonl"
    if ckpt is not None:
        state = load_checkpoint(ckpt, model, optimizer, scheduler)
        step, totals, elapsed = state["step"], state["totals"], state["seconds"]
        next_batch = tuple(state["next_batch"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        kept = metrics.read_text(encoding="utf-8").splitlines()[:step] if metrics.exists() else []
        metrics.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        print(f"[train] resuming from {ckpt.name} at step {step} (epoch {next_batch[0]}, batch {next_batch[1]})",
              flush=True)
    log = metrics.open("a" if ckpt is not None else "w", encoding="utf-8")
    try:
        done = False
        for epoch in range(args.epochs):
            # Iterating the loader draws this epoch's shuffle from the seeded generator, so skipping
            # batches (rather than not iterating) keeps the data order of an uninterrupted run.
            for index, pairs in enumerate(loader):
                if (epoch, index) < next_batch:
                    continue
                if index % args.accumulate == 0:
                    optimizer.zero_grad(set_to_none=True)
                    window = min(len(rows) - index * args.batch_size, args.batch_size * args.accumulate)
                encode = dict(tokenizer=tokenizer, max_length=args.max_length, max_prompt=args.max_prompt,
                              device=args.device, chat_prompt=args.chat_prompt, truncation=args.truncation)
                cs, rs, cl, rl = batch_logp_sums(model, pairs, **encode)
                ref = {}
                if args.reward in NEEDS_REFERENCE:
                    with torch.no_grad():
                        if reference is not None:
                            ref_cs, ref_rs, _, _ = batch_logp_sums(reference, pairs, **encode)
                        else:
                            with model.disable_adapter():
                                ref_cs, ref_rs, _, _ = batch_logp_sums(model, pairs, **encode)
                    ref = dict(ref_chosen_sum=ref_cs, ref_rejected_sum=ref_rs)
                gaps = reward_gap(args.reward, args.beta, cs, rs, cl, rl, alpha=args.rdpo_alpha, **ref)
                loss, scores, gamma, stats = margin_loss(gaps, cfg, ema)
                if args.reward == "orpo":
                    nll = -(cs.float() / cl).mean()
                    stats.update(pref_loss=loss.item(), nll=nll.item())
                    loss = nll + args.orpo_lambda * loss
                (loss * (len(pairs) / window)).backward()
                updated = (index + 1) % args.accumulate == 0 or index + 1 == len(loader)
                if updated:
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
                if stats.get("calib_c") is not None:  # logistic family; checkpoints before it lack these keys
                    totals["c"] = totals.get("c", 0.0) + stats["calib_c"]
                    totals["c_n"] = totals.get("c_n", 0) + 1
                totals["steps"] += 1
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in record.items()}), flush=True)
                following = (epoch, index + 1) if index + 1 < len(loader) else (epoch + 1, 0)
                last = following == (args.epochs, 0) or (args.max_steps and step >= args.max_steps)
                if updated and args.checkpoint_every and step % args.checkpoint_every == 0 and not last:
                    saving = time.time()
                    save_checkpoint(output, model, optimizer, scheduler, dict(
                        step=step, next_batch=following, totals=totals,
                        seconds=elapsed + time.time() - start, args=vars(args),
                        ema=None if ema is None else ema.state_dict()))
                    print(f"[train] checkpoint at step {step} ({time.time() - saving:.1f} s)", flush=True)
                if args.max_steps and step >= args.max_steps:
                    done = True
                    break
            if done:
                break
    finally:
        log.close()

    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    summary = dict(variant=variant, reward=args.reward, beta=args.beta, margin=cfg.to_dict(), steps=step, pairs_seen=totals["n"],
                   seconds=round(elapsed + time.time() - start, 1),
                   gamma_mean=totals["gamma"] / totals["n"],
                   active=totals["active"] / totals["n"])
    if totals["steps"]:
        summary.update(weight_mean=totals["weight"] / totals["steps"],
                       weight_share_misranked=totals["share"] / totals["steps"])
        if cfg.kind == "calibrated":
            summary.update(calib_k_mean=totals["k"] / totals["steps"])
        if totals.get("c_n"):
            summary.update(calib_c_mean=totals["c"] / totals["c_n"], calib_c_steps=totals["c_n"])
    # margin_stats.json last: run.sh treats it as "this variant is finished"
    (output / "training_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (output / "margin_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name in ("checkpoint", "checkpoint.old", "checkpoint.tmp"):
        shutil.rmtree(output / name, ignore_errors=True)
    print(f"[train] saved {output} | {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()

"""Peak GPU memory and time of one SimPO training step at several batch widths.

Run it once on a new machine before training with longer sequences: every batch of 16 pairs is
padded to its longest sequence, so the widths below cover the median, p90, p99 and longest
batch of data/uf_train4096.jsonl without a length limit (seed 42 order). The learning rate is 0,
so nothing is trained.

    .venv/bin/python cloud/len_probe.py                  # LoRA, no length limit
    .venv/bin/python cloud/len_probe.py --mode full                  # fp32 master weights on the GPU
    .venv/bin/python cloud/len_probe.py --mode full --fp32-master cpu
    .venv/bin/python cloud/len_probe.py --max-length 1024 --max-prompt 512 --truncation simpo --widths 448 800 1024
"""
import argparse
import bisect
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from adaptive_simpo.margins import MarginConfig, simpo_loss  # noqa: E402
from adaptive_simpo.modeling import TRUNCATIONS, batch_logps, encode_pair  # noqa: E402
from adaptive_simpo.optim import make_optimizer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("lora", "full"), default="lora")
    p.add_argument("--model", default=str(HERE / "models/Qwen3-0.6B"))
    p.add_argument("--data", default=str(HERE / "data/uf_train4096.jsonl"))
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--max-prompt", type=int, default=4096)
    p.add_argument("--truncation", choices=TRUNCATIONS, default="always")
    p.add_argument("--widths", type=int, nargs="+", default=[448, 1108, 1624, 2234, 2507])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--fp32-master", choices=("auto", "gpu", "cpu", "off"), default="auto",
                   help="as in train.py (auto = fp32 masters on the GPU for --mode full)")
    p.add_argument("--mem-fraction", type=float, default=0.95,
                   help="cap the CUDA allocator so a batch that does not fit fails with OOM")
    a = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(line) for line in open(a.data, encoding="utf-8")]

    def width(r):  # longer of the two sequences after truncation
        return max(len(ids) for ids, _ in encode_pair(r, tok, a.max_length, a.max_prompt, True, a.truncation))

    order = sorted(rows, key=width)
    lengths = [width(r) for r in order]
    batches = []
    for target in a.widths:  # the batch_size pairs just below each target width
        j = max(bisect.bisect_right(lengths, target), a.batch_size)
        batches.append(order[j - a.batch_size:j])

    if a.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
    if a.mode == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0,
                                                 target_modules=["q_proj", "v_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.mode == "lora":
        model.enable_input_require_grads()
    model.train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = make_optimizer(params, 0.0, a.fp32_master)
    cfg = MarginConfig(kind="calibrated")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"{torch.cuda.get_device_name(0)}, {total:.1f} GiB, mode {a.mode}, {type(opt).__name__} "
          f"(fp32 master {a.fp32_master}), cap {a.mem_fraction}", flush=True)
    for i, batch in enumerate([batches[0]] + batches):  # the first pass only warms up
        w = max(width(r) for r in batch)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        try:
            c, r = batch_logps(model, batch, tok, a.max_length, a.max_prompt, "cuda", True, truncation=a.truncation)
            loss, *_ = simpo_loss(c, r, 2.0, cfg)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            msg = (f"{time.time() - t:5.1f} s/step, peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
                   f"reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB")
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            msg = "OOM"
        print(f"{'warm-up' if i == 0 else 'width':8s} {w:5d}: {msg}", flush=True)


if __name__ == "__main__":
    main()

"""Tokenisation and length-normalised log-probabilities for preference pairs."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_ids(tokenizer, prompt, chat_prompt=True):
    """Token ids of the prompt, ending with the assistant generation prefix."""
    if chat_prompt and isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        kwargs = {"enable_thinking": False} if chat_prompt else {}
        ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                            return_dict=False, **kwargs)
    else:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if tokenizer.bos_token_id is not None:
            ids.insert(0, tokenizer.bos_token_id)
    if not ids:
        raise ValueError("prompt needs at least one token")
    return list(ids)


def encode_pair(row, tokenizer, max_length, max_prompt, chat_prompt=True):
    """Return [(ids, labels) for chosen, (ids, labels) for rejected].

    Long prompts keep their end; long answers keep their start. EOS is appended
    to each answer and counts as an answer token when it survives truncation."""
    prefix = prompt_ids(tokenizer, row["prompt"], chat_prompt)[-max_prompt:]
    result = []
    for key in ("chosen", "rejected"):
        answer = tokenizer.encode(row[key], add_special_tokens=False)
        if not answer:
            raise ValueError(f"empty {key} response")
        eos = tokenizer.eos_token_id
        if eos is not None and answer[-1] != eos:
            answer.append(eos)
        answer = answer[:max_length - len(prefix)]
        result.append((prefix + answer, [-100] * len(prefix) + answer))
    return result


def _chunked_answer_logps(lm, ids, mask, labels, chunk=64, average=True):
    """Project only answer positions through lm_head, in checkpointed chunks.

    Avoids materialising (batch, length, vocab) logits for 150k-vocab models."""
    hidden = lm.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    targets = labels[:, 1:]
    seq, pos = targets.ne(-100).nonzero(as_tuple=True)
    states_all, tokens_all = hidden[seq, pos], targets[seq, pos]

    def token_nll(states, tokens):
        return F.cross_entropy(lm.lm_head(states).float(), tokens, reduction="none")

    parts = []
    for states, tokens in zip(states_all.split(chunk), tokens_all.split(chunk)):
        if torch.is_grad_enabled():
            parts.append(checkpoint(token_nll, states, tokens, use_reentrant=False))
        else:
            parts.append(token_nll(states, tokens))
    nll = torch.cat(parts)
    totals = nll.new_zeros(ids.size(0)).index_add(0, seq, nll)
    return -totals / targets.ne(-100).sum(1) if average else -totals


def batch_logps(model, rows, tokenizer, max_length, max_prompt, device, chat_prompt=True, average=True):
    """Per-token mean (average=True, SimPO) or summed (average=False, DPO) log-probabilities
    (chosen, rejected) for a list of pairs."""
    encoded = [encode_pair(r, tokenizer, max_length, max_prompt, chat_prompt) for r in rows]
    sequences = [pair[i] for i in (0, 1) for pair in encoded]
    width = max(len(ids) for ids, _ in sequences)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids = torch.full((len(sequences), width), pad, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, (tokens, targets) in enumerate(sequences):
        ids[i, :len(tokens)] = torch.tensor(tokens)
        labels[i, :len(tokens)] = torch.tensor(targets)
        mask[i, :len(tokens)] = 1
    ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
    lm = model.get_base_model() if hasattr(model, "peft_config") else model
    if hasattr(lm, "model") and hasattr(lm, "lm_head"):
        logps = _chunked_answer_logps(lm, ids, mask, labels, average=average)
    else:
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        targets = labels[:, 1:]
        nll = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), targets,
                              ignore_index=-100, reduction="none")
        logps = -nll.sum(1) / targets.ne(-100).sum(1) if average else -nll.sum(1)
    return logps.chunk(2)


def adapter_key(name: str) -> str:
    """PEFT module names cannot contain '.', e.g. the directory name fixed-g0.5."""
    return name.replace(".", "_")


def load_policy(base, adapters=None, device="cuda", bf16=True):
    """Load a base causal LM, optionally with named LoRA adapters {name: path}.

    Select an adapter with model.set_adapter(adapter_key(name))."""
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    if adapters:
        from peft import PeftModel
        items = list(adapters.items())
        model = PeftModel.from_pretrained(model, str(items[0][1]), adapter_name=adapter_key(items[0][0]))
        for name, path in items[1:]:
            model.load_adapter(str(path), adapter_name=adapter_key(name))
    return model.eval()


def is_adapter_dir(path: Path) -> bool:
    return all((Path(path) / f).is_file() for f in
               ("adapter_model.safetensors", "adapter_config.json", "training_args.json"))


def is_full_dir(path: Path) -> bool:
    """A fully fine-tuned model saved by train.py --no-lora."""
    path = Path(path)
    return ((path / "config.json").is_file() and (path / "training_args.json").is_file()
            and not (path / "adapter_config.json").exists()
            and any(path.glob("model*.safetensors")))


def is_policy_dir(path: Path) -> bool:
    return is_adapter_dir(path) or is_full_dir(path)


class Policies:
    """The base model, LoRA adapters and fully fine-tuned models behind one interface.

    LoRA adapters share one copy of the base model. Each full model is a separate copy, loaded
    on first use; at most keep_full of them stay in memory (the least recently used is freed).

        policies = Policies(base, {"fixed-g2": path, ...}, "cuda")
        with policies.use("fixed-g2") as model: ...
    """

    def __init__(self, base, dirs, device="cuda", bf16=True, keep_full=1):
        self.device, self.keep_full = device, max(1, keep_full)
        self.dtype = torch.bfloat16 if bf16 else torch.float32
        self.lora = {n: Path(p) for n, p in dirs.items() if is_adapter_dir(p)}
        self.full = {n: Path(p) for n, p in dirs.items() if n not in self.lora}
        bad = [n for n, p in self.full.items() if not is_full_dir(p)]
        if bad:
            raise SystemExit(f"not a LoRA adapter or full model directory: {bad}")
        self.shared = load_policy(base, self.lora, device, bf16)
        self.loaded = {}  # name -> full model, least recently used first

    def __contains__(self, name):
        return name == "base" or name in self.lora or name in self.full

    def _full(self, name):
        model = self.loaded.pop(name, None)
        if model is None:
            while len(self.loaded) >= self.keep_full:
                del self.loaded[next(iter(self.loaded))]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(self.full[name], dtype=self.dtype).to(self.device).eval()
        self.loaded[name] = model
        return model

    @contextmanager
    def use(self, name):
        if name == "base" and self.lora:
            with self.shared.disable_adapter():
                yield self.shared
        elif name == "base":
            yield self.shared
        elif name in self.lora:
            self.shared.set_adapter(adapter_key(name))
            yield self.shared
        else:
            yield self._full(name)

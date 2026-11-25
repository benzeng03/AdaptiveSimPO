"""AdamW with fp32 master weights for full fine-tuning of a bf16 model.

AdamW applied directly to bf16 weights rounds each update to bf16's resolution (8 bits of
mantissa): at a weight of magnitude 0.03, any update below ~1e-4 is lost. Here the model keeps
bf16 weights for the forward and backward pass, while AdamW updates fp32 copies (and fp32
moments); after every step the fp32 masters are written back rounded to bf16, so small updates
accumulate in the masters instead of vanishing. The masters and moments take 12 bytes per
parameter and can be kept on the CPU (device="cpu") to save GPU memory.
"""
from __future__ import annotations

import torch


class MasterAdamW:
    def __init__(self, params, lr, weight_decay=0.0, device=None):
        self.params = list(params)
        self.device = None if device is None else torch.device(device)
        pin = self.device is not None and self.device.type == "cpu" and torch.cuda.is_available()
        self.master = []
        for p in self.params:
            m = p.detach().to(self.device or p.device, torch.float32, copy=True)
            self.master.append(m.pin_memory() if pin else m)
        # fused: one kernel over all masters (on an 8-vCPU VM, 0.24 s per step for 596M
        # parameters against 1.4 s for the default per-tensor loop)
        self.inner = torch.optim.AdamW(self.master, lr=lr, weight_decay=weight_decay, fused=True)
        self.param_groups = self.inner.param_groups

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for p, m in zip(self.params, self.master):
            m.grad = None if p.grad is None else p.grad.to(m.device, torch.float32)
            p.grad = None  # frees the bf16 gradient before the fp32 one of the next parameter
        self.inner.step()
        for p, m in zip(self.params, self.master):
            p.copy_(m)
            m.grad = None

    def state_dict(self):
        return {"inner": self.inner.state_dict(), "master": self.master}

    @torch.no_grad()
    def load_state_dict(self, state):
        for m, saved in zip(self.master, state["master"], strict=True):
            m.copy_(saved)
        self.inner.load_state_dict(state["inner"])
        for p, m in zip(self.params, self.master):
            p.copy_(m)


def make_optimizer(params, lr, mode="auto", weight_decay=0.0):
    """mode: auto (fp32 masters on the parameters' device if any parameter is below fp32),
    gpu (same, forced), cpu (masters and moments on the CPU) or off (plain AdamW)."""
    params = list(params)
    if mode == "auto":
        mode = "gpu" if any(p.dtype != torch.float32 for p in params) else "off"
    if mode == "off":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return MasterAdamW(params, lr, weight_decay, device="cpu" if mode == "cpu" else None)

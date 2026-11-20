"""Per-pair SimPO target margins.

All quantities are in beta-scaled reward units:

    r_i = beta * (mean_logp(y_w|x) - mean_logp(y_l|x)),   L_i = softplus(gamma_i - r_i).

r_i > 0 means the pair is ranked correctly. Margins are computed from the
detached reward gaps of the current batch, so dL_i/dr_i = -sigmoid(gamma_i - r_i)
and no gradient flows through gamma_i.

Rules
-----
fixed      gamma_i = gamma                                          (SimPO)
adaptive   gamma_i = gamma_max * erf( (-r_i)_+ / (sqrt(2) * s_B) )   (zero-threshold erf)

s_B is the standard deviation of the reward gaps in the current batch. Only
misranked pairs (r_i < 0) receive a positive margin; gamma_i is bounded by gamma_max.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

KINDS = ("fixed", "adaptive")


@dataclass(frozen=True)
class MarginConfig:
    kind: str = "fixed"
    gamma: float = 0.5          # fixed margin
    gamma_max: float = 2.0      # adaptive upper bound
    std_correction: int = 0     # 0: divisor B (population), 1: divisor B-1
    eps: float = 1e-6           # divisor used when the batch std is zero

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not all(math.isfinite(v) for v in (self.gamma, self.gamma_max, self.eps)):
            raise ValueError("margin parameters must be finite")
        if self.gamma < 0 or self.gamma_max < 0 or self.eps <= 0:
            raise ValueError("require gamma, gamma_max >= 0 and eps > 0")
        if self.std_correction not in (0, 1):
            raise ValueError("std_correction must be 0 or 1")

    @property
    def name(self) -> str:
        if self.kind == "fixed":
            return f"fixed-g{self.gamma:g}"
        return f"erf-zero-g{self.gamma_max:g}"

    def to_dict(self):
        return asdict(self)


def _adaptive(r: torch.Tensor, scale, gamma_max: float) -> torch.Tensor:
    return gamma_max * torch.erf((-r).clamp_min(0) / (math.sqrt(2.0) * scale))


@torch.no_grad()
def compute_margins(scores: torch.Tensor, cfg: MarginConfig):
    """Return (gamma, stats) for a 1-D tensor of reward gaps r_i."""
    r = scores.detach().double()
    if r.ndim != 1 or r.numel() == 0 or not torch.isfinite(r).all():
        raise ValueError("expected a finite, nonempty vector of reward gaps")
    correction = cfg.std_correction if r.numel() > 1 else 0
    std, mean = torch.std_mean(r, correction=correction)
    scale = torch.where(std > 0, std, torch.full_like(std, cfg.eps))
    if cfg.kind == "fixed":
        gamma = torch.full_like(r, cfg.gamma)
    else:
        gamma = _adaptive(r, scale, cfg.gamma_max)
    stats = dict(batch_mean=mean.item(), batch_std=std.item(),
                 gamma_mean=gamma.mean().item(), active=(gamma > 0).double().mean().item(),
                 misranked=(r < 0).double().mean().item())
    return gamma.to(scores.dtype if scores.is_floating_point() else torch.float32), stats


def simpo_loss(chosen_logps, rejected_logps, beta, cfg: MarginConfig):
    """Mean SimPO loss with per-pair detached margins.

    chosen_logps / rejected_logps are length-normalised (mean per-token) log
    probabilities. Returns (loss, scores, gamma, stats).
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if chosen_logps.shape != rejected_logps.shape:
        raise ValueError("expected paired chosen/rejected log probabilities")
    scores = beta * (chosen_logps.float() - rejected_logps.float())
    gamma, stats = compute_margins(scores, cfg)
    loss = F.softplus(gamma.detach() - scores).mean()
    return loss, scores.detach(), gamma.detach(), stats


@torch.no_grad()
def normal_reference(mu: float, sigma: float, cfg: MarginConfig, points: int = 20001):
    """Active fraction and mean margin expected if R ~ N(mu, sigma^2) and the batch
    statistics equal (mu, sigma). For the adaptive rule the active fraction is
    P(R < 0) = Phi(-mu / sigma)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = torch.linspace(-10, 10, points, dtype=torch.float64)
    weights = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi) * (z[1] - z[0])
    if cfg.kind == "fixed":
        g = torch.full_like(z, cfg.gamma)
    else:
        g = _adaptive(mu + sigma * z, sigma, cfg.gamma_max)
    return dict(active=float((weights * (g > 0)).sum()), gamma_mean=float((weights * g).sum()))

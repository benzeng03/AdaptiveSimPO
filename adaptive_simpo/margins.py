"""Per-pair SimPO target margins.

All quantities are in beta-scaled reward units:

    r_i = beta * (mean_logp(y+|x) - mean_logp(y-|x)),   L_i = softplus(gamma_i - r_i).

Margins are computed from the detached reward gaps of the current batch, so the
gradient w.r.t. r_i is -sigmoid(gamma_i - r_i) (no gradient flows through gamma_i).

Supported rules
---------------
fixed            gamma_i = gamma                                   (SimPO)
adaptive         gamma_i = gamma_max * f((t - r_i)_+ / s_B)         (this project)
                 t = 0 ("zero" threshold) or t = batch mean ("mean" threshold)
                 f(u) = erf(u / sqrt 2)  (percentile within the hard half-normal)
                      | tanh(u)          (earlier heuristic, mapping ablation)
amapo            gamma_i = beta * exp(z_i * mean(Delta)) for r_i < mean, else 0,
                 z_i = (mean(r) - r_i) / std(r), Delta = r / beta. This mirrors
                 the released AMaPO trainer (Deng et al., AAAI 2026), which uses
                 the unbiased std and no upper cap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
from torch.nn import functional as F

KINDS = ("fixed", "adaptive", "amapo")
THRESHOLDS = ("zero", "mean")
MAPPINGS = ("erf", "tanh")


@dataclass(frozen=True)
class MarginConfig:
    kind: str = "fixed"
    gamma: float = 0.5          # fixed margin
    gamma_max: float = 2.0      # adaptive upper bound
    threshold: str = "mean"     # adaptive: hard if r_i < t
    mapping: str = "erf"        # adaptive: difficulty -> fraction of gamma_max
    std_correction: int = 0     # 0: divisor B (population), 1: divisor B-1
    eps: float = 1e-6           # divisor used when the batch std is zero
    beta: float = 2.0           # only used by amapo

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.threshold not in THRESHOLDS:
            raise ValueError(f"threshold must be one of {THRESHOLDS}")
        if self.mapping not in MAPPINGS:
            raise ValueError(f"mapping must be one of {MAPPINGS}")
        values = (self.gamma, self.gamma_max, self.eps, self.beta)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("margin parameters must be finite")
        if self.gamma < 0 or self.gamma_max < 0 or self.eps <= 0 or self.beta <= 0:
            raise ValueError("require gamma, gamma_max >= 0 and eps, beta > 0")
        if self.std_correction not in (0, 1):
            raise ValueError("std_correction must be 0 or 1")

    @property
    def name(self) -> str:
        if self.kind == "fixed":
            return f"fixed-g{self.gamma:g}"
        if self.kind == "amapo":
            return "amapo"
        return f"{self.mapping}-{self.threshold}-g{self.gamma_max:g}"

    def to_dict(self):
        return asdict(self)


def _mapping(name: str, u: torch.Tensor) -> torch.Tensor:
    if name == "erf":
        return torch.erf(u / math.sqrt(2.0))
    return torch.tanh(u)


@torch.no_grad()
def compute_margins(scores: torch.Tensor, cfg: MarginConfig):
    """Return (gamma, stats) for a 1-D tensor of reward gaps r_i."""
    r = scores.detach().double()
    if r.ndim != 1 or r.numel() == 0 or not torch.isfinite(r).all():
        raise ValueError("expected a finite, nonempty vector of reward gaps")
    correction = 1 if cfg.kind == "amapo" else cfg.std_correction
    if r.numel() == 1:
        correction = 0
    std, mean = torch.std_mean(r, correction=correction)
    scale = torch.where(std > 0, std, torch.full_like(std, cfg.eps))

    if cfg.kind == "fixed":
        gamma = torch.full_like(r, cfg.gamma)
    elif cfg.kind == "adaptive":
        t = mean if cfg.threshold == "mean" else torch.zeros_like(mean)
        u = (t - r).clamp_min(0) / scale
        gamma = cfg.gamma_max * _mapping(cfg.mapping, u)
    else:  # amapo
        z = (mean - r) / scale
        gamma = torch.where(r < mean, cfg.beta * torch.exp(z * mean / cfg.beta),
                            torch.zeros_like(r))
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
    """Population active fraction and mean margin if R ~ N(mu, sigma^2) and the
    batch statistics equal (mu, sigma). For the mean-threshold erf rule this is
    (1/2, gamma_max/4) regardless of (mu, sigma)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = torch.linspace(-10, 10, points, dtype=torch.float64)
    density = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    weights = density * (z[1] - z[0])
    if cfg.kind == "fixed":
        g = torch.full_like(z, cfg.gamma)
    elif cfg.kind == "adaptive":
        t = mu if cfg.threshold == "mean" else 0.0
        g = cfg.gamma_max * _mapping(cfg.mapping, ((t - (mu + sigma * z)) / sigma).clamp_min(0))
    else:
        r = mu + sigma * z
        g = torch.where(r < mu, cfg.beta * torch.exp(((mu - r) / sigma) * mu / cfg.beta),
                        torch.zeros_like(r))
    return dict(active=float((weights * (g > 0)).sum()), gamma_mean=float((weights * g).sum()))

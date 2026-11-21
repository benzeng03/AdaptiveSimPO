"""Per-pair SimPO target margins: the fixed SimPO margin and the calibrated margin.

All quantities are in beta-scaled reward units:

    r_i = beta * (mean_logp(y_w|x) - mean_logp(y_l|x)),   L_i = softplus(gamma_i - r_i).

r_i > 0 means the pair is ranked correctly. Margins are computed from detached reward gaps,
so dL_i/dr_i = -sigmoid(gamma_i - r_i) and no gradient flows through gamma_i: a margin acts
only as a per-pair gradient weight w_i = sigmoid(gamma_i - r_i).

Rules
-----
fixed       gamma_i = gamma                       (SimPO)
calibrated  gamma_i = (1 - k) * r_i               (README, "Calibrated margins")

The calibrated margin maximises the calibrated likelihood sum_i log sigmoid(k r_i) with k held
fixed while the policy is updated. k is the slope of the logistic calibration curve
P(r > 0 | |r|) = sigmoid(k |r|). For normal gaps N(mu, sigma^2) this curve is exact with
k = 2 mu / sigma^2, which is estimated from the mean and (population) variance of the current
batch's gaps, with lower bound K_MIN (used when the batch mean is nonpositive or tiny).
With gamma_i = (1 - k) r_i the weight is
sigmoid(gamma_i - r_i) = sigmoid(-k r_i), the calibrated residual, and the forward loss is
-log sigmoid(k r_i). k = 1 means no temperature correction and the margin vanishes; k < 1
(over-confident gaps) gives margins that grow with the gap and negative margins for misranked pairs.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

KINDS = ("fixed", "calibrated")
K_MIN = 0.01                   # lower bound of the calibration slope


@dataclass(frozen=True)
class MarginConfig:
    kind: str = "fixed"
    gamma: float = 0.5           # fixed margin

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not math.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be finite and >= 0")

    @property
    def name(self) -> str:
        return f"fixed-g{self.gamma:g}" if self.kind == "fixed" else "calibrated"

    def to_dict(self):
        return asdict(self)


def calibration_slope(r: torch.Tensor, lo: float = K_MIN) -> float:
    """Normal-reference slope k = 2 mean / var of the gaps r (population variance), at least lo.
    A nonpositive mean gives lo, and so does a degenerate batch with zero variance."""
    r = r.detach().double().flatten()
    var, mean = torch.var_mean(r, correction=0)
    if mean.item() <= 0 or var.item() <= 0:
        return lo
    return max(2 * mean.item() / var.item(), lo)


@torch.no_grad()
def compute_margins(scores: torch.Tensor, cfg: MarginConfig):
    """Return (gamma, stats) for a 1-D tensor of reward gaps r_i (one batch).

    For the calibrated rule stats also report the slope ``calib_k`` fitted on this batch.
    """
    r = scores.detach().double()
    if r.ndim != 1 or r.numel() == 0 or not torch.isfinite(r).all():
        raise ValueError("expected a finite, nonempty vector of reward gaps")
    std, mean = torch.std_mean(r, correction=0)
    stats = dict(batch_mean=mean.item(), batch_std=std.item(), misranked=(r < 0).double().mean().item())
    if cfg.kind == "fixed":
        gamma = torch.full_like(r, cfg.gamma)
    else:
        k = calibration_slope(r)
        gamma = (1 - k) * r
        stats.update(calib_k=k)
    stats.update(gamma_mean=gamma.mean().item(), active=(gamma > 0).double().mean().item())
    return gamma.to(scores.dtype if scores.is_floating_point() else torch.float32), stats


def simpo_loss(chosen_logps, rejected_logps, beta, cfg: MarginConfig):
    """Mean SimPO loss with per-pair detached margins.

    chosen_logps / rejected_logps are length-normalised (mean per-token) log probabilities.
    Returns (loss, scores, gamma, stats). stats also reports the gradient weights
    w_i = sigmoid(gamma_i - r_i): their mean and the share of their sum that falls on misranked pairs.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if chosen_logps.shape != rejected_logps.shape:
        raise ValueError("expected paired chosen/rejected log probabilities")
    scores = beta * (chosen_logps.float() - rejected_logps.float())
    gamma, stats = compute_margins(scores, cfg)
    with torch.no_grad():
        w = torch.sigmoid(gamma - scores.detach())
        stats.update(weight_mean=w.mean().item(),
                     weight_share_misranked=(w * (scores.detach() < 0)).sum().item() / w.sum().item())
    loss = F.softplus(gamma.detach() - scores).mean()
    return loss, scores.detach(), gamma.detach(), stats

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
With k_ema = beta > 0 the mean and second moment are instead exponential moving averages over
micro-batches (MomentEMA, Adam-style bias correction, current batch included), so k varies less
from step to step and stays defined for micro-batches of one pair; k_ema = 0 is the per-batch rule.
With gamma_i = (1 - k) r_i the weight is
sigmoid(gamma_i - r_i) = sigmoid(-k r_i), the calibrated residual, and the forward loss is
-log sigmoid(k r_i). k = 1 means no temperature correction and the margin vanishes; k < 1
(over-confident gaps) gives margins that grow with the gap and negative margins for misranked pairs.

family = "logistic" replaces the normal reference by a logistic location model r = mu + s L
(L standard logistic), estimated from the same two moments: mu = mean, s = sqrt(3 var) / pi. Its
calibration logit log f(r)/f(-r) = 2 log cosh((r + mu)/2s) - 2 log cosh((r - mu)/2s) is odd, rises
with slope k = (2/s) tanh(mu/2s) at 0 and saturates at c = 2 mu / s, so confidently ranked pairs
agree with their labels with probability at most sigmoid(c). The margin is gamma_i = r_i - g(r_i),
the weight sigmoid(-g(r_i)) and the forward loss -log sigmoid(g(r_i)). When k would fall below
K_MIN (nonpositive mean, zero variance or a tiny slope) the rule falls back to the linear k = K_MIN.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

KINDS = ("fixed", "calibrated")
FAMILIES = ("normal", "logistic")  # calibrated: reference distribution of the gaps
K_MIN = 0.01                   # lower bound of the calibration slope


@dataclass(frozen=True)
class MarginConfig:
    kind: str = "fixed"
    gamma: float = 0.5           # fixed margin
    k_ema: float = 0.0           # calibrated: EMA decay of the gap moments (0 = current batch only)
    family: str = "normal"       # calibrated: normal (linear logit 2 mu r / var) or logistic (saturating)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not math.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be finite and >= 0")
        if not 0 <= self.k_ema < 1:
            raise ValueError("k_ema must be in [0, 1)")
        if self.k_ema and self.kind != "calibrated":
            raise ValueError("k_ema applies only to the calibrated margin")
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if self.family != "normal" and self.kind != "calibrated":
            raise ValueError("family applies only to the calibrated margin")

    @property
    def name(self) -> str:
        if self.kind == "fixed":
            return f"fixed-g{self.gamma:g}"
        name = "calibrated" + ("-logistic" if self.family == "logistic" else "")
        return name + (f"-ema{self.k_ema:g}" if self.k_ema else "")

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


def slope_from_moments(mean: float, second: float, lo: float = K_MIN) -> float:
    """k = 2 mean / (second - mean^2), at least lo; lo for a nonpositive mean or zero variance."""
    var = second - mean * mean
    if mean <= 0 or var <= 0:
        return lo
    return max(2 * mean / var, lo)


def logistic_location(mean: float, second: float):
    """(mu, s) of the logistic location model with this mean and second moment, or None when the
    variance is not positive."""
    var = second - mean * mean
    if var <= 0:
        return None
    return mean, math.sqrt(3 * var) / math.pi


def _logcosh(x: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(x, -x) - math.log(2)


def logistic_logit(r: torch.Tensor, mu: float, s: float) -> torch.Tensor:
    """log f(r)/f(-r) of the logistic location density: odd, slope (2/s) tanh(mu/2s) at 0, limit 2 mu/s."""
    return 2 * (_logcosh((r + mu) / (2 * s)) - _logcosh((r - mu) / (2 * s)))


class MomentEMA:
    """Exponential moving averages of the mean and second moment of the reward gaps.

    update(r) folds in one micro-batch (each micro-batch counts once, whatever its size);
    mean / second are bias-corrected by 1 - beta^t as in Adam. state_dict() is JSON-serialisable
    so a resumable checkpoint can carry it.
    """

    def __init__(self, beta: float):
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0, 1)")
        self.beta, self.m1, self.m2, self.count = float(beta), 0.0, 0.0, 0

    def update(self, r: torch.Tensor):
        r = r.detach().double().flatten()
        b = self.beta
        self.m1 = b * self.m1 + (1 - b) * r.mean().item()
        self.m2 = b * self.m2 + (1 - b) * (r * r).mean().item()
        self.count += 1

    def moments(self):
        if not self.count:
            raise RuntimeError("MomentEMA has no data yet")
        c = 1 - self.beta ** self.count
        return self.m1 / c, self.m2 / c

    def slope(self, lo: float = K_MIN) -> float:
        return slope_from_moments(*self.moments(), lo=lo)

    def state_dict(self):
        return dict(beta=self.beta, m1=self.m1, m2=self.m2, count=self.count)

    def load_state_dict(self, state):
        if float(state["beta"]) != self.beta:
            raise ValueError(f"EMA decay {state['beta']} in the state, {self.beta} configured")
        self.m1, self.m2, self.count = float(state["m1"]), float(state["m2"]), int(state["count"])


@torch.no_grad()
def compute_margins(scores: torch.Tensor, cfg: MarginConfig, ema: MomentEMA | None = None):
    """Return (gamma, stats) for a 1-D tensor of reward gaps r_i (one batch).

    For the calibrated rule stats also report the slope ``calib_k`` (and, for the logistic family,
    the ceiling ``calib_c``, None on the linear fallback): fitted on this batch, or,
    when cfg.k_ema > 0, from ``ema`` after this batch is folded in (stats then add ``ema_mean`` and
    ``ema_std``). ``ema`` is required exactly when cfg.k_ema > 0 and is updated in place.
    """
    if (ema is not None) != (cfg.kind == "calibrated" and cfg.k_ema > 0):
        raise ValueError("pass a MomentEMA exactly when the calibrated margin has k_ema > 0")
    if ema is not None and ema.beta != cfg.k_ema:
        raise ValueError("MomentEMA decay differs from cfg.k_ema")
    r = scores.detach().double()
    if r.ndim != 1 or r.numel() == 0 or not torch.isfinite(r).all():
        raise ValueError("expected a finite, nonempty vector of reward gaps")
    std, mean = torch.std_mean(r, correction=0)
    stats = dict(batch_mean=mean.item(), batch_std=std.item(), misranked=(r < 0).double().mean().item())
    if cfg.kind == "fixed":
        gamma = torch.full_like(r, cfg.gamma)
    else:
        if ema is None:
            m1, m2 = mean.item(), (r * r).mean().item()
        else:
            ema.update(r)
            m1, m2 = ema.moments()
            stats.update(ema_mean=m1, ema_std=max(m2 - m1 * m1, 0.0) ** 0.5)
        if cfg.family == "normal":
            k = calibration_slope(r) if ema is None else ema.slope()
            gamma = (1 - k) * r
        else:
            loc = logistic_location(m1, m2)
            k = 2 / loc[1] * math.tanh(loc[0] / (2 * loc[1])) if loc is not None else 0.0
            if loc is None or loc[0] <= 0 or k < K_MIN:
                k, c = K_MIN, None  # linear fallback: no ceiling
                gamma = (1 - k) * r
            else:
                c = 2 * loc[0] / loc[1]
                gamma = r - logistic_logit(r, *loc)
            stats.update(calib_c=c)
        stats.update(calib_k=k)
    stats.update(gamma_mean=gamma.mean().item(), active=(gamma > 0).double().mean().item())
    return gamma.to(scores.dtype if scores.is_floating_point() else torch.float32), stats


def margin_loss(scores: torch.Tensor, cfg: MarginConfig, ema: MomentEMA | None = None):
    """Mean softplus(gamma_i - r_i) for reward gaps r_i of any definition (adaptive_simpo.rewards),
    with detached margins gamma_i from compute_margins.

    Returns (loss, scores, gamma, stats); stats also reports the gradient weights
    w_i = sigmoid(gamma_i - r_i): their mean and the share of their sum that falls on misranked pairs.
    """
    if scores.ndim != 1:
        raise ValueError("expected a vector of reward gaps")
    scores = scores.float()
    gamma, stats = compute_margins(scores, cfg, ema)
    with torch.no_grad():
        w = torch.sigmoid(gamma - scores.detach())
        stats.update(weight_mean=w.mean().item(),
                     weight_share_misranked=(w * (scores.detach() < 0)).sum().item() / max(w.sum().item(), 1e-30))
    loss = F.softplus(gamma.detach() - scores).mean()
    return loss, scores.detach(), gamma.detach(), stats


def simpo_loss(chosen_logps, rejected_logps, beta, cfg: MarginConfig, ema: MomentEMA | None = None):
    """Mean SimPO loss with per-pair detached margins.

    chosen_logps / rejected_logps are length-normalised (mean per-token) log probabilities;
    ema is the running gap moments for a calibrated margin with k_ema > 0 (see compute_margins).
    Returns (loss, scores, gamma, stats) as margin_loss.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if chosen_logps.shape != rejected_logps.shape:
        raise ValueError("expected paired chosen/rejected log probabilities")
    return margin_loss(beta * (chosen_logps.float() - rejected_logps.float()), cfg, ema)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """Mean DPO loss (Rafailov et al., 2023) on summed log probabilities.

    r_i = beta * ((log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x))),
    L_i = softplus(-r_i): margin_loss with the zero fixed margin.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    shapes = {t.shape for t in (policy_chosen, policy_rejected, ref_chosen, ref_rejected)}
    if len(shapes) != 1:
        raise ValueError("expected paired policy/reference log probabilities")
    ref = (ref_chosen.detach().float() - ref_rejected.detach().float())
    scores = beta * (policy_chosen.float() - policy_rejected.float() - ref)
    return margin_loss(scores, MarginConfig("fixed", 0.0))

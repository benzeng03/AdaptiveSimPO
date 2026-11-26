"""Reward gaps r_i for the margin losses: which score of a pair the margin rule acts on.

Every definition gives a gap r_i = s(y_w) - s(y_l) (plus, for R-DPO, a per-pair offset) in the
units the sigmoid sees; the loss is then softplus(gamma_i - r_i) with a detached margin gamma_i
from margins.compute_margins (fixed or calibrated). With S(y) = sum_t log pi(y_t | x, y_<t),
|y| the number of answer tokens and S_ref the same under the frozen reference model:

simpo      beta * (S(y_w)/|y_w| - S(y_l)/|y_l|)                         (SimPO, length-normalised)
dpo        beta * ((S - S_ref)(y_w) - (S - S_ref)(y_l))                  (DPO)
dpo-noref  beta * (S(y_w) - S(y_l))                                      (DPO without the reference)
r-dpo      dpo + alpha * (|y_w| - |y_l|)                                 (R-DPO, Park et al. 2024)
orpo       beta * (logodds(y_w) - logodds(y_l)),  logodds = m - log(1 - e^m), m = S/|y|   (ORPO)

The R-DPO offset carries no gradient: it lowers the weight of pairs whose chosen answer is the
longer one. ORPO also adds the chosen answer's mean negative log-likelihood to the loss
(L = NLL(y_w) + lambda * L_pref, train.py --orpo-lambda); beta = 1 is ORPO's own odds-ratio term.
"""
from __future__ import annotations

import torch

REWARDS = ("simpo", "dpo", "dpo-noref", "r-dpo", "orpo")
NEEDS_REFERENCE = frozenset({"dpo", "r-dpo"})
DEFAULT_BETA = {"simpo": 2.0, "dpo": 0.1, "dpo-noref": 0.1, "r-dpo": 0.1, "orpo": 1.0}
LOG_ODDS_MAX = -1e-6  # mean log-probabilities are clamped below 0 so that log(1 - e^m) is finite


def log_odds(mean_logp: torch.Tensor) -> torch.Tensor:
    """log(p / (1 - p)) for p = exp(mean_logp), computed stably."""
    m = mean_logp.clamp(max=LOG_ODDS_MAX)
    return m - torch.log(-torch.expm1(m))


def reward_gap(reward, beta, chosen_sum, rejected_sum, chosen_len, rejected_len,
               ref_chosen_sum=None, ref_rejected_sum=None, alpha=0.0):
    """Per-pair reward gap r_i (float32, differentiable through the policy log-probabilities)."""
    if reward not in REWARDS:
        raise ValueError(f"reward must be one of {REWARDS}")
    if not beta > 0:
        raise ValueError("beta must be positive")
    if (ref_chosen_sum is None) != (reward not in NEEDS_REFERENCE) or (ref_rejected_sum is None) != (ref_chosen_sum is None):
        raise ValueError(f"reward {reward} {'needs' if reward in NEEDS_REFERENCE else 'takes no'} reference log-probabilities")
    cs, rs = chosen_sum.float(), rejected_sum.float()
    cl, rl = chosen_len.to(cs.dtype), rejected_len.to(cs.dtype)
    if reward == "simpo":
        return beta * (cs / cl - rs / rl)
    if reward == "dpo-noref":
        return beta * (cs - rs)
    if reward == "orpo":
        return beta * (log_odds(cs / cl) - log_odds(rs / rl))
    ref = ref_chosen_sum.detach().float() - ref_rejected_sum.detach().float()
    gap = beta * (cs - rs - ref)
    if reward == "r-dpo":
        gap = gap + alpha * (cl - rl)
    return gap

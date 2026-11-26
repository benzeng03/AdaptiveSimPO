import math
import unittest

import torch

from adaptive_simpo.margins import MarginConfig, margin_loss, simpo_loss
from adaptive_simpo.rewards import DEFAULT_BETA, REWARDS, log_odds, reward_gap

CS, RS = torch.tensor([-20.0, -6.0]), torch.tensor([-12.0, -9.0])   # summed log-probabilities
CL, RL = torch.tensor([10, 3]), torch.tensor([4, 6])                 # answer lengths
REF_CS, REF_RS = torch.tensor([-18.0, -7.0]), torch.tensor([-13.0, -8.0])
REF = dict(ref_chosen_sum=REF_CS, ref_rejected_sum=REF_RS)


class RewardGapTests(unittest.TestCase):
    def test_formulas(self):
        b = 0.5
        torch.testing.assert_close(reward_gap("simpo", b, CS, RS, CL, RL), b * (CS / CL - RS / RL))
        torch.testing.assert_close(reward_gap("dpo-noref", b, CS, RS, CL, RL), b * (CS - RS))
        dpo = b * ((CS - REF_CS) - (RS - REF_RS))
        torch.testing.assert_close(reward_gap("dpo", b, CS, RS, CL, RL, **REF), dpo)
        torch.testing.assert_close(reward_gap("r-dpo", b, CS, RS, CL, RL, alpha=0.1, **REF),
                                   dpo + 0.1 * (CL - RL).float())

    def test_orpo_log_odds(self):
        m_c, m_r = CS / CL, RS / RL
        direct = lambda m: torch.log(torch.exp(m) / (1 - torch.exp(m)))
        torch.testing.assert_close(reward_gap("orpo", 1.0, CS, RS, CL, RL), direct(m_c) - direct(m_r))
        self.assertTrue(torch.isfinite(log_odds(torch.tensor([0.0, -1e-9, -50.0]))).all())
        # log-odds is increasing in the mean log-probability, so the ranking matches SimPO's
        m = torch.linspace(-8, -0.01, 50)
        self.assertTrue((log_odds(m).diff() > 0).all())

    def test_reference_rules(self):
        for reward in ("dpo", "r-dpo"):
            with self.assertRaises(ValueError):
                reward_gap(reward, 0.1, CS, RS, CL, RL)
        for reward in ("simpo", "dpo-noref", "orpo"):
            with self.assertRaises(ValueError):
                reward_gap(reward, 0.1, CS, RS, CL, RL, **REF)
        with self.assertRaises(ValueError):
            reward_gap("ipo", 0.1, CS, RS, CL, RL)
        with self.assertRaises(ValueError):
            reward_gap("simpo", 0.0, CS, RS, CL, RL)
        self.assertEqual(set(DEFAULT_BETA), set(REWARDS))

    def test_gradients_flow_through_the_policy_only(self):
        cs, rs = CS.clone().requires_grad_(), RS.clone().requires_grad_()
        ref_cs = REF_CS.clone().requires_grad_()
        gap = reward_gap("r-dpo", 0.1, cs, rs, CL, RL, alpha=0.2, ref_chosen_sum=ref_cs, ref_rejected_sum=REF_RS)
        gap.sum().backward()
        torch.testing.assert_close(cs.grad, torch.full_like(CS, 0.1))
        torch.testing.assert_close(rs.grad, torch.full_like(RS, -0.1))
        self.assertIsNone(ref_cs.grad)

    def test_margin_loss_matches_simpo_loss(self):
        cfg = MarginConfig(kind="calibrated")
        mc, mr = CS / CL, RS / RL
        a = simpo_loss(mc, mr, 2.0, cfg)
        b = margin_loss(reward_gap("simpo", 2.0, CS, RS, CL, RL), cfg)
        self.assertAlmostEqual(a[0].item(), b[0].item(), places=6)
        self.assertEqual(a[3], b[3])

    def test_zero_dpo_gap_gives_log2(self):
        loss, *_ = margin_loss(reward_gap("dpo", 0.1, CS, RS, CL, RL, ref_chosen_sum=CS, ref_rejected_sum=RS),
                               MarginConfig(kind="fixed", gamma=0.0))
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)


if __name__ == "__main__":
    unittest.main()

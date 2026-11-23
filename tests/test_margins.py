import math
import unittest

import torch
from torch.nn import functional as F

from adaptive_simpo.margins import K_MIN, MarginConfig, calibration_slope, compute_margins, dpo_loss, simpo_loss

CALIBRATED = MarginConfig(kind="calibrated")


class FixedTests(unittest.TestCase):
    def test_fixed(self):
        g, s = compute_margins(torch.tensor([-2.0, 0.0, 0.0, 6.0]), MarginConfig(kind="fixed", gamma=0.75))
        torch.testing.assert_close(g, torch.full((4,), 0.75))
        self.assertEqual(s["active"], 1.0)
        self.assertAlmostEqual(s["batch_mean"], 1.0)
        self.assertAlmostEqual(s["batch_std"], 3.0)
        self.assertAlmostEqual(s["misranked"], 0.25)
        self.assertNotIn("calib_k", s)


class CalibrationSlopeTests(unittest.TestCase):
    def test_two_mean_over_var(self):
        r = torch.tensor([-2.0, 0.0, 0.0, 6.0])          # mean 1, population variance 9
        self.assertAlmostEqual(calibration_slope(r), 2 / 9)

    def test_normal_gaps_give_k_two_mu_over_var(self):
        torch.manual_seed(0)
        for mu, sd in ((0.5, 2.0), (0.8, 1.6), (1.5, math.sqrt(3.0))):
            r = mu + sd * torch.randn(200000, dtype=torch.float64)
            self.assertAlmostEqual(calibration_slope(r), 2 * mu / sd ** 2, delta=0.01, msg=(mu, sd))

    def test_normal_gaps_balance_the_weights(self):
        # For normal gaps sigmoid(k r) with k = 2 mu / sigma^2 is the exact calibration curve, so the
        # weighted mean gap is zero and half of the weight falls on misranked pairs.
        torch.manual_seed(1)
        r = 0.5 + 2.3 * torch.randn(400000, dtype=torch.float64)
        g, s = compute_margins(r, CALIBRATED)
        w = torch.sigmoid(g.double() - r)
        self.assertLess(abs((w * r).sum().item()) / w.sum().item(), 0.01)
        self.assertAlmostEqual((w * (r < 0)).sum().item() / w.sum().item(), 0.5, delta=0.01)

    def test_calibrated_gaps_need_no_margin(self):
        torch.manual_seed(2)
        r = 1.5 + math.sqrt(3.0) * torch.randn(200000, dtype=torch.float64)   # mu = sigma^2 / 2
        g, s = compute_margins(r, CALIBRATED)
        self.assertAlmostEqual(s["calib_k"], 1.0, delta=0.02)
        self.assertLess(g.abs().mean().item(), 0.05)

    def test_bounds(self):
        self.assertEqual(calibration_slope(torch.tensor([-1.0, -2.0, 0.5])), K_MIN)    # mean <= 0
        self.assertEqual(calibration_slope(torch.zeros(4)), K_MIN)
        self.assertEqual(calibration_slope(torch.full((4,), 2.0)), K_MIN)              # zero variance
        self.assertEqual(calibration_slope(torch.tensor([0.01, -3.0, 3.02])), K_MIN)    # tiny slope -> lower bound
        r = torch.tensor([10.0, 10.1, 9.9])                                            # no upper bound
        self.assertAlmostEqual(calibration_slope(r), 2 * r.double().mean().item() / r.double().var(correction=0).item())

    def test_uses_the_current_batch_only(self):
        compute_margins(torch.tensor([-9.0, 9.0, 8.0, -7.0]), CALIBRATED)
        b = torch.tensor([-1.0, 2.0, 0.5, 3.0])
        _, s = compute_margins(b, CALIBRATED)
        self.assertAlmostEqual(s["calib_k"], calibration_slope(b), places=12)


class CalibratedMarginTests(unittest.TestCase):
    def test_margin_is_linear_and_tempers_the_weights(self):
        r = torch.tensor([-3.0, -1.0, 0.5, 1.0, 2.0, 4.0])
        g, s = compute_margins(r, CALIBRATED)
        k = s["calib_k"]
        self.assertAlmostEqual(k, calibration_slope(r))
        torch.testing.assert_close(g, (1 - k) * r)
        torch.testing.assert_close(torch.sigmoid(g - r), torch.sigmoid(-k * r))

    def test_finite_on_hard_batches(self):
        for r in (torch.tensor([-3.0]), torch.zeros(8), torch.full((8,), -2.0), torch.randn(64) * 30):
            g, s = compute_margins(r, CALIBRATED)
            self.assertTrue(torch.isfinite(g).all())
            self.assertTrue(K_MIN <= s["calib_k"] < float("inf"))

    def test_forward_loss_is_calibrated_likelihood(self):
        # With gamma = (1 - k) r the forward loss is -log sigmoid(k r).
        chosen = torch.tensor([-1.0, 0.0, 0.5, 3.0])
        loss, scores, _, s = simpo_loss(chosen, torch.zeros(4), 2.0, CALIBRATED)
        torch.testing.assert_close(loss, F.softplus(-s["calib_k"] * scores).mean())

    def test_loss_gradient(self):
        chosen = torch.tensor([-1.0, 0.0, 0.5, 3.0], requires_grad=True)
        loss, scores, gamma, s = simpo_loss(chosen, torch.zeros(4), 2.0, CALIBRATED)
        loss.backward()
        w = torch.sigmoid(-s["calib_k"] * scores)
        torch.testing.assert_close(chosen.grad, -2 * w / 4)


class ConfigTests(unittest.TestCase):
    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            compute_margins(torch.tensor([float("nan")]), MarginConfig())
        for kw in (dict(kind="percentile"), dict(gamma=-1.0), dict(gamma=float("inf"))):
            with self.assertRaises(ValueError):
                MarginConfig(**kw)
        with self.assertRaises(ValueError):
            simpo_loss(torch.zeros(2), torch.zeros(2), 0.0, MarginConfig())

    def test_names(self):
        self.assertEqual(MarginConfig(kind="fixed", gamma=0.75).name, "fixed-g0.75")
        self.assertEqual(MarginConfig(kind="fixed", gamma=2).name, "fixed-g2")
        self.assertEqual(CALIBRATED.name, "calibrated")


class LossTests(unittest.TestCase):
    def test_loss_and_stopgrad_gradient(self):
        chosen = torch.tensor([-1.0, 0.0, 0.0, 3.0], requires_grad=True)
        loss, scores, gamma, s = simpo_loss(chosen, torch.zeros(4), 2.0, MarginConfig(kind="fixed", gamma=1.0))
        torch.testing.assert_close(loss, F.softplus(gamma - scores).mean())
        loss.backward()
        # dL/dchosen_i = beta * (-sigmoid(gamma_i - r_i)) / B; gamma contributes no gradient.
        w = torch.sigmoid(gamma - scores)
        torch.testing.assert_close(chosen.grad, -2 * w / 4)
        self.assertAlmostEqual(s["weight_mean"], w.mean().item(), places=6)
        self.assertAlmostEqual(s["weight_share_misranked"], (w[0] / w.sum()).item(), places=6)

    def test_zero_margin_still_trains(self):
        chosen = torch.tensor([5.0, 6.0], requires_grad=True)
        loss, _, gamma, _ = simpo_loss(chosen, torch.zeros(2), 1.0, MarginConfig(kind="fixed", gamma=0.0))
        loss.backward()
        self.assertEqual(gamma.tolist(), [0.0, 0.0])
        self.assertTrue((chosen.grad < 0).all())


class DPOTests(unittest.TestCase):
    def test_loss_and_gradient(self):
        pc = torch.tensor([-10.0, -20.0, -5.0], requires_grad=True)
        pr = torch.tensor([-12.0, -15.0, -5.0], requires_grad=True)
        rc, rr = torch.tensor([-11.0, -19.0, -6.0]), torch.tensor([-11.0, -16.0, -4.0])
        loss, scores, gamma, s = dpo_loss(pc, pr, rc, rr, 0.1)
        r = 0.1 * ((pc - rc) - (pr - rr)).detach()
        torch.testing.assert_close(scores, r)
        torch.testing.assert_close(loss, F.softplus(-r).mean())
        self.assertEqual(gamma.tolist(), [0.0, 0.0, 0.0])
        loss.backward()
        w = torch.sigmoid(-r)
        torch.testing.assert_close(pc.grad, -0.1 * w / 3)
        torch.testing.assert_close(pr.grad, 0.1 * w / 3)
        self.assertAlmostEqual(s["weight_share_misranked"], (w * (r < 0)).sum().item() / w.sum().item(), places=6)

    def test_policy_equal_to_reference_gives_log2(self):
        x = torch.tensor([-3.0, -7.0])
        loss, _, _, _ = dpo_loss(x, x + 1, x, x + 1, 0.5)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            dpo_loss(torch.zeros(2), torch.zeros(2), torch.zeros(3), torch.zeros(2), 0.1)
        with self.assertRaises(ValueError):
            dpo_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2), 0.0)


if __name__ == "__main__":
    unittest.main()

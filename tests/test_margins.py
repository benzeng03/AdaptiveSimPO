import math
import unittest

import torch
from torch.nn import functional as F

from adaptive_simpo.margins import (K_MIN, MarginConfig, MomentEMA, calibration_slope, compute_margins, dpo_loss,
                                    logistic_location, logistic_logit,
                                    simpo_loss, slope_from_moments)

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


class LogisticFamilyTests(unittest.TestCase):
    CFG = MarginConfig(kind="calibrated", family="logistic")

    def test_logit_shape(self):
        mu, s = 1.5, 2.0
        r = torch.linspace(-30, 30, 2001, dtype=torch.float64)
        g = logistic_logit(r, mu, s)
        torch.testing.assert_close(g, -logistic_logit(-r, mu, s))           # odd
        slope = (logistic_logit(torch.tensor([1e-4], dtype=torch.float64), mu, s) / 1e-4).item()
        self.assertAlmostEqual(slope, 2 / s * math.tanh(mu / (2 * s)), places=5)
        self.assertAlmostEqual(g[-1].item(), 2 * mu / s, places=4)          # ceiling c = 2 mu / s
        self.assertTrue((g.diff() >= 0).all())                              # monotone
        # it is log f(r)/f(-r) of the logistic location density
        logf = lambda x: -(x - mu) / s - 2 * F.softplus(-(x - mu) / s)
        torch.testing.assert_close(g, logf(r) - logf(-r))

    def test_moments(self):
        mu, s = logistic_location(2.0, 2.0 ** 2 + 9.0)
        self.assertEqual(mu, 2.0)
        self.assertAlmostEqual(s, math.sqrt(27) / math.pi)
        self.assertIsNone(logistic_location(1.0, 1.0))

    def test_margin_weight_and_loss(self):
        r = torch.tensor([3.0, -1.0, 0.5, 8.0, 2.0, -4.0], dtype=torch.float64)
        gamma, stats = compute_margins(r, self.CFG)
        var, mean = torch.var_mean(r, correction=0)
        mu, s = mean.item(), math.sqrt(3 * var.item()) / math.pi
        g = logistic_logit(r, mu, s)
        torch.testing.assert_close(gamma, r - g)
        torch.testing.assert_close(torch.sigmoid(gamma - r), torch.sigmoid(-g))   # weight sigma(-g)
        torch.testing.assert_close(F.softplus(gamma - r), -F.logsigmoid(g))       # forward loss
        self.assertAlmostEqual(stats["calib_c"], 2 * mu / s)
        self.assertAlmostEqual(stats["calib_k"], 2 / s * math.tanh(mu / (2 * s)))
        # near 0 the logistic slope is pi^2/6 times the normal one when mu / s is small
        r2 = torch.randn(4000, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 3 + 0.2
        _, st2 = compute_margins(r2, self.CFG)
        self.assertAlmostEqual(st2["calib_k"] / calibration_slope(r2), math.pi ** 2 / 6, delta=0.02)

    def test_fallback(self):
        for r in (torch.tensor([-1.0, 0.5, -2.0]), torch.tensor([2.0, 2.0])):   # mean <= 0, zero variance
            gamma, stats = compute_margins(r, self.CFG)
            torch.testing.assert_close(gamma, (1 - K_MIN) * r)
            self.assertEqual(stats["calib_k"], K_MIN)
            self.assertIsNone(stats["calib_c"])

    def test_scale_free(self):
        r = torch.tensor([3.0, -1.0, 0.5, 8.0, 2.0, -4.0], dtype=torch.float64)
        g1 = r - compute_margins(r, self.CFG)[0]
        g5 = 5 * r - compute_margins(5 * r, self.CFG)[0]
        torch.testing.assert_close(g1, g5)

    def test_with_ema(self):
        cfg = MarginConfig(kind="calibrated", family="logistic", k_ema=0.9)
        ema = MomentEMA(0.9)
        for value in (2.0, -1.0, 3.0, 1.0, 4.0):
            gamma, stats = compute_margins(torch.tensor([value]), cfg, ema)
        m1, m2 = ema.moments()
        mu, s = logistic_location(m1, m2)
        torch.testing.assert_close(gamma, torch.tensor([4.0]) - logistic_logit(torch.tensor([4.0], dtype=torch.float64), mu, s).float())
        self.assertAlmostEqual(stats["calib_c"], 2 * mu / s)


class MomentEMATests(unittest.TestCase):
    def test_one_batch_equals_the_batch_slope(self):
        r = torch.tensor([2.0, -1.0, 3.0, 0.5])
        for beta in (0.0, 0.5, 0.99):  # bias correction makes the first estimate the batch itself
            ema = MomentEMA(beta)
            ema.update(r)
            self.assertAlmostEqual(ema.slope(), calibration_slope(r), places=12)

    def test_beta_zero_is_the_per_batch_rule(self):
        torch.manual_seed(0)
        ema = MomentEMA(0.0)
        for _ in range(5):
            r = torch.randn(16) * 2 + 0.5
            ema.update(r)
            self.assertAlmostEqual(ema.slope(), calibration_slope(r), places=12)

    def test_bias_corrected_average(self):
        a, b = torch.tensor([1.0, 3.0]), torch.tensor([0.0, 4.0])
        ema = MomentEMA(0.5)
        ema.update(a)
        ema.update(b)
        # weights (1 - beta) beta and (1 - beta), normalised by 1 - beta^2: 1/3 and 2/3
        m1, m2 = ema.moments()
        self.assertAlmostEqual(m1, (2.0 + 2 * 2.0) / 3)
        self.assertAlmostEqual(m2, (5.0 + 2 * 8.0) / 3)
        self.assertAlmostEqual(ema.slope(), slope_from_moments(m1, m2))
        self.assertAlmostEqual(ema.slope(), 2 * 2.0 / (7.0 - 4.0))

    def test_single_pair_batches(self):
        # one pair per micro-batch: the batch variance is 0, but the moving average is not
        cfg = MarginConfig(kind="calibrated", k_ema=0.9)
        ema = MomentEMA(0.9)
        ks = []
        for value in (2.0, -1.0, 3.0, 1.0):
            gamma, stats = compute_margins(torch.tensor([value]), cfg, ema)
            ks.append(stats["calib_k"])
            self.assertAlmostEqual(gamma.item(), (1 - stats["calib_k"]) * value, places=5)
            self.assertIn("ema_std", stats)
        self.assertEqual(ks[0], K_MIN)
        self.assertGreater(ks[-1], K_MIN)
        self.assertEqual(calibration_slope(torch.tensor([3.0])), K_MIN)

    def test_state_round_trip(self):
        ema = MomentEMA(0.8)
        for r in (torch.tensor([1.0, 2.0]), torch.tensor([-1.0, 4.0])):
            ema.update(r)
        other = MomentEMA(0.8)
        other.load_state_dict(ema.state_dict())
        self.assertEqual(other.moments(), ema.moments())
        with self.assertRaises(ValueError):
            MomentEMA(0.9).load_state_dict(ema.state_dict())
        with self.assertRaises(RuntimeError):
            MomentEMA(0.8).slope()

    def test_ema_must_match_the_config(self):
        r = torch.tensor([1.0, 2.0])
        with self.assertRaises(ValueError):
            compute_margins(r, MarginConfig(kind="calibrated", k_ema=0.9))
        with self.assertRaises(ValueError):
            compute_margins(r, CALIBRATED, MomentEMA(0.9))
        with self.assertRaises(ValueError):
            compute_margins(r, MarginConfig(kind="calibrated", k_ema=0.9), MomentEMA(0.5))


class ConfigTests(unittest.TestCase):
    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            compute_margins(torch.tensor([float("nan")]), MarginConfig())
        for kw in (dict(kind="percentile"), dict(gamma=-1.0), dict(gamma=float("inf")),
                   dict(kind="calibrated", k_ema=1.0), dict(kind="calibrated", k_ema=-0.1),
                   dict(kind="fixed", k_ema=0.9), dict(kind="fixed", family="logistic"),
                   dict(kind="calibrated", family="t")):
            with self.assertRaises(ValueError):
                MarginConfig(**kw)
        with self.assertRaises(ValueError):
            simpo_loss(torch.zeros(2), torch.zeros(2), 0.0, MarginConfig())

    def test_names(self):
        self.assertEqual(MarginConfig(kind="fixed", gamma=0.75).name, "fixed-g0.75")
        self.assertEqual(MarginConfig(kind="fixed", gamma=2).name, "fixed-g2")
        self.assertEqual(CALIBRATED.name, "calibrated")
        self.assertEqual(MarginConfig(kind="calibrated", k_ema=0.9).name, "calibrated-ema0.9")
        self.assertEqual(MarginConfig(kind="calibrated", family="logistic").name, "calibrated-logistic")
        self.assertEqual(MarginConfig(kind="calibrated", family="logistic", k_ema=0.9).name,
                         "calibrated-logistic-ema0.9")


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

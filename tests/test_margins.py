import math
import unittest

import torch
from torch.nn import functional as F

from adaptive_simpo.margins import MarginConfig, compute_margins, normal_reference, simpo_loss


def erf(x):
    return torch.erf(torch.as_tensor(x, dtype=torch.float64) / math.sqrt(2))


class MarginRuleTests(unittest.TestCase):
    # Reward gaps [-2, 0, 0, 6]: mean 1, population std 3.
    r = torch.tensor([-2.0, 0.0, 0.0, 6.0])

    def test_fixed(self):
        g, s = compute_margins(self.r, MarginConfig(kind="fixed", gamma=0.7))
        torch.testing.assert_close(g, torch.full((4,), 0.7))
        self.assertEqual(s["active"], 1.0)

    def test_zero_threshold_erf(self):
        g, s = compute_margins(self.r, MarginConfig(kind="adaptive", threshold="zero", gamma_max=2))
        expected = 2 * erf(torch.tensor([2 / 3, 0, 0, 0])).float()
        torch.testing.assert_close(g, expected)
        self.assertAlmostEqual(s["batch_mean"], 1.0)
        self.assertAlmostEqual(s["batch_std"], 3.0)
        self.assertAlmostEqual(s["active"], 0.25)

    def test_mean_threshold_erf_covers_correct_but_small_gaps(self):
        g, s = compute_margins(self.r, MarginConfig(kind="adaptive", threshold="mean", gamma_max=2))
        expected = 2 * erf(torch.tensor([1.0, 1 / 3, 1 / 3, 0])).float()
        torch.testing.assert_close(g, expected)
        self.assertAlmostEqual(s["active"], 0.75)  # r=0 pairs are below the mean

    def test_one_sigma_gets_68_percent(self):
        r = torch.tensor([-1.0, 1.0])  # mean 0, population std 1
        g, _ = compute_margins(r, MarginConfig(kind="adaptive", threshold="mean", gamma_max=1))
        self.assertAlmostEqual(g[0].item(), 0.6827, places=4)
        self.assertEqual(g[1].item(), 0.0)

    def test_tanh_mapping(self):
        g, _ = compute_margins(self.r, MarginConfig(kind="adaptive", threshold="mean",
                                                    mapping="tanh", gamma_max=1))
        torch.testing.assert_close(g, torch.tanh(torch.tensor([1.0, 1 / 3, 1 / 3, 0])))

    def test_negative_mean_direction(self):
        r = torch.tensor([-5.0, -3.0, -1.0])  # mean -3: only r=-5 is harder than average
        g, _ = compute_margins(r, MarginConfig(kind="adaptive", threshold="mean"))
        self.assertGreater(g[0].item(), 0)
        self.assertEqual(g[1:].tolist(), [0.0, 0.0])
        g0, _ = compute_margins(r, MarginConfig(kind="adaptive", threshold="zero"))
        self.assertTrue((g0 > 0).all())

    def test_amapo_matches_released_formula(self):
        beta = 2.0
        logratio = self.r / beta
        g, _ = compute_margins(self.r, MarginConfig(kind="amapo", beta=beta))
        mean, std = logratio.mean(), logratio.std()  # unbiased, as in amapo_trainer.py
        ref = beta * torch.where(logratio < mean, torch.exp((mean - logratio) / std * mean), 0)
        torch.testing.assert_close(g, ref)

    def test_constant_and_singleton_batches_are_finite(self):
        for r in (torch.tensor([-3.0]), torch.zeros(8), torch.full((8,), 2.0)):
            for cfg in (MarginConfig(kind="adaptive", threshold="zero"),
                        MarginConfig(kind="adaptive", threshold="mean"), MarginConfig(kind="amapo")):
                g, _ = compute_margins(r, cfg)
                self.assertTrue(torch.isfinite(g).all())
                self.assertTrue((g >= 0).all() and (g <= max(cfg.gamma_max, 1e9)).all())

    def test_bounded_by_gamma_max(self):
        r = torch.randn(1000) * 5
        g, _ = compute_margins(r, MarginConfig(kind="adaptive", gamma_max=1.5))
        self.assertTrue((g >= 0).all() and (g <= 1.5).all())

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            compute_margins(torch.tensor([float("nan")]), MarginConfig())
        with self.assertRaises(ValueError):
            MarginConfig(kind="adaptive", threshold="median")


class LossTests(unittest.TestCase):
    def test_loss_and_stopgrad_gradient(self):
        chosen = torch.tensor([-1.0, 0.0, 0.0, 3.0], requires_grad=True)
        cfg = MarginConfig(kind="adaptive", threshold="mean", gamma_max=2)
        loss, scores, gamma, _ = simpo_loss(chosen, torch.zeros(4), 2.0, cfg)
        torch.testing.assert_close(loss, F.softplus(gamma - scores).mean())
        loss.backward()
        # dL/dchosen_i = beta * (-sigmoid(gamma_i - r_i)) / B; gamma contributes no gradient.
        torch.testing.assert_close(chosen.grad, -2 * torch.sigmoid(gamma - scores) / 4)

    def test_zero_margin_still_trains(self):
        chosen = torch.tensor([5.0, 6.0], requires_grad=True)
        loss, _, gamma, _ = simpo_loss(chosen, torch.zeros(2), 1.0, MarginConfig(kind="fixed", gamma=0))
        loss.backward()
        self.assertEqual(gamma.tolist(), [0.0, 0.0])
        self.assertTrue((chosen.grad < 0).all())


class NormalReferenceTests(unittest.TestCase):
    def test_mean_threshold_erf_reference_is_quarter(self):
        cfg = MarginConfig(kind="adaptive", threshold="mean", gamma_max=0.8)
        for mu, sigma in ((0.0, 1.0), (1.3, 0.4), (-2.0, 3.0)):
            ref = normal_reference(mu, sigma, cfg)
            self.assertAlmostEqual(ref["active"], 0.5, places=3)
            self.assertAlmostEqual(ref["gamma_mean"], 0.2, places=4)

    def test_zero_threshold_active_fraction(self):
        cfg = MarginConfig(kind="adaptive", threshold="zero")
        ref = normal_reference(1.0, 2.0, cfg)
        phi = 0.5 * (1 + math.erf(-0.5 / math.sqrt(2)))  # P(R < 0)
        self.assertAlmostEqual(ref["active"], phi, places=3)

    def test_empirical_matches_reference_for_large_batch(self):
        torch.manual_seed(0)
        r = 0.7 + 1.5 * torch.randn(200000, dtype=torch.float64)
        cfg = MarginConfig(kind="adaptive", threshold="mean", gamma_max=2)
        g, s = compute_margins(r, cfg)
        self.assertAlmostEqual(s["gamma_mean"], 0.5, places=2)
        self.assertAlmostEqual(s["active"], 0.5, places=2)


if __name__ == "__main__":
    unittest.main()

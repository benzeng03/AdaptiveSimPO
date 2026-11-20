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
        g, s = compute_margins(self.r, MarginConfig(kind="fixed", gamma=0.75))
        torch.testing.assert_close(g, torch.full((4,), 0.75))
        self.assertEqual(s["active"], 1.0)

    def test_zero_threshold_erf(self):
        g, s = compute_margins(self.r, MarginConfig(kind="adaptive", gamma_max=2))
        expected = 2 * erf(torch.tensor([2 / 3, 0, 0, 0])).float()
        torch.testing.assert_close(g, expected)
        self.assertAlmostEqual(s["batch_mean"], 1.0)
        self.assertAlmostEqual(s["batch_std"], 3.0)
        self.assertAlmostEqual(s["active"], 0.25)
        self.assertAlmostEqual(s["misranked"], 0.25)

    def test_only_misranked_pairs_get_margin(self):
        r = torch.tensor([-5.0, -3.0, -1.0, 0.5, 4.0])
        g, _ = compute_margins(r, MarginConfig(kind="adaptive"))
        self.assertTrue((g[:3] > 0).all())
        self.assertEqual(g[3:].tolist(), [0.0, 0.0])
        self.assertTrue((g[0] > g[1]) and (g[1] > g[2]))  # harder pairs get more

    def test_one_sigma_gets_68_percent(self):
        r = torch.tensor([-1.0, 1.0])  # population std 1, r=-1 is one std below zero
        g, _ = compute_margins(r, MarginConfig(kind="adaptive", gamma_max=1))
        self.assertAlmostEqual(g[0].item(), 0.6827, places=4)
        self.assertEqual(g[1].item(), 0.0)

    def test_std_correction(self):
        g, s = compute_margins(self.r, MarginConfig(kind="adaptive", std_correction=1))
        std = math.sqrt(36 / 3)
        self.assertAlmostEqual(s["batch_std"], std, places=5)
        self.assertAlmostEqual(g[0].item(), 2 * math.erf(2 / std / math.sqrt(2)), places=5)

    def test_constant_and_singleton_batches_are_finite(self):
        for r in (torch.tensor([-3.0]), torch.zeros(8), torch.full((8,), -2.0)):
            g, _ = compute_margins(r, MarginConfig(kind="adaptive"))
            self.assertTrue(torch.isfinite(g).all())
            self.assertTrue((g >= 0).all() and (g <= 2).all())

    def test_bounded_by_gamma_max(self):
        r = torch.randn(1000) * 5
        g, _ = compute_margins(r, MarginConfig(kind="adaptive", gamma_max=1.5))
        self.assertTrue((g >= 0).all() and (g <= 1.5).all())

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            compute_margins(torch.tensor([float("nan")]), MarginConfig())
        for kind in ("amapo", "tanh"):
            with self.assertRaises(ValueError):
                MarginConfig(kind=kind)

    def test_names(self):
        self.assertEqual(MarginConfig(kind="fixed", gamma=0.75).name, "fixed-g0.75")
        self.assertEqual(MarginConfig(kind="adaptive", gamma_max=2).name, "erf-zero-g2")


class LossTests(unittest.TestCase):
    def test_loss_and_stopgrad_gradient(self):
        chosen = torch.tensor([-1.0, 0.0, 0.0, 3.0], requires_grad=True)
        cfg = MarginConfig(kind="adaptive", gamma_max=2)
        loss, scores, gamma, _ = simpo_loss(chosen, torch.zeros(4), 2.0, cfg)
        torch.testing.assert_close(loss, F.softplus(gamma - scores).mean())
        loss.backward()
        # dL/dchosen_i = beta * (-sigmoid(gamma_i - r_i)) / B; gamma contributes no gradient.
        torch.testing.assert_close(chosen.grad, -2 * torch.sigmoid(gamma - scores) / 4)

    def test_zero_margin_still_trains(self):
        chosen = torch.tensor([5.0, 6.0], requires_grad=True)
        loss, _, gamma, _ = simpo_loss(chosen, torch.zeros(2), 1.0, MarginConfig(kind="adaptive"))
        loss.backward()
        self.assertEqual(gamma.tolist(), [0.0, 0.0])
        self.assertTrue((chosen.grad < 0).all())


class NormalReferenceTests(unittest.TestCase):
    def test_active_fraction_is_prob_misranked(self):
        cfg = MarginConfig(kind="adaptive")
        for mu, sigma in ((1.0, 2.0), (0.0, 1.0), (-0.5, 0.7)):
            ref = normal_reference(mu, sigma, cfg)
            phi = 0.5 * (1 + math.erf(-mu / sigma / math.sqrt(2)))  # P(R < 0)
            self.assertAlmostEqual(ref["active"], phi, places=3)

    def test_zero_mean_reduces_to_quarter(self):
        # With mu = 0 the zero threshold equals the mean: half-normal percentile, E[gamma] = gamma_max / 4.
        ref = normal_reference(0.0, 1.3, MarginConfig(kind="adaptive", gamma_max=0.8))
        self.assertAlmostEqual(ref["gamma_mean"], 0.2, places=4)

    def test_empirical_matches_reference_for_large_batch(self):
        torch.manual_seed(0)
        mu, sigma = 0.7, 1.5
        r = mu + sigma * torch.randn(200000, dtype=torch.float64)
        cfg = MarginConfig(kind="adaptive", gamma_max=2)
        _, s = compute_margins(r, cfg)
        ref = normal_reference(mu, sigma, cfg)
        self.assertAlmostEqual(s["gamma_mean"], ref["gamma_mean"], places=2)
        self.assertAlmostEqual(s["active"], ref["active"], places=2)

    def test_fixed_reference(self):
        ref = normal_reference(0.3, 1.0, MarginConfig(kind="fixed", gamma=0.75))
        self.assertAlmostEqual(ref["gamma_mean"], 0.75, places=4)
        self.assertAlmostEqual(ref["active"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

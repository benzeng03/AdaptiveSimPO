"""fp32 master weights keep updates that plain AdamW on bf16 weights rounds away."""
import unittest

import torch

from adaptive_simpo.optim import MasterAdamW, make_optimizer


def run(opt_factory, steps=100, lr=1e-4):
    p = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    opt = opt_factory([p], lr)
    for _ in range(steps):
        p.grad = torch.ones_like(p)  # Adam step ~= lr per step, far below bf16's spacing at 1.0 (2^-7)
        opt.step()
        opt.zero_grad()
    return p, opt


class MasterAdamWTests(unittest.TestCase):
    def test_plain_bf16_adamw_loses_small_updates(self):
        p, _ = run(lambda ps, lr: torch.optim.AdamW(ps, lr=lr, weight_decay=0.0))
        self.assertTrue(torch.all(p == 1))

    def test_masters_accumulate_small_updates(self):
        p, opt = run(lambda ps, lr: MasterAdamW(ps, lr))
        self.assertAlmostEqual(opt.master[0][0].item(), 1 - 100 * 1e-4, places=4)
        self.assertEqual(p.dtype, torch.bfloat16)
        torch.testing.assert_close(p.float(), opt.master[0].to(torch.bfloat16).float())
        self.assertTrue(torch.all(p < 1))

    def test_cpu_masters_and_state_round_trip(self):
        p, opt = run(lambda ps, lr: MasterAdamW(ps, lr, device="cpu"), steps=5)
        state = {k: v for k, v in opt.state_dict().items()}
        q = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
        other = MasterAdamW([q], 1e-4, device="cpu")
        other.load_state_dict(state)
        torch.testing.assert_close(other.master[0], opt.master[0])
        torch.testing.assert_close(q, p)
        self.assertEqual(other.inner.state_dict()["state"][0]["step"], 5)

    def test_auto_mode(self):
        self.assertIsInstance(make_optimizer([torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))], 1e-3),
                              MasterAdamW)
        self.assertIsInstance(make_optimizer([torch.nn.Parameter(torch.ones(2))], 1e-3), torch.optim.AdamW)


if __name__ == "__main__":
    unittest.main()

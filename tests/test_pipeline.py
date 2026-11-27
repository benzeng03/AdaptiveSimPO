"""CPU smoke test: train -> dev/test scoring -> summary -> generation with a tiny random Qwen3."""
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "models/Qwen3-0.6B"


@unittest.skipUnless((TOKENIZER / "tokenizer.json").exists(), "needs the Qwen3-0.6B tokenizer")
class PipelineSmokeTest(unittest.TestCase):
    def test_end_to_end(self):
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

        import generate
        import score_pairs
        import summarize
        import train

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tok = AutoTokenizer.from_pretrained(TOKENIZER)
            cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                              num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                              head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
            torch.manual_seed(0)
            model_dir = tmp / "tiny"
            Qwen3ForCausalLM(cfg).save_pretrained(model_dir)
            tok.save_pretrained(model_dir)
            data = [json.loads(l) for l in (ROOT / "data/uf_original_train.jsonl").read_text(encoding="utf-8").splitlines()[:8]]
            train_file = tmp / "train.jsonl"
            train_file.write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
            common = ["--model", str(model_dir), "--data", str(train_file), "--device", "cpu",
                      "--no-bf16", "--no-gradient-checkpointing", "--batch-size", "4",
                      "--max-length", "96", "--max-prompt", "48", "--lr", "1e-3"]
            run = tmp / "run"
            train.main(common + ["--margin", "fixed", "--gamma", "0.5", "--output", str(run / "fixed-g0.5")])
            train.main(common + ["--margin", "calibrated", "--output", str(run / "calibrated")])
            lg = tmp / "logistic-run" / "calibrated-logistic-ema0.5"
            train.main(common + ["--k-family", "logistic", "--k-ema", "0.5", "--output", str(lg)])
            self.assertEqual(json.loads((lg / "margin_stats.json").read_text())["variant"], "calibrated-logistic-ema0.5")
            self.assertIn("calib_c", json.loads((lg / "metrics.jsonl").read_text().splitlines()[0]))
            with self.assertRaises(SystemExit):  # the family belongs to the calibrated margin
                train.main(common + ["--margin", "fixed", "--k-family", "logistic", "--output", str(tmp / "bad")])
            train.main(common + ["--margin", "fixed", "--gamma", "0.75", "--max-steps", "1",
                                 "--output", str(run / "fixed-g0.75")])

            stats = json.loads((run / "calibrated" / "margin_stats.json").read_text())
            self.assertEqual(stats["steps"], 2)
            self.assertEqual(stats["variant"], "calibrated")
            self.assertIn("weight_share_misranked", stats)
            self.assertIn("calib_k_mean", stats)
            lines = (run / "calibrated" / "metrics.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("calib_k", json.loads(lines[1]))
            self.assertNotIn("calib_k_mean", json.loads((run / "fixed-g0.5" / "margin_stats.json").read_text()))

            dev = tmp / "dev.jsonl"
            dev.write_text("\n".join(json.dumps(r) for r in data[:4]), encoding="utf-8")

            # Other rewards. DPO: the reference is the model with the adapter disabled, and the fresh
            # LoRA adapter is a no-op, so the first loss is exactly log 2.
            rw_run = tmp / "reward-run"
            for name, extra in (("dpo-fixed-g0", ["--reward", "dpo", "--margin", "fixed", "--gamma", "0"]),
                                ("dpo-calibrated", ["--reward", "dpo"]),
                                ("dpo-noref-calibrated", ["--reward", "dpo-noref"]),
                                ("rdpo-a0.05-fixed-g0", ["--reward", "r-dpo", "--margin", "fixed", "--gamma", "0"]),
                                ("orpo-l0.1-calibrated", ["--reward", "orpo", "--max-steps", "1"])):
                train.main(common + extra + ["--output", str(rw_run / name)])
                stats_ = json.loads((rw_run / name / "margin_stats.json").read_text())
                self.assertEqual(stats_["variant"], name)
            first_dpo = json.loads((rw_run / "dpo-fixed-g0" / "metrics.jsonl").read_text().splitlines()[0])
            self.assertAlmostEqual(first_dpo["loss"], math.log(2), places=5)
            first_cal = json.loads((rw_run / "dpo-calibrated" / "metrics.jsonl").read_text().splitlines()[0])
            self.assertEqual(first_cal["calib_k"], 0.01)  # all gaps are 0 at the start: k at its lower bound
            first_rdpo = json.loads((rw_run / "rdpo-a0.05-fixed-g0" / "metrics.jsonl").read_text().splitlines()[0])
            self.assertNotAlmostEqual(first_rdpo["reward_gap"], 0.0, places=4)  # the length offset alone
            orpo = json.loads((rw_run / "orpo-l0.1-calibrated" / "metrics.jsonl").read_text().splitlines()[0])
            self.assertAlmostEqual(orpo["loss"], orpo["nll"] + 0.1 * orpo["pref_loss"], places=5)
            # train.py --eval-data caches the trained model's log-probabilities; score_pairs.py uses
            # them instead of loading the model, with the same result as scoring from scratch
            from adaptive_simpo import evalcache
            ev_run = tmp / "eval-run"
            train.main(common + ["--margin", "fixed", "--gamma", "1", "--max-steps", "2", "--eval-data", str(dev),
                                 "--output", str(ev_run / "fixed-g1")])
            self.assertEqual(len(list((ev_run / "fixed-g1" / "eval_cache").glob("*.pt"))), 1)
            score_pairs.main([str(ev_run), "--split", "test", "--data", str(dev), "--batch-size", "2", "--no-base"])
            cached_gaps = json.loads((ev_run / "test_scores.json").read_text())[0]["gaps"]
            enc = (96, 48, True, "always")
            self.assertIsNotNone(evalcache.load(ev_run / "fixed-g1", dev, enc))
            self.assertIsNone(evalcache.load(ev_run / "fixed-g1", dev, (96, 40, True, "always")))  # other lengths
            shutil.rmtree(ev_run / "fixed-g1" / "eval_cache")
            score_pairs.main([str(ev_run), "--split", "test", "--data", str(dev), "--batch-size", "2", "--no-base"])
            fresh_gaps = json.loads((ev_run / "test_scores.json").read_text())[0]["gaps"]
            torch.testing.assert_close(torch.tensor(cached_gaps), torch.tensor(fresh_gaps), atol=1e-5, rtol=1e-5)
            other = tmp / "dev-other.jsonl"
            other.write_text(dev.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            self.assertIsNone(evalcache.load(ev_run / "fixed-g1", other, enc))  # other file content

            # full fine-tuning keeps a frozen copy of the base model as the reference: the first
            # step starts from the reference exactly, so the loss is log 2 again
            full_run = tmp / "full-dpo-run"
            train.main(common + ["--reward", "dpo", "--margin", "fixed", "--gamma", "0", "--no-lora",
                                 "--output", str(full_run / "dpo-fixed-g0")])
            full_dpo = [json.loads(l) for l in (full_run / "dpo-fixed-g0" / "metrics.jsonl").read_text().splitlines()]
            self.assertAlmostEqual(full_dpo[0]["loss"], math.log(2), places=5)
            self.assertNotAlmostEqual(full_dpo[-1]["reward_gap"], 0.0, places=6)  # the policy moved
            score_pairs.main([str(full_run), "--split", "test", "--data", str(dev), "--batch-size", "2", "--no-base"])
            full_scores = json.loads((full_run / "test_scores.json").read_text())
            self.assertEqual(full_scores[0]["reward"], "dpo")
            score_pairs.main([str(rw_run), "--split", "test", "--data", str(dev), "--batch-size", "2", "--no-base"])
            rw_scores = {r["adapter"]: r for r in json.loads((rw_run / "test_scores.json").read_text())}
            self.assertEqual(rw_scores["rdpo-a0.05-fixed-g0"]["reward"], "r-dpo")
            self.assertEqual(rw_scores["dpo-calibrated"]["margin"], "calibrated")
            self.assertIn("accuracy_mean_logp", rw_scores["orpo-l0.1-calibrated"])
            self.assertIsNone(score_pairs.select_fixed(list(rw_scores.values())))  # baselines are SimPO only

            score_pairs.main([str(run), "--data", str(dev), "--batch-size", "2"])
            scores = json.loads((run / "dev_scores.json").read_text())
            self.assertEqual([s["adapter"] for s in scores], ["base", "calibrated", "fixed-g0.5", "fixed-g0.75"])
            self.assertIn(json.loads((run / "selection.json").read_text())["best_fixed"], ("fixed-g0.5", "fixed-g0.75"))
            score_pairs.main([str(run), "--split", "test", "--data", str(dev), "--batch-size", "2"])
            self.assertTrue((run / "test_scores.json").exists())
            summarize.main([str(run)])
            self.assertIn("McNemar", (run / "results.md").read_text(encoding="utf-8"))

            gen = run / "gen"
            args = ["--model", str(model_dir), "--output", str(gen), "--device", "cpu", "--base",
                    "--adapter", f"fixed-g0.5={run / 'fixed-g0.5'}", "--adapter", f"calibrated={run / 'calibrated'}",
                    "--num-prompts", "3", "--group", "2", "--max-new-tokens", "4"]
            generate.main(args)
            files = sorted(p.name for p in gen.glob("*-*.json"))
            self.assertEqual(len(files), 6)  # 2 groups x 3 policies
            generate.main(args)  # cached rerun is a no-op
            first = json.loads((gen / "000-fixed-g0.5.json").read_text(encoding="utf-8"))
            self.assertEqual(len(first), 2)
            self.assertLessEqual(first[0]["tokens"], 4)
            # a cache made with other sampling settings is refused, not mixed in
            with self.assertRaises(SystemExit):
                generate.main(args + ["--top-k", "20"])
            topk = run / "gen-k20"
            generate.main([a if a != str(gen) else str(topk) for a in args]
                          + ["--temperature", "0.7", "--top-p", "0.8", "--top-k", "20"])
            meta = json.loads((topk / "prompts.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["temperature"], meta["top_p"], meta["top_k"]), (0.7, 0.8, 20))
            self.assertEqual(len(list(topk.glob("*-*.json"))), 6)

    def test_resume_matches_uninterrupted(self):
        """Kill training mid-run, resume from the checkpoint, and get the same run as without the kill."""
        from unittest import mock

        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

        import train

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tok = AutoTokenizer.from_pretrained(TOKENIZER)
            cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                              num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                              head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
            torch.manual_seed(0)
            model_dir = tmp / "tiny"
            Qwen3ForCausalLM(cfg).save_pretrained(model_dir)
            tok.save_pretrained(model_dir)
            data = (ROOT / "data/uf_original_train.jsonl").read_text(encoding="utf-8").splitlines()[:16]
            train_file = tmp / "train.jsonl"
            train_file.write_text("\n".join(data), encoding="utf-8")
            # 4 batches x 2 epochs = 8 steps; checkpoints after steps 3 and 6 (the second one
            # crosses the epoch boundary, so the resumed run must redraw epoch 1's shuffle).
            common = ["--model", str(model_dir), "--data", str(train_file), "--device", "cpu",
                      "--no-bf16", "--no-gradient-checkpointing", "--batch-size", "4", "--epochs", "2",
                      "--max-length", "96", "--max-prompt", "48", "--lr", "1e-3", "--margin", "calibrated",
                      "--checkpoint-every", "3"]
            train.main(common + ["--output", str(tmp / "full")])

            calls = {"n": 0}
            real = train.batch_logp_sums

            def dies_at_step_8(*a, **k):
                calls["n"] += 1
                if calls["n"] == 8:
                    raise KeyboardInterrupt
                return real(*a, **k)

            cut = tmp / "cut"
            with mock.patch.object(train, "batch_logp_sums", dies_at_step_8), self.assertRaises(KeyboardInterrupt):
                train.main(common + ["--output", str(cut)])
            self.assertTrue((cut / "checkpoint" / "state.json").exists())
            self.assertEqual(len((cut / "metrics.jsonl").read_text().splitlines()), 7)
            with self.assertRaises(SystemExit):  # without --resume a non-empty directory is refused
                train.main(common + ["--output", str(cut)])
            with self.assertRaises(SystemExit):  # resuming with other settings is refused
                train.main(common + ["--output", str(cut), "--resume", "--lr", "1e-2"])
            # a checkpoint written before --k-ema existed has no k_ema key: it counts as the default
            state_file = cut / "checkpoint" / "state.json"
            state = json.loads(state_file.read_text())
            del state["args"]["k_ema"]
            state_file.write_text(json.dumps(state))
            train.main(common + ["--output", str(cut), "--resume"])

            self.assertFalse((cut / "checkpoint").exists())
            full_log = [json.loads(l) for l in (tmp / "full" / "metrics.jsonl").read_text().splitlines()]
            cut_log = [json.loads(l) for l in (cut / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in cut_log], list(range(1, 9)))
            for a, b in zip(full_log, cut_log):
                self.assertAlmostEqual(a["loss"], b["loss"], places=5)
            full_w = load_file(tmp / "full" / "adapter_model.safetensors")
            cut_w = load_file(cut / "adapter_model.safetensors")
            for name, value in full_w.items():
                torch.testing.assert_close(cut_w[name], value)
            full_stats = json.loads((tmp / "full" / "margin_stats.json").read_text())
            cut_stats = json.loads((cut / "margin_stats.json").read_text())
            for key in ("steps", "pairs_seen", "gamma_mean", "calib_k_mean", "weight_share_misranked"):
                self.assertAlmostEqual(full_stats[key], cut_stats[key], places=5)

            # --k-ema: the moving averages are carried by the checkpoint, so k matches step by step.
            ema_args = common + ["--k-ema", "0.8"]
            train.main(ema_args + ["--output", str(tmp / "ema-full")])
            calls["n"] = 0
            with mock.patch.object(train, "batch_logp_sums", dies_at_step_8), self.assertRaises(KeyboardInterrupt):
                train.main(ema_args + ["--output", str(tmp / "ema-cut")])
            with self.assertRaises(SystemExit):  # the decay cannot change on resume
                train.main(common + ["--k-ema", "0.5", "--output", str(tmp / "ema-cut"), "--resume"])
            train.main(ema_args + ["--output", str(tmp / "ema-cut"), "--resume"])
            ema_full = [json.loads(l) for l in (tmp / "ema-full" / "metrics.jsonl").read_text().splitlines()]
            ema_cut = [json.loads(l) for l in (tmp / "ema-cut" / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(ema_cut), 8)
            for a, b in zip(ema_full, ema_cut):
                self.assertAlmostEqual(a["calib_k"], b["calib_k"], places=6)
                self.assertAlmostEqual(a["loss"], b["loss"], places=5)
            per_batch = [r["calib_k"] for r in full_log]
            self.assertNotEqual([round(r["calib_k"], 6) for r in ema_full][1:], [round(k, 6) for k in per_batch][1:])
            self.assertEqual(json.loads((tmp / "ema-full" / "margin_stats.json").read_text())["variant"],
                             "calibrated-ema0.8")

            # Killed before the first checkpoint: --resume starts over in the same directory.
            early = tmp / "early"
            calls["n"] = 0
            with mock.patch.object(train, "batch_logp_sums",
                                   lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt)), \
                    self.assertRaises(KeyboardInterrupt):
                train.main(common + ["--output", str(early)])
            train.main(common + ["--output", str(early), "--resume", "--max-steps", "1"])
            self.assertEqual(json.loads((early / "margin_stats.json").read_text())["steps"], 1)

    def test_full_finetune(self):
        """--no-lora in bf16: fp32 masters, resume, then scoring and generation with full models."""
        from unittest import mock

        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

        import generate
        import score_pairs
        import train
        from adaptive_simpo.modeling import is_adapter_dir, is_full_dir

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tok = AutoTokenizer.from_pretrained(TOKENIZER)
            cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                              num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                              head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
            torch.manual_seed(0)
            model_dir = tmp / "tiny"
            Qwen3ForCausalLM(cfg).save_pretrained(model_dir)
            tok.save_pretrained(model_dir)
            data = (ROOT / "data/uf_original_train.jsonl").read_text(encoding="utf-8").splitlines()[:16]
            train_file = tmp / "train.jsonl"
            train_file.write_text("\n".join(data), encoding="utf-8")
            common = ["--model", str(model_dir), "--data", str(train_file), "--device", "cpu",
                      "--no-gradient-checkpointing", "--batch-size", "4", "--max-length", "96",
                      "--max-prompt", "48", "--lr", "1e-3", "--margin", "fixed", "--gamma", "1",
                      "--checkpoint-every", "2"]
            run = tmp / "run"
            train.main(common + ["--no-lora", "--output", str(run / "full")])
            self.assertTrue(is_full_dir(run / "full"))
            self.assertFalse(is_adapter_dir(run / "full"))
            trained = load_file(run / "full" / "model.safetensors")
            base = load_file(model_dir / "model.safetensors")
            self.assertTrue(any(not torch.equal(trained[k].float(), base[k].float()) for k in base))

            calls = {"n": 0}
            real = train.batch_logp_sums

            def dies_at_step_4(*a, **k):
                calls["n"] += 1
                if calls["n"] == 4:
                    raise KeyboardInterrupt
                return real(*a, **k)

            cut = tmp / "cut"
            with mock.patch.object(train, "batch_logp_sums", dies_at_step_4), self.assertRaises(KeyboardInterrupt):
                train.main(common + ["--no-lora", "--output", str(cut)])
            ckpt = sorted(p.name for p in (cut / "checkpoint").iterdir())
            self.assertEqual(ckpt, ["optimizer.pt", "state.json"])  # the weights are the fp32 masters
            train.main(common + ["--no-lora", "--output", str(cut), "--resume"])
            resumed = load_file(cut / "model.safetensors")
            for name, value in trained.items():
                torch.testing.assert_close(resumed[name], value)

            # a run directory mixing a full model and a LoRA adapter
            train.main(common + ["--output", str(run / "lora")])
            dev = tmp / "dev.jsonl"
            dev.write_text("\n".join(data[:4]), encoding="utf-8")
            score_pairs.main([str(run), "--split", "test", "--data", str(dev), "--batch-size", "2"])
            scores = {s["adapter"]: s for s in json.loads((run / "test_scores.json").read_text())}
            self.assertEqual(sorted(scores), ["base", "full", "lora"])
            self.assertNotEqual(scores["full"]["gaps"], scores["base"]["gaps"])

            gen = tmp / "gen"
            generate.main(["--model", str(model_dir), "--output", str(gen), "--device", "cpu", "--base",
                           "--adapter", f"full={run / 'full'}", "--adapter", f"lora={run / 'lora'}",
                           "--num-prompts", "2", "--group", "2", "--max-new-tokens", "4"])
            self.assertEqual(sorted(p.name for p in gen.glob("*-*.json")),
                             ["000-base.json", "000-full.json", "000-lora.json"])


if __name__ == "__main__":
    unittest.main()

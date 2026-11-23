"""CPU smoke test: train -> dev/test scoring -> summary -> generation with a tiny random Qwen3."""
import json
import math
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

            # DPO: the reference is the model with the adapter disabled, and the fresh LoRA
            # adapter is a no-op, so the first loss is exactly log 2.
            dpo_run = tmp / "dpo-run"
            train.main(common + ["--loss", "dpo", "--output", str(dpo_run / "dpo-b0.1")])
            first_dpo = json.loads((dpo_run / "dpo-b0.1" / "metrics.jsonl").read_text().splitlines()[0])
            self.assertAlmostEqual(first_dpo["loss"], math.log(2), places=5)
            dpo_stats = json.loads((dpo_run / "dpo-b0.1" / "margin_stats.json").read_text())
            self.assertEqual((dpo_stats["variant"], dpo_stats["loss"], dpo_stats["steps"]), ("dpo-b0.1", "dpo", 2))
            score_pairs.main([str(dpo_run), "--split", "test", "--data", str(dev), "--batch-size", "2", "--no-base"])
            self.assertEqual(json.loads((dpo_run / "test_scores.json").read_text())[0]["margin"], "dpo")

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


if __name__ == "__main__":
    unittest.main()

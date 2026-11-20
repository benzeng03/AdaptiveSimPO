"""CPU smoke test: train -> dev scoring -> generation with a tiny random Qwen3."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT.parent / "Qwen3-0.6B"


@unittest.skipUnless((TOKENIZER / "tokenizer.json").exists(), "needs the Qwen3-0.6B tokenizer")
class PipelineSmokeTest(unittest.TestCase):
    def test_end_to_end(self):
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

        import generate
        import score_pairs
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
            train.main(common + ["--margin", "adaptive", "--threshold", "mean", "--output", str(run / "erf-mean-g2")])
            train.main(common + ["--margin", "amapo", "--max-steps", "1", "--output", str(run / "amapo")])

            stats = json.loads((run / "erf-mean-g2" / "margin_stats.json").read_text())
            self.assertEqual(stats["steps"], 2)
            self.assertAlmostEqual(stats["normal_ref_gamma_mean"], 0.5, places=3)
            lines = (run / "erf-mean-g2" / "metrics.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)

            dev = tmp / "dev.jsonl"
            dev.write_text("\n".join(json.dumps(r) for r in data[:4]), encoding="utf-8")
            score_pairs.main([str(run), "--data", str(dev), "--batch-size", "2"])
            scores = json.loads((run / "dev_scores.json").read_text())
            self.assertEqual([s["adapter"] for s in scores], ["base", "amapo", "erf-mean-g2", "fixed-g0.5"])
            self.assertEqual(json.loads((run / "selection.json").read_text())["best_fixed"], "fixed-g0.5")

            gen = run / "gen"
            args = ["--model", str(model_dir), "--output", str(gen), "--device", "cpu", "--base",
                    "--adapter", f"fixed-g0.5={run / 'fixed-g0.5'}", "--adapter", f"erf-mean-g2={run / 'erf-mean-g2'}",
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

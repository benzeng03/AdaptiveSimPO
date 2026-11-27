"""On-policy data: sampling with several seeds, combining, reward-model binarization (SimPO rules)."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from onpolicy import annotate, decode

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "models/Qwen3-0.6B"


def write_seed(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class AnnotateTests(unittest.TestCase):
    def test_combine_binarize_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prompts = ["p1", "p2", "p3"]
            answers = {13: ["a", "same", "x"], 21: ["bb", "same", "y"], 42: ["ccc", "same", "z"]}
            for seed, texts in answers.items():
                write_seed(tmp / f"output_{seed}.jsonl",
                           [dict(prompt=p, generated_text=t, tokens=1, finish_reason="stop") for p, t in zip(prompts, texts)])
            records, identical = annotate.combine(tmp)
            self.assertEqual(identical, 1)  # p2 answered identically under every seed
            self.assertEqual([r["prompt"] for r in records], ["p1", "p3"])
            self.assertEqual(records[0]["all_generated_responses"], ["a", "bb", "ccc"])  # seed order

            calls = []

            def score(prompt, answer):  # longer answers score higher; x/y/z tie
                calls.append(answer)
                return float(len(answer)) if prompt == "p1" else 0.5

            cache = tmp / "pairs.jsonl.scores.jsonl"
            pairs = annotate.annotate(records, score, cache)
            self.assertEqual((pairs[0]["chosen"], pairs[0]["rejected"]), ("ccc", "a"))
            self.assertEqual((pairs[0]["score_chosen"], pairs[0]["score_rejected"]), (3.0, 1.0))
            self.assertEqual((pairs[1]["chosen"], pairs[1]["rejected"]), ("x", "x"))  # ties: first index
            self.assertEqual(len(calls), 6)
            again = annotate.annotate(records, score, cache)  # cached: no new scoring
            self.assertEqual(len(calls), 6)
            self.assertEqual(again, pairs)

    def test_mismatched_seed_files_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_seed(tmp / "output_1.jsonl", [dict(prompt="p", generated_text="a")])
            write_seed(tmp / "output_2.jsonl", [dict(prompt="q", generated_text="b")])
            with self.assertRaises(SystemExit):
                annotate.combine(tmp)


@unittest.skipUnless((TOKENIZER / "tokenizer.json").exists(), "needs the Qwen3-0.6B tokenizer")
class DecodeTests(unittest.TestCase):
    def test_hf_engine_seeds_and_resume(self):
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tok = AutoTokenizer.from_pretrained(TOKENIZER)
            cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                              num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                              max_position_embeddings=512, tie_word_embeddings=True)
            torch.manual_seed(0)
            Qwen3ForCausalLM(cfg).save_pretrained(tmp / "tiny")
            tok.save_pretrained(tmp / "tiny")
            data = tmp / "prompts.jsonl"
            rows = [dict(prompt=p, chosen="c", rejected="r") for p in ("Hi", "Name a fruit.", "Hi", "2+2?")]
            data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            out = tmp / "gen"
            args = ["--model", str(tmp / "tiny"), "--data", str(data), "--output-dir", str(out), "--engine", "hf",
                    "--device", "cpu", "--seeds", "13", "21", "--max-tokens", "6", "--batch-size", "2"]
            decode.main(args)
            s13 = [json.loads(l) for l in (out / "output_13.jsonl").read_text(encoding="utf-8").splitlines()]
            s21 = [json.loads(l) for l in (out / "output_21.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["prompt"] for r in s13], ["Hi", "Name a fruit.", "2+2?"])  # unique, in order
            self.assertTrue(all(r["tokens"] <= 6 for r in s13 + s21))
            self.assertNotEqual([r["generated_text"] for r in s13], [r["generated_text"] for r in s21])
            before = (out / "output_13.jsonl").stat().st_mtime_ns
            decode.main(args + ["--seeds", "13", "21", "42"])  # finished seeds are kept, 42 is added
            self.assertEqual((out / "output_13.jsonl").stat().st_mtime_ns, before)
            self.assertTrue((out / "output_42.jsonl").exists())
            with self.assertRaises(SystemExit):  # other sampling settings would mix into the cache
                decode.main(args + ["--temperature", "0.5"])
            with self.assertRaises(SystemExit):
                decode.main(args + ["--top-k", "20"])
            k20 = tmp / "gen-k20"
            decode.main([a if a != str(out) else str(k20) for a in args]
                        + ["--temperature", "0.7", "--top-p", "0.8", "--top-k", "20"])
            meta = json.loads((k20 / "decode_args.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["temperature"], meta["top_p"], meta["top_k"]), (0.7, 0.8, 20))
            records, _ = annotate.combine(out)
            self.assertEqual(len(records[0]["all_generated_responses"]), 3)


if __name__ == "__main__":
    unittest.main()

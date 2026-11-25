"""encode_pair truncation: the project's 'always' rule and SimPO's (TRL keep_end) rule."""
import unittest

from adaptive_simpo.modeling import encode_pair


class CharTokenizer:
    """One token per character; BOS 1 is prepended to (non-chat) prompts, EOS 2 ends answers."""
    bos_token_id, eos_token_id = 1, 2

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def lengths(prompt, chosen, rejected, truncation, max_length=10, max_prompt=4):
    row = dict(prompt=prompt, chosen=chosen, rejected=rejected)
    out = encode_pair(row, CharTokenizer(), max_length, max_prompt, chat_prompt=False, truncation=truncation)
    prompt_len = [labels.count(-100) for _, labels in out]
    assert prompt_len[0] == prompt_len[1]
    return prompt_len[0], [len(ids) - p for (ids, _), p in zip(out, prompt_len)]


def simpo_reference(prompt_len, answer_lens, max_length, max_prompt):
    """SimPO scripts/simpo_trainer.py tokenize_row (keep_end), on lengths only."""
    longer = max(answer_lens)
    if prompt_len + longer > max_length:
        prompt_len = min(prompt_len, max_prompt)
    if prompt_len + longer > max_length:
        answer_lens = [min(a, max_length - max_prompt) for a in answer_lens]
    return prompt_len, answer_lens


class TruncationTests(unittest.TestCase):
    def test_short_pair_is_untouched(self):
        # prompt 1 + 3 = 4 tokens, answers 2 + EOS and 1 + EOS
        self.assertEqual(lengths("abc", "xy", "z", "always"), (4, [3, 2]))
        self.assertEqual(lengths("abc", "xy", "z", "simpo"), (4, [3, 2]))

    def test_simpo_keeps_a_long_prompt_when_the_pair_fits(self):
        # prompt 7 tokens + longer answer 3 = 10 <= max_length
        self.assertEqual(lengths("abcdef", "xy", "z", "simpo"), (7, [3, 2]))
        self.assertEqual(lengths("abcdef", "xy", "z", "always"), (4, [3, 2]))

    def test_simpo_cuts_the_prompt_first(self):
        # prompt 9 + 3 > 10: prompt -> last 4 tokens, then 4 + 3 fits
        self.assertEqual(lengths("abcdefgh", "xy", "z", "simpo"), (4, [3, 2]))

    def test_simpo_cuts_answers_to_max_length_minus_max_prompt(self):
        # prompt 3, answers 12 and 3: still too long after the prompt cut, so both answers are
        # cut to 10 - 4 = 6 tokens even though the prompt is shorter than 4
        self.assertEqual(lengths("ab", "x" * 11, "yy", "simpo"), (3, [6, 3]))
        self.assertEqual(lengths("ab", "x" * 11, "yy", "always"), (3, [7, 3]))

    def test_simpo_matches_reference_on_a_grid(self):
        for p in range(1, 14):
            for a in range(2, 14):
                for b in range(2, 6):
                    got = lengths("p" * (p - 1), "a" * (a - 1), "b" * (b - 1), "simpo")
                    self.assertEqual(got, simpo_reference(p, [a, b], 10, 4), (p, a, b))
                    pl, (al, bl) = got
                    self.assertLessEqual(pl + max(al, bl), 10)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            lengths("a", "b", "c", "keep_start")


if __name__ == "__main__":
    unittest.main()

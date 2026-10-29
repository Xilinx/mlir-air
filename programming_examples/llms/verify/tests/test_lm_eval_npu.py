# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""lm_eval_npu.py against a fake runner and tokenizer (no NPU, no weights)."""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lm_eval_npu as le  # noqa: E402

try:
    import lm_eval  # noqa: F401
except ImportError:
    lm_eval = None

VOCAB = 10


class CharTokenizer:
    """One token per character, digits only; 9 is EOS."""

    bos_token_id = None
    bos_token = None
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return [int(c) for c in text if c.isdigit()]

    def decode(self, ids):
        return "".join(str(i) for i in ids)


class CountingRunner:
    """Next token is always (last token + 1) % VOCAB."""

    def __init__(self):
        self.prefills = 0

    @staticmethod
    def _logits(tok):
        row = np.zeros(VOCAB, np.float32)
        row[(tok + 1) % VOCAB] = 5.0
        return row

    def prefill_logits(self, toks, logits_from):
        self.prefills += 1
        return np.stack([self._logits(t) for t in toks[logits_from:]])

    def prefill(self, toks):
        self.prompt = list(toks)
        return SimpleNamespace(top1_token=(int(toks[-1]) + 1) % VOCAB)

    def decode_step(self, tok, pos):
        return SimpleNamespace(top1_token=(tok + 1) % VOCAB)


def req(*args):
    return SimpleNamespace(args=args)


@unittest.skipIf(lm_eval is None, "needs lm_eval")
class NpuLMTest(unittest.TestCase):
    def setUp(self):
        self.runner = CountingRunner()
        self.lm = le._lm_class()(self.runner, CharTokenizer(), None, 64, 64)

    def test_shared_context_single_token_choices_share_a_prefill(self):
        res = self.lm.loglikelihood([req("123", "4"), req("123", "5"), req("123", "6")])
        self.assertEqual(self.runner.prefills, 1)
        self.assertEqual([g for _, g in res], [True, False, False])
        self.assertGreater(res[0][0], res[1][0])

    def test_multi_token_continuation(self):
        ((lp, greedy),) = self.lm.loglikelihood([req("12", "345")])
        self.assertTrue(greedy)
        self.assertAlmostEqual(lp, 3 * float(-np.log1p(9 * math.exp(-5))), places=4)

    def test_empty_context_starts_from_eos(self):
        ((lp, greedy),) = self.lm.loglikelihood([req("", "0")])
        self.assertTrue(greedy)

    def test_generate_stops_at_until_and_eos(self):
        out = self.lm.generate_until(
            [
                req("1", {"until": ["4"], "max_gen_toks": 20}),
                req("6", {"until": [], "max_gen_toks": 20}),
            ]
        )
        self.assertEqual(out, ["23", "78"])

    def test_generate_stops_at_any_stop_id(self):
        lm = le._lm_class()(self.runner, CharTokenizer(), None, 64, 64, {5})
        self.assertEqual(
            lm.generate_until([req("1", {"until": [], "max_gen_toks": 20})]), ["234"]
        )

    def test_prompt_cut_leaves_the_whole_budget(self):
        lm = le._lm_class()(self.runner, CharTokenizer(), None, 8, 8)
        lm.generate_until([req("12345678", {"until": [], "max_gen_toks": 3})])
        self.assertEqual(self.runner.prompt, [4, 5, 6, 7, 8])
        with self.assertRaises(ValueError):
            lm.generate_until([req("1", {"until": [], "max_gen_toks": 8})])


class BosTokenizer(CharTokenizer):
    """CharTokenizer whose BOS (id 8) is spelled "<s>"."""

    bos_token_id = 8
    bos_token = "<s>"

    def encode(self, text, add_special_tokens=False):
        if text.startswith("<s>"):
            return [8] + super().encode(text[3:])
        return super().encode(text)


@unittest.skipIf(lm_eval is None, "needs lm_eval")
class ChatPromptTest(unittest.TestCase):
    def test_no_second_bos_after_a_template(self):
        runner = CountingRunner()
        lm = le._lm_class()(runner, BosTokenizer(), 8, 64, 64)
        lm.generate_until([req("<s>12", {"until": [], "max_gen_toks": 2})])
        self.assertEqual(runner.prompt, [8, 1, 2])
        lm.generate_until([req("12", {"until": [], "max_gen_toks": 2})])
        self.assertEqual(runner.prompt, [8, 1, 2])


class StatsTest(unittest.TestCase):
    def test_chat_only_for_chat_tasks_when_both_sides_have_a_template(self):
        chat = SimpleNamespace(chat_template="{{ messages }}")
        plain = SimpleNamespace(chat_template=None)
        self.assertEqual(
            le.chat_tasks(["gsm8k", "mmlu"], chat, chat),
            {"gsm8k": True, "mmlu": False},
        )
        self.assertEqual(le.chat_tasks(["gsm8k"], chat, plain), {"gsm8k": False})

    def test_mcnemar(self):
        self.assertEqual(le.mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(le.mcnemar_p(0, 6), 2 / 64)
        self.assertEqual(le.mcnemar_p(5, 5), 1.0)

    def test_group_shots(self):
        shots = {"gsm8k": 5, "mmlu_anatomy": 0, "mmlu_virology": 0}
        self.assertEqual(le.group_shots(shots), {"gsm8k": 5, "mmlu": 0})
        self.assertEqual(le.group_shots({"mmlu_a": 0, "mmlu_b": 5}), {"mmlu": None})

    def test_compare_pairs_items(self):
        npu = {"gsm8k": {"a:0": 1.0, "a:1": 0.0, "a:2": 1.0, "a:3": 1.0}}
        ref = {"gsm8k": {"a:0": 1.0, "a:1": 1.0, "a:2": 0.0, "a:9": 1.0}}
        r = le.compare(npu, ref)["gsm8k"]
        self.assertEqual(r["n"], 3)
        self.assertEqual((r["npu_only_right"], r["ref_only_right"]), (1, 1))
        self.assertAlmostEqual(r["npu"], 2 / 3)


if __name__ == "__main__":
    unittest.main()

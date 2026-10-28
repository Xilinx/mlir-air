# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""dequant_reference.py on a tiny tied Llama saved locally (no download)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:  # the host suite may run without torch/transformers
    torch = None

import dequant_reference as dr  # noqa: E402


@unittest.skipIf(torch is None, "needs torch and transformers")
class DequantizedReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cfg = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
        torch.manual_seed(0)
        LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_replaces_weights_and_unties_head(self):
        name = "model.layers.1.mlp.down_proj.weight"
        model = dr.dequantized_reference(
            self.tmp.name,
            [
                (name, lambda shape: np.full(shape, 0.5)),
                (dr.LM_HEAD, lambda shape: np.full(shape, 0.25)),
            ],
        )
        params = dict(model.named_parameters())
        self.assertTrue(torch.all(params[name] == 0.5))
        self.assertTrue(torch.all(model.get_output_embeddings().weight == 0.25))
        embed = model.get_input_embeddings().weight
        self.assertFalse(torch.all(embed == 0.25))
        self.assertFalse(model.config.tie_word_embeddings)

    def test_unknown_or_misshapen_weight_is_an_error(self):
        with self.assertRaises(KeyError):
            dr.dequantized_reference(
                self.tmp.name, [("model.layers.9.mlp.up_proj.weight", np.zeros)]
            )
        with self.assertRaises(ValueError):
            dr.dequantized_reference(
                self.tmp.name,
                [("model.layers.0.mlp.up_proj.weight", lambda shape: np.zeros((2, 2)))],
            )


@unittest.skipIf(torch is None, "needs torch")
class PackedRequantTest(unittest.TestCase):
    def test_scale_and_min_are_bf16(self):
        import verify_runner  # noqa: F401  (publishes air_examples)
        from air_examples.llms.shared.infra import host_weights
        from ml_dtypes import bfloat16

        w = np.random.default_rng(0).standard_normal((64, 128)).astype(np.float32)
        q, sc, mn = host_weights.requant_q4k(w, 32)
        bf = lambda a: a.astype(bfloat16).astype(np.float32)  # noqa: E731
        want = host_weights.dequant_q4(q, bf(sc), bf(mn), 32)
        got = host_weights.requant_q4k_packed(w, 32)
        np.testing.assert_array_equal(got, want)
        self.assertGreater(
            np.abs(got - host_weights.requant_q4k_dequant(w, 32)).max(), 0
        )


if __name__ == "__main__":
    unittest.main()

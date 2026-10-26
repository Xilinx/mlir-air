# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host checks for logical Q4 and the fixed-input model gate; no device imports."""

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
from .affine_q4 import (
    CanonicalWeights,
    file_hash,
    pack_nibbles,
    unpack_nibbles,
    reconstruct,
)
from .deployment import MODELS, configuration, require_bfp16, spec
from .deployment_cli import compare_capture, save_capture
from .prepare_affine_q4 import prepare, quantize


class AffineDeploymentTests(unittest.TestCase):
    def test_checkpoint_to_canonical_q4(self):
        from ml_dtypes import bfloat16

        matrix = np.tile(np.arange(16, dtype=np.float32), (4, 4)).astype(bfloat16)
        norm = np.ones(64, dtype=bfloat16)
        tensors = {"projection.weight": matrix, "norm.weight": norm}
        header, offset = {}, 0
        for name, value in tensors.items():
            header[name] = dict(
                dtype="BF16",
                shape=list(value.shape),
                data_offsets=[offset, offset + value.nbytes],
            )
            offset += value.nbytes
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "hf", Path(tmp) / "q4"
            source.mkdir()
            (source / "config.json").write_text("{}\n")
            shard = source / "model.safetensors"
            shard.write_bytes(
                len(encoded).to_bytes(8, "little")
                + encoded
                + b"".join(a.tobytes() for a in tensors.values())
            )
            model = dict(
                checkpoint="local/unit-test",
                revision="test",
                config_sha256=file_hash(source / "config.json"),
                source_shards=[
                    dict(
                        file=shard.name,
                        bytes=shard.stat().st_size,
                        sha256=file_hash(shard),
                    )
                ],
                q4_aliases={"lm_head.weight": "projection.weight"},
                bf16_matrices=[],
            )
            prepare(source, out, model, chunk_rows=3)
            reader = CanonicalWeights(out)
            np.testing.assert_array_equal(reader.dequant("projection.weight"), matrix)
            np.testing.assert_array_equal(reader.dequant("lm_head.weight"), matrix)
            np.testing.assert_array_equal(
                reader.bf16("norm.weight", as_bits=True), norm.view(np.uint16)
            )
            for entry in reader.entries.values():
                for array in entry["arrays"].values():
                    self.assertEqual(file_hash(out / array["path"]), array["sha256"])

    def test_cache_close_releases_only_its_contexts(self):
        import weakref
        from .infra.cache import KernelCache

        class Backend:
            def __init__(self):
                self.unloaded = 0

            def unload(self):
                self.unloaded += 1

        with tempfile.TemporaryDirectory() as tmp:
            a = KernelCache(cache_dir=Path(tmp) / "a")
            b = KernelCache(cache_dir=Path(tmp) / "b")
            ba, bb = Backend(), Backend()
            for cache, backend in ((a, ba), (b, bb)):
                cache._loaded["kernel"] = backend
                cache._contexts[(cache._uid, "kernel")] = (weakref.ref(cache), "kernel")
            a.close()
            self.assertEqual(ba.unloaded, 1)
            self.assertEqual(bb.unloaded, 0)
            self.assertNotIn((a._uid, "kernel"), a._contexts)
            self.assertIn((b._uid, "kernel"), b._contexts)
            a.close()
            self.assertEqual(ba.unloaded, 1)
            b.close()

    def test_quantization_known_values(self):
        x = np.tile(np.arange(16, dtype=np.float32), (4, 2))
        q, scale, minimum = quantize(x)
        np.testing.assert_array_equal(q, x.astype(np.uint8))
        np.testing.assert_array_equal(unpack_nibbles(pack_nibbles(q)), q)
        np.testing.assert_array_equal(reconstruct(q, scale, minimum), x)
        flat = np.full((2, 32), -2.0, dtype=np.float32)
        np.testing.assert_array_equal(reconstruct(*quantize(flat)), flat)

    def test_nonfinite_quantization_rejected(self):
        x = np.zeros((1, 32), dtype=np.float32)
        x[0, -1] = np.nan
        with self.assertRaises(ValueError):
            quantize(x)

    def test_native_configuration_matrix(self):
        counts = []
        for model in MODELS:
            native = dict(spec(model)["configurations"])
            counts.append(len(native))
            for p in (512, 1024, 2048, 4096):
                if p in native:
                    row = configuration(model, p)
                    self.assertEqual(row["capacity"], native[p])
                    self.assertEqual(len(row["token_ids"]), p)
                else:
                    with self.assertRaises(ValueError):
                        configuration(model, p)
        self.assertEqual(sum(counts), 19)

    def test_native_bf16_artifact_rejected(self):
        for metadata in ({}, {"gemm_bfp16": False}, {"gemm_bfp16": "true"}):
            with self.assertRaises(ValueError):
                require_bfp16(metadata)
        require_bfp16({"gemm_bfp16": True})

    def test_complete_capture_and_first_divergence(self):
        model = "phi2_2_7b_q4nx"
        selected = configuration(model, 512)
        vocab = spec(model)["vocab_size"]
        reference = np.zeros((32, vocab), dtype=np.float32)
        reference[:, 3] = 10
        reference[:, 7] = 9
        candidate = reference.copy()
        candidate[5:, 7] = 11
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.npz", Path(tmp) / "b.npz"

            def save(path, z):
                save_capture(
                    path,
                    selected,
                    dict(logits=z, tokens=z.argmax(axis=1).tolist()),
                    model,
                )

            save(a, candidate)
            save(b, reference)
            result = compare_capture(a, b, selected, model)
            self.assertEqual(result["status"], "OK")
            self.assertEqual(result["divergence_step"], 5)
            candidate[5, 100] = 20
            save(a, candidate)
            self.assertEqual(compare_capture(a, b, selected, model)["status"], "FAIL")
            # A malformed tail must fail even when an earlier divergence passed.
            candidate = reference.copy()
            candidate[-1, -1] = np.nan
            save(a, candidate)
            with self.assertRaises(ValueError):
                compare_capture(a, b, selected, model)
            save(a, reference[:-1])
            with self.assertRaises(ValueError):
                compare_capture(a, b, selected, model)


if __name__ == "__main__":
    unittest.main()

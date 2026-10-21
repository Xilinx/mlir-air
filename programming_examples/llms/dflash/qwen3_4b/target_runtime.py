# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Manifest-driven Qwen3-4B target runtime; no ambient builder configuration.

The caller owns the committed position. Verification writes a candidate block's
KV in place; only rows through the accepted prefix become live. Subsequent calls
overwrite rejected slots and causal attention must ignore every future slot.
"""

from __future__ import annotations
from paths import HERE, COMMON
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

from target_config import ABI_VERSION, HERE, FUSED, TargetConfig
from model_paths import TARGET_MODEL, default_target_cache, resolve_target_model


from common.decoder import DecoderState, Verification


class TargetDecoder(DecoderState):
    def __init__(
        self,
        artifacts,
        *,
        packed_cache=None,
        model_source=TARGET_MODEL,
        model_weights=None,
        stream_weights=None,
    ):
        import pyxrt as xrt

        self.xrt = xrt
        self.artifacts = Path(artifacts).resolve()
        manifest = json.loads((self.artifacts / "manifest.json").read_text())
        if (
            manifest["status"] != "compiled"
            or manifest["abi"]["version"] != ABI_VERSION
        ):
            raise ValueError("Artifact compilation or ABI validation is incomplete")
        self.interface = manifest["abi"]
        self.config = TargetConfig(**manifest["config"])
        for name, want in manifest["artifacts"].items():
            if hashlib.sha256((self.artifacts / name).read_bytes()).hexdigest() != want:
                raise ValueError("Artifact hash mismatch: " + name)
        expected = {
            f"decode_dyn_L{self.config.capacity}.{ext}" for ext in ("xclbin", "txn.h")
        }
        if set(manifest["artifacts"]) != expected:
            raise ValueError("Unexpected template set")
        self.batch = self.config.batch
        self.capacity = self.config.capacity
        self.layers = self.config.layers
        self.g = self.interface["geometry"]
        self.n = self.interface["elements"]
        self.offset = self.interface["offsets"]
        self.hidden_size = self.g["hidden"]
        self.vocab_size = self.g["vocab"]
        self.kv_shape = (
            self.layers,
            2 * self.g["kv_groups"],
            self.capacity,
            self.g["kv_region_width"],
        )
        for path in (FUSED, HERE.parents[1] / "qwen3_4b_q4nx"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from common.decode_dynseq import DynseqInstsGen, make_insts_states, patch_insts
        from decode_staircase import open_windows
        import qwen3_4b_q4nx_weights as weights

        self.weights = weights
        self.gen = DynseqInstsGen(str(self.artifacts), max_L=self.capacity)
        if self.gen.calibrated_windows() != [self.capacity]:
            raise ValueError("Template capacity disagrees with manifest")
        self.dev = xrt.device(0)
        self._windows = open_windows(self.dev, xrt, self.gen, [self.capacity])
        self.context, self.kernel = self._windows[self.capacity]
        self.TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.bo = {
            name: xrt.bo(
                self.dev,
                self.n[name] * 2,
                xrt.bo.host_only,
                self.kernel.group_id(i + 3),
            )
            for i, name in enumerate(("x", "w", "r", "y", "kv"))
        }
        self.dispatch_buffers = [self.bo[n] for n in ("x", "w", "r", "y", "kv")]
        self.insts = make_insts_states(
            self.gen, xrt, self.dev, self.kernel.group_id(1), [self.capacity]
        )[self.capacity]
        self.patch_insts = patch_insts
        self._last_length = None
        self.timeout_ms = 10000
        if self.config.role == "drafter" and model_weights is None:
            raise ValueError("The drafter requires an explicit weight provider")
        self.model = (
            model_weights
            if model_weights is not None
            else weights.Q4nxModel(resolve_target_model(model_source))
        )
        if model_weights is None:
            # The bundle stores this tensor as BF16. The generic accessor
            # expands it to FP32, creating a large temporary while prefill and
            # decoder BOs are resident. Keep its original bits for embedding.
            name = "model.embed_tokens.weight"
            if self.model._hdr[name]["dtype"] != "BF16":
                raise ValueError("Qwen3-4B requires a BF16 embedding tensor")
            self.embed = self.model._raw_i16(name).view(bfloat16)
            final_norm = self.model.bf16("model.norm.weight")
        else:
            embed, final_norm, _ = self.model.embed_norm_lmhead()
            self.embed = np.asarray(embed, dtype=bfloat16)
        self.embed = self.embed.reshape(-1, self.hidden_size)
        self.qk = [self.model.layer_qk_norm(i) for i in range(self.layers)]
        self.rope = np.empty(
            (self.layers, self.batch, self.g["rope_width"]), dtype=bfloat16
        )
        # q/k normalization is constant across positions.
        for i, (q, k) in enumerate(self.qk):
            self.rope[i, :, 128:256] = q
            self.rope[i, :, 256:384] = k
        packed_cache = (
            Path(packed_cache or default_target_cache()).expanduser().resolve()
        )
        if not packed_cache.is_file():
            raise FileNotFoundError(
                f"Packed weight cache is missing: {packed_cache}; run make weights"
            )
        streamed = (
            self.batch == 16 if stream_weights is None else stream_weights
        ) and (self.layers == 36 or self.config.role == "drafter")
        if streamed:
            from common.weight_io import upload_weights

            upload_weights(self.bo["w"], packed_cache, self.n["w"])
            self.bo["w"].sync(self.TO)
        with np.load(packed_cache) as z:
            if not streamed:
                w = z["W"].view(np.int16).reshape(-1)
                if self.layers != 36 and self.config.role == "target":
                    from target_config import load_geometry

                    geometry = load_geometry(self.config)
                    head_offset = 36 * geometry.W_LAYER
                    expected = (
                        head_offset
                        + geometry.UNI_LM
                        * geometry.VOCAB_W_BLOCKS
                        * geometry.BLOCK_BF16
                    )
                    if w.size != expected:
                        raise ValueError(
                            ("Full weight cache geometry mismatch", w.size, expected)
                        )
                    w = np.concatenate(
                        (w[: self.layers * geometry.W_LAYER], w[head_offset:])
                    )
                if w.size != self.n["w"]:
                    raise ValueError(
                        ("Packed weight geometry mismatch", w.size, self.n["w"])
                    )
                self.bo["w"].write(w, 0)
                self.bo["w"].sync(self.TO)
            norms = (
                np.stack(
                    [z["RMS_in"][: self.layers], z["RMS_post"][: self.layers]], axis=1
                )
                .reshape(-1)
                .view(np.uint16)
            )
        r = np.zeros(self.n["r"], np.uint16)
        if norms.size != self.offset["rope"]:
            raise ValueError("Normalization weight geometry mismatch")
        r[: norms.size] = norms
        r[self.offset["final_norm"] :] = np.asarray(final_norm, bfloat16).view(
            np.uint16
        )
        self.bo["r"].write(r.view(np.int16), 0)
        self.bo["r"].sync(self.TO)
        for name in ("x", "y", "kv"):
            # Initialize every element without a second, capacity-sized array.
            np.frombuffer(self.bo[name].map(), np.int16, count=self.n[name]).fill(0)
            self.bo[name].sync(self.TO)
        print(
            json.dumps(
                {
                    "loaded": self.config.role,
                    "batch": self.batch,
                    "capacity": self.capacity,
                    "taps": self.config.taps,
                    "weights_bytes": self.n["w"] * 2,
                }
            ),
            flush=True,
        )

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Manifest-driven Qwen3-4B target runtime; no ambient builder configuration.

The caller owns the committed position. Verification writes a candidate block's
KV in place; only rows through the accepted prefix become live. Subsequent calls
overwrite rejected slots and causal attention must ignore every future slot.
"""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

from target_config import ABI_VERSION, HERE, FUSED, TargetConfig
from model_paths import TARGET_MODEL, default_target_cache, resolve_target_model


@dataclass
class Verification:
    logits: np.ndarray
    wall_seconds: float
    device_wait_seconds: float


class TargetDecoder:
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
        from decode_dynseq import DynseqInstsGen, make_insts_states, patch_insts
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
            from weight_io import upload_weights

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

    def restore_kv(self, packed):
        packed = np.asarray(packed)
        if packed.shape != self.kv_shape or packed.dtype != np.uint16:
            raise ValueError(
                (
                    "Expected packed BF16 words",
                    self.kv_shape,
                    packed.shape,
                    packed.dtype,
                )
            )
        self.bo["kv"].write(np.ascontiguousarray(packed).view(np.int16).reshape(-1), 0)
        self.bo["kv"].sync(self.TO)

    def read_kv(self, length=None):
        """Copy only the requested prefix for validation or state inspection."""
        length = self.capacity if length is None else int(length)
        if not 0 <= length <= self.capacity:
            raise ValueError("KV read length exceeds capacity")
        self.bo["kv"].sync(self.FROM)
        view = np.frombuffer(self.bo["kv"].map(), np.uint16).reshape(self.kv_shape)
        return view[:, :, :length].copy()

    def seed_kv(self, k, v, position):
        groups = self.g["kv_groups"]
        width = self.g["kv_region_width"]
        if not 0 <= position <= self.capacity - self.batch:
            raise ValueError("Prefill leaves no room for a whole verification block")
        packed = np.zeros(self.kv_shape, np.uint16)
        for offset, array in ((0, k), (groups, v)):
            a = np.asarray(array)
            if a.shape != (self.layers, position, groups * width):
                raise ValueError(("Unexpected prefill KV shape", a.shape, position))
            bits = a if a.dtype == np.uint16 else a.astype(bfloat16).view(np.uint16)
            packed[:, offset : offset + groups, :position] = bits.reshape(
                self.layers, position, groups, width
            ).transpose(0, 2, 1, 3)
        self.restore_kv(packed)

    def append_context_kv(self, k, v, position):
        """Upload only changed context rows, preserving all other device KV.

        k and v have shape [layer, new position, KV hidden]. This is used to
        replace accepted draft-block rows with projections of target features.
        Each group is a contiguous span in the explicit cache ABI.
        """
        groups = self.g["kv_groups"]
        width = self.g["kv_region_width"]
        k = np.asarray(k)
        v = np.asarray(v)
        if (
            k.ndim != 3
            or k.shape != v.shape
            or k.shape[0] != self.layers
            or k.shape[2] != groups * width
        ):
            raise ValueError(("Invalid context KV", k.shape, v.shape))
        n = k.shape[1]
        if position < 0 or position + n > self.capacity:
            raise ValueError("Context KV upload exceeds capacity")
        for offset, array in ((0, k), (groups, v)):
            bits = (
                array
                if array.dtype == np.uint16
                else array.astype(bfloat16).view(np.uint16)
            )
            rows = bits.reshape(self.layers, n, groups, width).transpose(0, 2, 1, 3)
            for layer in range(self.layers):
                for group in range(groups):
                    start = (
                        (
                            (layer * 2 * groups + offset + group) * self.capacity
                            + position
                        )
                        * width
                        * 2
                    )
                    data = (
                        np.ascontiguousarray(rows[layer, group])
                        .view(np.int16)
                        .reshape(-1)
                    )
                    if data.nbytes:
                        self.bo["kv"].write(data, start)
                        self.bo["kv"].sync(self.TO, data.nbytes, start)

    def hidden(self, boundaries=None):
        if not self.config.taps:
            raise ValueError("This artifact has no layer-boundary capture")
        boundaries = (
            list(range(self.layers + 1)) if boundaries is None else list(boundaries)
        )
        if any(i < 0 or i > self.layers for i in boundaries):
            raise ValueError("Invalid layer boundary")
        stride = self.batch * self.hidden_size
        out = np.empty((len(boundaries), self.batch, self.hidden_size), np.uint16)
        for j, i in enumerate(boundaries):
            self.bo["x"].sync(self.FROM, stride * 2, i * stride * 2)
            out[j] = np.frombuffer(
                self.bo["x"].map(), np.uint16, count=stride, offset=i * stride * 2
            ).reshape(self.batch, self.hidden_size)
        return out

    def verify(self, tokens, position):
        ids = np.asarray(tokens, dtype=np.int64)
        if (
            ids.shape != (self.batch,)
            or np.any(ids < 0)
            or np.any(ids >= self.vocab_size)
        ):
            raise ValueError("Verification requires one valid token per block position")
        if not 0 <= position <= self.capacity - self.batch:
            raise ValueError("Verification block exceeds artifact capacity")
        start = time.perf_counter()
        length = position + 1
        if self._last_length != length:
            self.patch_insts(self.insts, length, self.xrt, self.TO)
            self._last_length = length
        # Generate the shared single-theta cos/sin once per position, then
        # broadcast it across layers. The existing helper defines the BF16 bits.
        for row in range(self.batch):
            self.rope[:, row, :128] = self.weights.generate_rope_lut(position + row)
        self.bo["r"].write(
            self.rope.view(np.int16).reshape(-1), self.offset["rope"] * 2
        )
        self.bo["r"].sync(self.TO, self.rope.nbytes, self.offset["rope"] * 2)
        x = np.ascontiguousarray(self.embed[ids])
        self.bo["x"].write(x.view(np.int16).reshape(-1), 0)
        self.bo["x"].sync(self.TO, x.nbytes, 0)
        dispatched = time.perf_counter()
        state = self.kernel(
            3,
            self.insts["ib"],
            self.insts["size"],
            *(self.bo[n] for n in ("x", "w", "r", "y", "kv")),
            *([length] if self.config.dynamic_context else []),
        ).wait(self.timeout_ms)
        device = time.perf_counter() - dispatched
        if not str(state).endswith("COMPLETED"):
            raise RuntimeError(f"Target dispatch failed at {position}: {state}")
        count = self.g["vocab_chunks"] * self.batch * self.g["vocab_chunk_width"]
        off = self.offset["logits"] * 2
        self.bo["y"].sync(self.FROM, count * 2, off)
        logits = np.frombuffer(self.bo["y"].map(), bfloat16, count=count, offset=off)
        logits = logits.reshape(
            self.g["vocab_chunks"], self.batch, self.g["vocab_chunk_width"]
        )
        logits = (
            logits.transpose(1, 0, 2)
            .reshape(self.batch, -1)[:, : self.vocab_size]
            .astype(np.float32)
        )
        return Verification(logits, time.perf_counter() - start, device)

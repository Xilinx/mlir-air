# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Manifest-driven 8B target and drafter runtime; no ambient builder configuration.

The caller owns the committed position. Verification writes a candidate block's
KV in place; only rows through the accepted prefix become live. Subsequent calls
overwrite rejected slots and causal attention must ignore every future slot.
"""

from __future__ import annotations
from paths import HERE, COMMON
import hashlib
import json
import mmap
import os
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
        for path in (FUSED,):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from common.decode_dynseq import DynseqInstsGen, make_insts_states, patch_insts
        from decode_staircase import open_windows
        import target_weights as weights

        self.weights = weights
        self.rope_row = lambda position: weights.rope_row(position, self.config.role)
        self.gen = DynseqInstsGen(str(self.artifacts), max_L=self.capacity)
        if self.gen.calibrated_windows() != [self.capacity]:
            raise ValueError("Template capacity disagrees with manifest")
        self.dev = xrt.device(0)
        self._windows = open_windows(self.dev, xrt, self.gen, [self.capacity])
        self.context, self.kernel = self._windows[self.capacity]
        self.TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        from common.buffers import BufferView

        physical = self.interface["physical_abi"]
        if physical["version"] != 1:
            raise ValueError("Unsupported physical buffer ABI")
        self.dispatch_buffers = [
            xrt.bo(
                self.dev,
                v["bytes"],
                xrt.bo.host_only,
                self.kernel.group_id(v["argument"] + 3),
            )
            for v in physical["buffers"]
        ]

        def view(argument):
            v = physical["logical_views"][str(argument)]
            return BufferView(
                self.dispatch_buffers[v["buffer"]], v["offset_bytes"], v["bytes"]
            )

        self.bo = {name: view(i) for i, name in enumerate(("x", "w", "r", "y", "kv"))}
        self.w_bos = [
            view(item["argument"]) for item in self.interface["weight_buffers"]
        ]
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
                raise ValueError("The 8B target requires a BF16 embedding tensor")
            self.embed = self.model._raw_i16(name).view(bfloat16)
            final_norm = self.model.bf16("model.norm.weight")
        else:
            embed = self.model.bf16("model.embed_tokens.weight")
            final_norm = self.model.bf16("model.norm.weight")
            self.embed = np.asarray(embed, dtype=bfloat16)
        self.embed = self.embed.reshape(-1, self.hidden_size)
        self.qk = [self.model.layer_qk_norm(i) for i in range(self.layers)]
        self.rope = np.empty(
            (self.layers, self.batch, self.g["rope_width"]), dtype=bfloat16
        )
        # q/k normalization is constant across positions.
        for i, norms in enumerate(self.qk):
            if norms is not None:
                q, k = norms
                self.rope[i, :, 128:256] = q
                self.rope[i, :, 256:384] = k
        packed_cache = (
            Path(packed_cache or default_target_cache()).expanduser().resolve()
        )
        if not (packed_cache / "manifest.json").is_file():
            raise FileNotFoundError(
                f"Missing packed cache: {packed_cache}; run make weights"
            )
        receipt = json.loads((packed_cache / "manifest.json").read_text())
        if receipt["status"] != "prepared" or receipt["layers"] < self.layers:
            raise ValueError("Cache layer count/status differs from compiled model")
        # mmap and upload bounded slices: the 8B cache exceeds 4 GiB and must
        # never be copied into a single Python array or a single XRT buffer.
        w = np.load(packed_cache / "W.npy", mmap_mode="r")
        layer_words, head_words = receipt["layer_words"], receipt["head_words"]
        if w.size != receipt["layers"] * layer_words + head_words:
            raise ValueError("Packed weight geometry mismatch")
        body_words = self.layers * layer_words
        if body_words + head_words != self.n["w"]:
            raise ValueError("Artifact and packed-cache layouts differ")
        cursor = 0
        for bo, item in zip(self.w_bos, self.interface["weight_buffers"]):
            words = item["elements"]
            # Logical groups can contain body, head, or both (small builds).
            # Select ranges without concatenating multi-gigabyte arrays.
            source_spans = (
                (0, body_words, 0),
                (body_words, body_words + head_words, w.size - head_words),
            )
            for lo, hi, source_start in source_spans:
                begin, end = max(cursor, lo), min(cursor + words, hi)
                for start in range(begin, end, 8 * 1024 * 1024):
                    stop = min(end, start + 8 * 1024 * 1024)
                    data = w[source_start + start - lo : source_start + stop - lo].view(
                        np.int16
                    )
                    bo.write(data, (start - cursor) * 2)
            bo.sync(self.TO)
            cursor += words
        # The resident device BOs now own these weights. Release the mapped
        # packing-cache pages before allocating prefill scratch/other models.
        w._mmap.madvise(mmap.MADV_DONTNEED)
        with (packed_cache / "W.npy").open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if cursor != self.n["w"]:
            raise ValueError("Incomplete weight upload")
        norms = (
            np.load(packed_cache / "RMS.npy", mmap_mode="r")[: self.layers]
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

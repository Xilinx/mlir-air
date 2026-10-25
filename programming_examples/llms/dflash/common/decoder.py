# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Manifest-bound decoder state, dispatch and outputs shared by model mappings.

Model adapters initialize the weights and physical BOs. Every adapter exposes
logical x/r/y/KV views and an ordered dispatch_buffers list. The caller owns
the committed position; future KV slots can be overwritten but never observed.
"""

from dataclasses import dataclass
import time
import numpy as np
from ml_dtypes import bfloat16


@dataclass
class Verification:
    logits: np.ndarray
    wall_seconds: float
    device_wait_seconds: float


class DecoderState:
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
            self.rope[:, row, :128] = (
                self.rope_row(position + row)
                if hasattr(self, "rope_row")
                else self.weights.generate_rope_lut(position + row)
            )
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
            *self.dispatch_buffers,
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

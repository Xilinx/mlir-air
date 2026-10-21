# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Execution of newly committed context rows in blocks of eight."""

import time
import numpy as np


class NPUContextPrepass:
    def chunk(self, taps, positions, *, capture=False):
        taps = np.asarray(taps, np.float32)
        positions = np.asarray(positions, np.int64)
        n = len(positions)
        if not 1 <= n <= 8 or taps.shape != (n, 12800):
            raise ValueError(
                ("Expected one to eight prepass rows", taps.shape, positions.shape)
            )
        if np.any(positions < 0) or np.any(positions >= self.capacity):
            raise ValueError("Prepass positions outside capacity")
        begin = time.perf_counter()
        self.maps[0].fill(0)
        self.maps[0][:n] = taps
        self.maps[4][:, 640:].fill(0)
        self.maps[4][:n, 640:] = self.rope[positions]
        for i in (0, 4):
            self.bos[i].sync(self.to)
        dispatch = time.perf_counter()
        run = self.xrt.run(self.backend.kernel)
        for i, bo in enumerate(self.bos):
            run.set_arg(i, bo)
        run.start()
        state = run.wait(10000)
        if not str(state).endswith("COMPLETED"):
            raise RuntimeError(f"Prepass dispatch failed: {state}")
        returned = time.perf_counter()
        for i in ((5, 6, 7, 8) if capture else (6, 7, 8)):
            self.bos[i].sync(self.fro)
        features = self.maps[6][:n].copy()
        keys = self.maps[8][:n].reshape(n, 5, 1024).transpose(1, 0, 2).copy()
        values = (
            self.maps[7][:n].reshape(n, 5, 2048)[:, :, 1024:].transpose(1, 0, 2).copy()
        )
        if capture:
            self.last_raw = {
                "fused": self.maps[5][:n].copy(),
                "raw_kv": self.maps[7][:n].copy(),
            }
        end = time.perf_counter()
        self.last_calls.append(
            dict(
                rows=n,
                write_ms=(dispatch - begin) * 1000,
                dispatch_ms=(returned - dispatch) * 1000,
                read_ms=(end - returned) * 1000,
                wall_ms=(end - begin) * 1000,
            )
        )
        return features, keys, values

    def run(self, taps, positions):
        taps = np.asarray(taps, np.float32)
        positions = np.asarray(positions, np.int64)
        if taps.shape != (len(positions), 12800) or positions.ndim != 1:
            raise ValueError("Target taps and absolute positions disagree")
        self.last_calls = []
        if not len(positions):
            return (
                np.empty((0, 2560), np.float32),
                np.empty((5, 0, 1024), np.float32),
                np.empty((5, 0, 1024), np.float32),
            )
        chunks = [
            self.chunk(taps[i : i + 8], positions[i : i + 8])
            for i in range(0, len(positions), 8)
        ]
        return tuple(
            np.concatenate([c[j] for c in chunks], axis=0 if j == 0 else 1)
            for j in range(3)
        )

    def warmup(self):
        for _ in range(3):
            self.run(np.zeros((8, 12800), np.float32), np.arange(8))

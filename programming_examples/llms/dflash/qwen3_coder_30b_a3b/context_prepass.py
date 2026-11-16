# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The drafter's context K and V of committed target taps, on the host.

While the target verifies a block, each tap layer's rows are projected as
they arrive, inside the verify dispatch's wait; run() then returns the rows
the block committed. Rows without that (the prompt's) are computed in run().

  python3 context_prepass.py --output DIR   # record the prepass for a build
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import paths  # noqa: F401
from draft_weights import LAYERS, D, DH, NKV, THETA
import model_paths


def _rms(x, g, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


class ContextPrepass:
    def __init__(self, artifacts, reference, capacity):
        import torch

        self.torch = torch
        self.capacity = capacity
        f = lambda n: torch.from_numpy(  # noqa: E731
            np.asarray(reference.tensor(n), np.float32)
        )
        self.taps = len(reference.tap_slots)
        self.fc, self.hn = f("fc.weight"), f("hidden_norm.weight").numpy()
        # fc by tap, transposed: [tap][D in, D out]
        self.fcs = [
            self.fc[:, i * D : (i + 1) * D].T.contiguous() for i in range(self.taps)
        ]
        self.kv = [
            torch.cat(
                [f(f"layers.{l}.self_attn.{n}_proj.weight") for n in "kv"]
            ).T.contiguous()
            for l in range(LAYERS)
        ]
        self.kn = [
            f(f"layers.{l}.self_attn.k_norm.weight").numpy() for l in range(LAYERS)
        ]
        self.pre = None
        from target_runtime import loaded_target

        target = loaded_target()
        if target is not None:
            target.tap_listeners.append(self._on_tap)

    def warmup(self):
        self.run(np.zeros((1, self.taps * D), np.float32), np.arange(1))

    def _layers_kv(self, c, p0, layers):
        """K and V [n, 1024] of each of layers for the normed rows c [n, D]
        at p0 .., each KV head twice as the drafter's build stores them."""
        n = len(c)
        inv = 1.0 / (THETA ** (np.arange(0, DH, 2) / DH))
        f = (p0 + np.arange(n))[:, None, None] * inv
        cos, sin = np.cos(f), np.sin(f)
        K, V = [], []
        for l in layers:
            kv = (c @ self.kv[l]).numpy().reshape(n, 2, NKV, DH)
            k = _rms(kv[:, 0], self.kn[l])
            k1, k2 = k[..., : DH // 2], k[..., DH // 2 :]
            k = np.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], -1)
            K.append(np.repeat(k, 2, 1).reshape(n, -1))
            V.append(np.repeat(kv[:, 1], 2, 1).reshape(n, -1))
        return K, V

    def _normed(self, c):
        c = _rms(np.asarray(c), self.hn).astype(bfloat16).astype(np.float32)
        return self.torch.from_numpy(c)

    def run(self, taps, positions):
        """(None, K, V [layers, n, 1024]) of taps [n, taps x D] at positions."""
        n = len(positions)
        p0 = int(positions[0]) if n else 0
        pre, self.pre = self.pre, None
        if pre is not None and pre[0] == p0 and pre[1].shape[1] >= n:
            return None, pre[1][:, :n], pre[2][:, :n]
        t = self.torch.from_numpy(np.ascontiguousarray(taps, np.float32))
        K, V = self._layers_kv(self._normed((t @ self.fc.T).numpy()), p0, range(LAYERS))
        return None, np.stack(K), np.stack(V)

    # A verified block's K and V, in pieces a dispatch's wait can hold: a
    # tap's share of fc as it arrives, then the norm and the layers' K and V.
    def _on_tap(self, i, rows, position):
        work = [lambda: self._tap(i, rows)]
        if i == self.taps - 1:
            work += [lambda h=h: self._finish(position, h) for h in (0, 1)]
        return work

    def _tap(self, i, rows):
        part = self.torch.from_numpy(np.ascontiguousarray(rows, np.float32))
        part = part @ self.fcs[i]
        self.acc = part if i == 0 else self.acc + part

    def _finish(self, p0, half):
        if half == 0:
            self.c_pre, self.kv_pre = self._normed(self.acc.numpy()), ([], [])
        h = LAYERS // 2
        K, V = self._layers_kv(self.c_pre, p0, range(h * half, h * half + h))
        self.kv_pre[0].extend(K)
        self.kv_pre[1].extend(V)
        if half == 1:
            self.pre = (p0, np.stack(self.kv_pre[0]), np.stack(self.kv_pre[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        status="host",
        drafter=model_paths.DRAFT_MODEL,
        revision=model_paths.DRAFT_REVISION,
        layers=LAYERS,
        source="context_prepass.py",
    )
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"host context prepass recorded in {a.output}")


if __name__ == "__main__":
    main()

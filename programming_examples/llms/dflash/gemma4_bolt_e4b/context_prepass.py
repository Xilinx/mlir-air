# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The drafter's context K/V from the target's features, on the host.

For each committed position: the five tap boundaries, concatenated, through
the draft's fc and hidden norm; then every draft layer's K projection, k
norm and rope, and V projection (draft_cpu.Draft.add_context's arithmetic,
its matrix products in BF16).
"""

import numpy as np

from draft_cpu import rms, rope


class ContextPrepass:
    """`reference` is the draft_weights.DraftWeights the drafter was loaded
    with; no device program is involved, so `artifacts` is unused."""

    def __init__(self, artifacts, reference, capacity):
        import torch

        self.capacity = capacity
        d = self.draft = reference.draft
        bf = lambda w: torch.from_numpy(w).to(torch.bfloat16)  # noqa: E731
        self.fc = bf(d.fc)
        self.kv = bf(
            np.concatenate(
                [np.concatenate([lw["attn_k"], lw["attn_v"]]) for lw in d.layers]
            )
        )

    def warmup(self):
        self.run(np.zeros((1, self.fc.shape[1]), np.float32), np.zeros(1, np.int64))

    def run(self, taps, positions):
        """taps [n, 5 x hidden] at `positions` -> (None, k, v), k and v
        [draft layers, n, KV heads x head dim]."""
        import torch

        d, n = self.draft, len(taps)
        x = torch.from_numpy(np.ascontiguousarray(taps, np.float32))
        feat = (x.to(torch.bfloat16) @ self.fc.T).float().numpy()
        feat = rms(feat, d.hidden_norm, d.eps)
        kv = (torch.from_numpy(feat).to(torch.bfloat16) @ self.kv.T).float().numpy()
        kv = kv.reshape(n, d.n_layers, 2, d.kv_heads, d.dh)
        pos = np.asarray(positions)
        k = np.stack(
            [
                rope(rms(kv[:, i, 0], lw["attn_k_norm"], d.eps), pos, d.theta)
                for i, lw in enumerate(d.layers)
            ]
        ).reshape(d.n_layers, n, -1)
        v = kv[:, :, 1].reshape(n, d.n_layers, -1).transpose(1, 0, 2)
        return None, k, np.ascontiguousarray(v)


def main():
    """Record the prepass in a build directory, beside the device programs:
    it has no program of its own, so the manifest names its source."""
    import argparse
    import hashlib
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve()
    manifest = dict(
        status="compiled",
        backend="host",
        source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    )
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

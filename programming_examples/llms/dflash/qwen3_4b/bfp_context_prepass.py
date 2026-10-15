# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Standalone prepass with official BF16 weights and standard BFP matrix math."""

import hashlib
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from context_prepass import ContextPrepass, rmsnorm
from draft_weights import DraftWeights
from npu_context_prepass import NPUContextPrepass


class BF16ContextReference(ContextPrepass):
    """CPU reference: BF16 checkpoint weights, BF16 normalized features.

    BLAS accumulates in FP32. This reference measures BFP projection error;
    it is not an emulator for block-floating-point matrix instructions.
    """

    def __init__(self, weights=None):
        self.weights = weights or DraftWeights()
        self.eps = float(self.weights.config["rms_norm_eps"])
        self.theta = float(self.weights.config["rope_theta"])
        self.fc = self.weights.tensor("fc.weight").astype(np.float32)
        self.hidden_norm = self.weights.tensor("hidden_norm.weight").astype(np.float32)
        self.k_norm = [
            self.weights.layer_qk_norm(i)[1].astype(np.float32) for i in range(5)
        ]
        self.kv_weights = [
            np.concatenate(
                [
                    self.weights.tensor(
                        f"layers.{i}.self_attn.{kind}_proj.weight"
                    ).astype(np.float32)
                    for kind in ("k", "v")
                ],
                axis=0,
            )
            for i in range(5)
        ]

    def features(self, taps):
        taps = np.asarray(taps, bfloat16).astype(np.float32)
        fused = taps @ self.fc.T
        return (
            rmsnorm(fused, self.hidden_norm, self.eps)
            .astype(bfloat16)
            .astype(np.float32)
        )


def pack_bfp_weights(weight, k_tile, n_tile):
    """[N,K] -> [Ntile,Ktile,N/8,K/8,Kinner,Ninner] BF16 operand layout."""
    weight = np.asarray(weight, bfloat16)
    n, k = weight.shape
    if n % n_tile or k % k_tile or n_tile % 8 or k_tile % 8:
        raise ValueError(("Invalid projection tile", weight.shape, k_tile, n_tile))
    return np.ascontiguousarray(
        weight.reshape(
            n // n_tile, n_tile // 8, 8, k // k_tile, k_tile // 8, 8
        ).transpose(0, 3, 1, 4, 5, 2)
    ).reshape(-1)


class BFPContextPrepass(NPUContextPrepass):
    def __init__(self, artifacts, reference=None, capacity=4096):
        import pyxrt as xrt
        from air.backend.xrt import XRTBackend, XRTCompileArtifact

        reference = reference or BF16ContextReference()
        # Projection weights become resident BOs below; retain no duplicate
        # FP32 CPU-reference matrices after initialization.
        self.artifacts = Path(artifacts).resolve()
        manifest = json.loads((self.artifacts / "manifest.json").read_text())
        if (
            manifest.get("interface") != "bfp-prepass-v1"
            or manifest["status"] != "compiled"
        ):
            raise ValueError("Expected compiled BF16/BFP prepass artifact")
        for name, digest in manifest["artifacts"].items():
            if (
                hashlib.sha256((self.artifacts / name).read_bytes()).hexdigest()
                != digest
            ):
                raise ValueError(("Artifact hash mismatch", name))
        if reference.eps != 1e-6:
            raise ValueError("Epsilon mismatch")
        self.capacity = capacity
        self.weights = reference.weights
        self.eps, self.theta = reference.eps, reference.theta
        self.xrt = xrt
        self.backend = XRTBackend(**manifest["backend"])
        self.backend.load(
            XRTCompileArtifact(
                str(self.artifacts / "prepass.elf"), "main:context_prepass", None
            )
        )
        self.to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.fro = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.shapes = [
            (8, 12800),
            (12800 * 2560,),
            (3200,),
            (2560 * 10240,),
            (8, 768),
            (8, 2560),
            (8, 2560),
            (8, 10240),
            (8, 5120),
        ]
        dtypes = [
            bfloat16,
            bfloat16,
            np.float32,
            bfloat16,
            np.float32,
            np.float32,
            bfloat16,
            np.float32,
            np.float32,
        ]
        self.bos = [
            xrt.ext.bo(self.backend.device, int(np.prod(s)) * np.dtype(t).itemsize)
            for s, t in zip(self.shapes, dtypes)
        ]
        self.maps = [
            np.frombuffer(b.map(), t, count=int(np.prod(s))).reshape(s)
            for b, s, t in zip(self.bos, self.shapes, dtypes)
        ]
        k, n = manifest["projection_k"], manifest["projection_n"]
        self.maps[1][:] = pack_bfp_weights(reference.fc, k, n)
        self.maps[2][:] = np.concatenate([reference.hidden_norm, *reference.k_norm])
        self.maps[3][:] = pack_bfp_weights(
            np.concatenate(reference.kv_weights, axis=0), k, n
        )
        for i in (1, 2, 3):
            self.bos[i].sync(self.to)
        inv = 1.0 / (self.theta ** (np.arange(64, dtype=np.float64) / 64))
        angles = np.arange(capacity, dtype=np.float64)[:, None] * inv[None, :]
        self.rope = np.concatenate([np.cos(angles), np.sin(angles)], axis=1).astype(
            np.float32
        )
        self.maps[4][:, :640] = np.concatenate(reference.k_norm)
        self.last_calls = []
        self.last_raw = None

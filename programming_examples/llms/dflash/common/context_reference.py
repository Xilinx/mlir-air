# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""CPU arithmetic helpers for checking the NPU context-preparation program."""

import numpy as np


def rmsnorm(x, weight, eps):
    x = np.asarray(x, np.float32)
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * np.asarray(
        weight, np.float32
    )


def rope(x, positions, theta=1_000_000):
    """Half-split RoPE on [position, KV head, dimension], FP32 CPU reference."""
    half = x.shape[-1] // 2
    inv = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
    angle = np.asarray(positions, np.float64)[:, None] * inv[None, :]
    cos = np.cos(angle).astype(np.float32)[:, None, :]
    sin = np.sin(angle).astype(np.float32)[:, None, :]
    left, right = x[..., :half], x[..., half:]
    return np.concatenate((left * cos - right * sin, left * sin + right * cos), axis=-1)


class ContextPrepass:
    """Reference operations shared by the BF16 checkpoint reference."""

    def from_features(self, features, positions):
        features = np.asarray(features, np.float32)
        if features.shape != (len(positions), self.fc.shape[0]):
            raise ValueError("Features and absolute positions disagree")
        keys, values = [], []
        for layer, weight in enumerate(self.kv_weights):
            kv = features @ weight.T
            k = rmsnorm(kv[:, :1024].reshape(-1, 8, 128), self.k_norm[layer], self.eps)
            keys.append(rope(k, positions, self.theta).reshape(-1, 1024))
            values.append(kv[:, 1024:])
        return np.stack(keys), np.stack(values)

    def run(self, taps, positions):
        features = self.features(taps)
        k, v = self.from_features(features, positions)
        return features, k, v

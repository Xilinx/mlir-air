# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The z-lab drafter on the shared DFlash B8 decoder: all 8 layers and the head
# in one dispatch. Each of its 4 KV heads is stored twice, which gives the
# decoder's 8-head attention. The context's K and V are computed on the host.
import json
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
TOK, DH, NKV, THETA = 8, 128, 4, 1e7


def _rms(x, g, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def rope_row(p):
    inv = 1.0 / (THETA ** (np.arange(0, DH // 2, dtype=np.float64) / (DH // 2)))
    a = p * inv
    return np.concatenate([np.cos(a), np.sin(a)]).astype(bfloat16).astype(np.float32)


class _Spin:
    """A kernel whose runs poll for completion: a blocked host wakes late and
    clocks down."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, *args):
        return _SpinRun(self.kernel(*args))

    def __getattr__(self, name):
        return getattr(self.kernel, name)


class _SpinRun:
    def __init__(self, run):
        self.run = run

    def wait(self, timeout_ms):
        done = __import__("pyxrt").ert_cmd_state.ERT_CMD_STATE_COMPLETED
        end = time.perf_counter() + timeout_ms / 1e3
        while self.run.state() != done and time.perf_counter() < end:
            pass
        return self.run.wait(timeout_ms)


class Drafter:
    """artifacts: the drafter's build (build_target.py --role drafter);
    packed: its weights (prepare_coder_draft.py); draft_path: the z-lab
    checkpoint."""

    def __init__(self, w, draft_path, artifacts, packed):
        import torch

        sys.path[:0] = [str(HERE), str(HERE.parent)]
        # the decoder's package is also named common: hold the decode's module
        # aside while it loads
        held = sys.modules.pop("common", None)
        try:
            self._init(torch, w, draft_path, artifacts, packed)
        finally:
            if held is not None:
                sys.modules["common"] = held

    def _init(self, torch, w, draft_path, artifacts, packed):
        import paths  # noqa: F401
        from common.decoder_8b import TargetDecoder
        from prepare_coder_draft import Safetensors

        self.torch = torch
        st = Safetensors(Path(draft_path) / "model.safetensors")
        cfg = json.loads((Path(draft_path) / "config.json").read_text())
        self.mask = cfg["dflash_config"]["mask_token_id"]
        self.taps = cfg["dflash_config"]["target_layer_ids"]
        t = lambda n: torch.from_numpy(
            np.asarray(st.tensor(n), np.float32)
        )  # noqa: E731
        self.fc, self.hn = t("fc.weight"), t("hidden_norm.weight").numpy()
        # fc by tap, transposed: [5][2048 in, 2048 out]
        self.fcs = [
            self.fc[:, i * 2048 : (i + 1) * 2048].T.contiguous() for i in range(5)
        ]
        self.kv = [
            torch.cat(
                [t(f"layers.{l}.self_attn.{n}_proj.weight") for n in "kv"]
            ).T.contiguous()
            for l in range(8)
        ]
        self.kn = [t(f"layers.{l}.self_attn.k_norm.weight").numpy() for l in range(8)]
        D = 2048

        class Provider:
            def bf16(self, name):
                if name == "model.embed_tokens.weight":
                    return np.asarray(w.raw(name)).view(bfloat16).reshape(-1, D)
                if name == "model.norm.weight":
                    return np.asarray(st.tensor("norm.weight"), bfloat16)
                raise KeyError(name)

            def layer_qk_norm(self, l):
                return tuple(
                    np.asarray(
                        st.tensor(f"layers.{l}.self_attn.{k}_norm.weight"), bfloat16
                    )
                    for k in "qk"
                )

        self.dec = TargetDecoder(
            artifacts, packed_cache=packed, model_weights=Provider()
        )
        self.dec.rope_row = rope_row
        self.dec.kernel = _Spin(self.dec.kernel)
        self.times = dict.fromkeys(("kv", "upload", "dispatch"), 0.0)

    def _kv(self, taps, p0):
        """Rows' K and V [8 layers, n, 1024] for taps [n, 5 D] at p0 .."""
        torch = self.torch
        c = (
            torch.from_numpy(np.ascontiguousarray(taps, np.float32)) @ self.fc.T
        ).numpy()
        c = torch.from_numpy(_rms(c, self.hn).astype(bfloat16).astype(np.float32))
        n = len(c)
        inv = 1.0 / (THETA ** (np.arange(0, DH, 2) / DH))
        f = (p0 + np.arange(n))[:, None, None] * inv
        cos, sin = np.cos(f), np.sin(f)
        K, V = [], []
        for l in range(8):
            kv = (c @ self.kv[l]).numpy().reshape(n, 2, NKV, DH)
            k = _rms(kv[:, 0], self.kn[l])
            k1, k2 = k[..., : DH // 2], k[..., DH // 2 :]
            k = np.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], -1)
            K.append(np.repeat(k, 2, 1).reshape(n, -1))
            V.append(np.repeat(kv[:, 1], 2, 1).reshape(n, -1))
        return np.stack(K), np.stack(V)

    # A verified block's K and V, in pieces a dispatch's wait can hold: a
    # tap's share of fc as it arrives, then the norm and the layers' K and V.
    def tap(self, i, rows):
        part = self.torch.from_numpy(np.ascontiguousarray(rows, np.float32))
        part = part @ self.fcs[i]
        self.acc = part if i == 0 else self.acc + part

    def finish(self, p0, half):
        if half == 0:
            c = _rms(self.acc.numpy(), self.hn).astype(bfloat16).astype(np.float32)
            self.c_pre, self.kv_pre = self.torch.from_numpy(c), ([], [])
        c, n = self.c_pre, len(self.c_pre)
        inv = 1.0 / (THETA ** (np.arange(0, DH, 2) / DH))
        f = (p0 + np.arange(n))[:, None, None] * inv
        cos, sin = np.cos(f), np.sin(f)
        for l in range(4 * half, 4 * half + 4):
            kv = (c @ self.kv[l]).numpy().reshape(n, 2, NKV, DH)
            k = _rms(kv[:, 0], self.kn[l])
            k1, k2 = k[..., : DH // 2], k[..., DH // 2 :]
            k = np.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], -1)
            self.kv_pre[0].append(np.repeat(k, 2, 1).reshape(n, -1))
            self.kv_pre[1].append(np.repeat(kv[:, 1], 2, 1).reshape(n, -1))
        if half == 1:
            self.pre = (p0, np.stack(self.kv_pre[0]), np.stack(self.kv_pre[1]))

    def context(self, taps, p0):
        """The committed rows' taps [n, 5 D] at p0 ..: their K and V into the cache."""
        t0 = time.perf_counter()
        pre, self.pre = getattr(self, "pre", None), None
        n = len(taps)
        if pre and pre[0] == p0 and pre[1].shape[1] >= n:
            K, V = pre[1][:, :n], pre[2][:, :n]
        else:
            K, V = self._kv(taps, p0)
        t1 = time.perf_counter()
        self.dec.append_context_kv(K, V, p0)
        self.times["kv"] += t1 - t0
        self.times["upload"] += time.perf_counter() - t1

    def propose(self, token, start):
        t0 = time.perf_counter()
        r = self.dec.verify([token] + [self.mask] * (TOK - 1), start)
        self.times["dispatch"] += time.perf_counter() - t0
        return r.logits[1:].argmax(1).tolist()

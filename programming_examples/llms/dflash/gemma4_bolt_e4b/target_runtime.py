# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The Gemma4-E4B eight-token decoder (build_target.py's artifact) as the
shared DFlash loop's target, and the factory load_pair calls for both roles.

verify(ids, p) runs tokens ids[0..7] at positions p..p+7 over the KV cache's
first p positions, appends their K/V and returns every token's logits (no
softcap). hidden() reads layer boundaries, rescaled from the device's
residual (weights_b8.residual_scales) to the model's.

A prompt's K/V travels between prefill, decoders and saved benchmark states
as [KV owner layer, position, KV_ROW]: each owning layer's K (or V) row, its
real width (KV heads x that layer's head dim) first and zeros after.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

import b8_config
import weights_b8 as W
from model_paths import TARGET_MODEL, resolve_target_model

gw = W.gw
KV_ROW = b8_config.KV_HEADS * b8_config.DH


def pack_prefill_kv(ks, vs, n):
    """Per-layer lists of [n, KV heads * head dim] -> two [owners, n, KV_ROW]."""
    out = []
    for xs in (ks, vs):
        a = np.zeros((b8_config.KV_OWNERS, n, KV_ROW), np.float32)
        for L in range(b8_config.KV_OWNERS):
            row = np.asarray(xs[L][:n], np.float32)
            a[L, :, : row.shape[1]] = row
        out.append(a)
    return out


def unpack_prefill_kv(k, v):
    """The inverse: per-layer lists for the owning layers."""
    out = []
    for a in (k, v):
        a = np.asarray(a)
        a = a.view(bfloat16) if a.dtype == np.uint16 else a
        out.append(
            [
                np.asarray(a[L, :, : b8_config.KV_HEADS * gw.head_dim(L)], np.float32)
                for L in range(b8_config.KV_OWNERS)
            ]
        )
    return out


class GemmaTarget:
    def __init__(self, artifact, model=None, verbose=False):
        import pyxrt as xrt
        from air.backend.txn_builder import TxnBuilder

        self.xrt = xrt
        self.dir = Path(artifact)
        man = json.loads((self.dir / "manifest.json").read_text())
        self.abi = man["abi"]
        self.cfg = b8_config.Config(**man["config"])
        self.layers = self.cfg.layers
        self.batch, self.capacity = self.cfg.batch, self.cfg.capacity
        self.config = SimpleNamespace(
            role="target", causal=True, taps=True, **man["config"]
        )
        self.f = b8_config.load_layout(self.cfg)
        if model is None or isinstance(model, (str, Path)):
            model = gw.Q4nxModel(resolve_target_model(str(model or TARGET_MODEL)))
        self.model = model
        self.dev = xrt.device(0)
        xb = xrt.xclbin(str(self.dir / "decode.xclbin"))
        self.dev.register_xclbin(xb)
        self.ctx = xrt.hw_context(self.dev, xb.get_uuid())
        self.kern = xrt.kernel(self.ctx, "MLIR_AIE")
        self.tb = TxnBuilder(str(self.dir / "decode.txn.h"))
        (self.fn,) = self.tb.function_names
        ins = self.tb(self.fn, self.cfg.build_length)
        self.ib = xrt.bo(
            self.dev, ins.nbytes * 2, xrt.bo.cacheable, self.kern.group_id(1)
        )
        self.TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        el = self.abi["elements"]
        self.names = ("x", "w", "rms", "y", "kv", "pli")
        self.bo = {
            n: xrt.bo(self.dev, el[n] * 2, xrt.bo.host_only, self.kern.group_id(i + 3))
            for i, n in enumerate(self.names)
        }
        self._last_L = None
        self._load(verbose)

    def _write(self, name, arr, offset=0):
        a = np.ascontiguousarray(arr).view(np.int16).reshape(-1)
        self.bo[name].write(a, offset * 2)
        self.bo[name].sync(self.TO, a.nbytes, offset * 2)

    def _load(self, verbose):
        f, t0 = self.f, time.perf_counter()
        self.c = W.residual_scales(self.model, self.layers)
        for L in range(self.layers):
            self._write("w", W.pack_layer(f, self.model, L, self.c[L]), f.W_OFFS[L])
            self._write(
                "rms", W.norm_slab(f, self.model, L, self.c[L]), L * f.RMS_LAYER
            )
        off = f.W_OFFS[self.layers]
        for c in W.pack_lm_head(f, self.model):
            self._write("w", c, off)
            off += c.size
        self._write(
            "rms",
            np.asarray(self.model.globals()["final_norm"], np.float32).astype(bfloat16),
            self.abi["offsets"]["final_norm"],
        )
        self._write("kv", np.zeros(self.abi["elements"]["kv"], np.int16))
        self.pli = W.PerLayerInputs(self.model, self.layers)
        self.rope = W.RopeRows(self.model, self.layers)
        self.embed = self.pli.embed
        if verbose:
            print(f"target weights loaded in {time.perf_counter() - t0:.1f}s")

    def seed_kv(self, k, v, position):
        """A prompt's first `position` positions of K/V ([owners, n, KV_ROW],
        see the module docstring), into the decoder's spread head layout."""
        if not 0 <= position <= self.capacity - self.batch:
            raise ValueError("Prefill leaves no room for a whole verification block")
        ks, vs = unpack_prefill_kv(k, v)
        cap = self.capacity
        for L in range(min(self.layers, b8_config.KV_OWNERS)):
            dh = gw.head_dim(L)
            heads = b8_config.KV_HEADS
            kk = W._spread(ks[L][:position].T, heads, dh).T
            vv = W._spread(vs[L][:position].T, heads, dh, front=True).T
            base = L * self.abi["geometry"]["kv_layer"]
            self._write("kv", kk.astype(bfloat16), base)
            self._write("kv", vv.astype(bfloat16), base + cap * KV_ROW)

    def read_kv(self, length=None):
        """[slab, K or V, position, KV_ROW] BF16 words of the first `length`
        positions (the scratch slab of KV-shared layers' appends included)."""
        length = self.capacity if length is None else int(length)
        self.bo["kv"].sync(self.FROM)
        slabs = self.abi["geometry"]["kv_slabs"]
        view = np.frombuffer(self.bo["kv"].map(), np.uint16)
        view = view[: slabs * 2 * self.capacity * KV_ROW]
        view = view.reshape(slabs, 2, self.capacity, KV_ROW)
        return view[:, :, :length].copy()

    def verify(self, tokens, position):
        B, p = self.batch, int(position)
        ids = [int(t) for t in tokens]
        if len(ids) != B or not 0 <= p <= self.capacity - B:
            raise ValueError("Verification block exceeds the artifact")
        start = time.perf_counter()
        L = p + 1
        if L != self._last_L:
            ins = self.tb(self.fn, L)
            self.ib.write(ins, 0)
            self.ib.sync(self.TO)
            self.n_ins, self._last_L = ins.size, L
        emb = self.embed(ids) * gw.EMBED_SCALE
        self._write("x", emb.astype(bfloat16))
        self._write("pli", self.pli(ids, emb))
        self._write("rms", self.rope(range(p, p + B)), self.abi["offsets"]["rope"])
        dispatched = time.perf_counter()
        st = self.kern(
            3, self.ib, self.n_ins, *[self.bo[n] for n in self.names], L
        ).wait(30000)
        device = time.perf_counter() - dispatched
        if not str(st).endswith("COMPLETED"):
            raise RuntimeError(f"decoder dispatch at p={p}: {st}")
        g = self.abi["geometry"]
        n = g["vocab_chunks"] * B * g["vocab_chunk_width"]
        off = self.abi["offsets"]["logits"]
        self.bo["y"].sync(self.FROM, n * 2, off * 2)
        lg = np.frombuffer(self.bo["y"].map(), bfloat16, count=n, offset=off * 2)
        lg = lg.reshape(g["vocab_chunks"], B, g["vocab_chunk_width"])
        lg = lg.transpose(1, 0, 2).reshape(B, -1)[:, : g["vocab"]].astype(np.float32)
        return SimpleNamespace(
            logits=lg,
            wall_seconds=time.perf_counter() - start,
            device_wait_seconds=device,
        )

    def boundaries(self, which):
        """{boundary: [batch, hidden] float32} of the last verify call."""
        B, K = self.batch, self.f.K
        out = {}
        for b in which:
            o = b * B * K
            self.bo["x"].sync(self.FROM, B * K * 2, o * 2)
            x = np.frombuffer(self.bo["x"].map(), bfloat16, count=B * K, offset=o * 2)
            out[b] = x.astype(np.float32).reshape(B, K) * self.c[b]
        return out

    def hidden(self, boundaries):
        """BF16 words [len(boundaries), batch, hidden], as the shared loop
        reads its taps."""
        rows = self.boundaries(boundaries)
        return np.stack([rows[b] for b in boundaries]).astype(bfloat16).view(np.uint16)


def target_decoder(
    artifact, *, model_weights=None, packed_cache=None, model_source=None, **_
):
    """load_pair's decoder constructor: the target, or with model_weights
    (draft_weights.DraftWeights) the drafter."""
    if model_weights is None:
        return GemmaTarget(artifact, model_source)
    from draft_weights import GemmaDrafter

    return GemmaDrafter(artifact, model_weights, packed_cache)

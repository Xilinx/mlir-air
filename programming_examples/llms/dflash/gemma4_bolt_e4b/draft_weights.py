# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The Nemotron-3.5-Bolt-E4B DFlash drafter on the NPU: dflash/qwen3_4b's
B8 drafter graph (build_draft.py) with this checkpoint's weights.

The draft's layers come from NVIDIA's BF16 GGUF, requantized to the
projections' group-32 Q4. It has no embedding or LM head of its own and uses
the target's (the target package's int8 embedding rows and Q4 LM head).
"""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import paths  # noqa: F401  (dflash on sys.path)

# Last, so that this directory's modules win where qwen3_4b has one of the
# same name.
QWEN = Path(__file__).resolve().parents[1] / "qwen3_4b"
sys.path.append(str(QWEN))
QWEN_PKG = "air_examples.llms.dflash.qwen3_4b"

import build_draft  # noqa: E402,F401  (the draft geometry in target_config)
import target_config  # noqa: E402

import model_spec  # noqa: E402
from draft_cpu import Draft, rms  # noqa: E402
from model_paths import DRAFT_FILE, default_draft_snapshot  # noqa: E402
from weights_b8 import TokenRows, gw, host_weights  # noqa: E402


def _qwen_decoder():
    """qwen3_4b's TargetDecoder, imported with qwen3_4b's own model_paths
    (this directory has a module of that name too)."""
    own = sys.modules.get("model_paths")
    sys.modules["model_paths"] = importlib.import_module(f"{QWEN_PKG}.model_paths")
    try:
        return importlib.import_module(f"{QWEN_PKG}.target_runtime").TargetDecoder
    finally:
        if own is not None:
            sys.modules["model_paths"] = own
        else:
            del sys.modules["model_paths"]


def _q4(w, group):
    return host_weights.requant_q4k(np.asarray(w, np.float32), group)


def _file_id(path):
    """A file by content where the Hugging Face cache names its blob by hash,
    otherwise by path, size and modification time."""
    p = Path(path).resolve()
    st = p.stat()
    if "blobs" in p.parts:
        return dict(blob=p.name, size=st.st_size)
    return dict(path=str(p), size=st.st_size, mtime_ns=st.st_mtime_ns)


def _identity(weights):
    """What the packed cache depends on: the checkpoints it reads, the
    packing code (here and the shared functions it calls) and the drafter's
    geometry."""
    import inspect
    from gemma4_e2b_q4nx_requant import __file__ as requant

    fused = paths.EXAMPLES / "fused_decode"
    sources = [
        Path(build_draft.__file__),
        Path(target_config.__file__),
        Path(host_weights.__file__),
        Path(requant),
        fused / "fused_decode.py",
        fused / "proj_qmm_pack.py",
    ]
    code = inspect.getsource(_q4) + inspect.getsource(build_cache)
    return dict(
        packing=hashlib.sha256(code.encode()).hexdigest(),
        sources={
            str(p.relative_to(paths.EXAMPLES)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
        draft=_file_id(weights.snapshot / DRAFT_FILE),
        target=_file_id(weights.qm.path),
    )


def build_cache(draft, qm, path, identity):
    """Packed weights (the five layers, then the LM head) and the layers' two
    norms, in the drafter artifact's layout."""
    from gemma4_e2b_q4nx_requant import _interleave512

    cfg = target_config.TargetConfig(
        role="drafter", layers=draft.n_layers, causal=False
    )
    fd = target_config.load_geometry(cfg)
    G = fd.GROUP
    packed, rin, rpost = [], [], []
    for lw in draft.layers:
        phase = [None] * fd.NPH
        phase[0] = _q4(np.concatenate([lw["attn_q"], lw["attn_k"], lw["attn_v"]]), G)
        phase[fd.OPROJ_PHASE] = _q4(lw["attn_output"], G)
        phase[fd.GLU_PHASE] = _interleave512(
            _q4(lw["ffn_up"], G), _q4(lw["ffn_gate"], G), fd.GLU_CHUNK
        )
        phase[fd.DOWN_PHASE] = _q4(lw["ffn_down"], G)
        layer = np.concatenate(
            [
                fd.pack_q4k_cascade(
                    *phase[p], fd.NCX, fd.NCY, iter_major=True, dual_chan=True
                )
                for p in range(fd.NPH)
            ]
        )
        assert layer.size == fd.W_LAYER, (layer.size, fd.W_LAYER)
        packed.append(layer)
        rin.append(np.asarray(lw["attn_norm"], bfloat16))
        rpost.append(np.asarray(lw["ffn_norm"], bfloat16))
    VP, VPF = fd.VOCAB_SIZE_PADDED, fd.UNI_LM * fd.VOCAB_SIZE_PADDED
    lq = np.zeros((VPF, fd.K), np.uint8)
    ls = np.zeros((VPF, fd.K // G), np.float32)
    lmn = np.zeros((VPF, fd.K // G), np.float32)
    for r0 in range(0, gw.VOCAB, 16384):
        r1 = min(r0 + 16384, gw.VOCAB)
        lq[r0:r1], ls[r0:r1], lmn[r0:r1] = qm.lm_head_q4nx(r0, r1)
    for c in range(fd.UNI_LM):
        sl = slice(c * VP, (c + 1) * VP)
        packed.append(
            fd.pack_q4k_cascade(
                lq[sl], ls[sl], lmn[sl], fd.NCX, fd.NCY, iter_major=True, dual_chan=True
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez(
        partial,
        W=np.concatenate(packed).view(np.int16),
        RMS_in=np.stack(rin).view(np.int16),
        RMS_post=np.stack(rpost).view(np.int16),
    )
    partial.replace(path)
    path.with_suffix(".json").write_text(json.dumps(identity, indent=2) + "\n")
    return path


def packed_cache(directory, weights):
    """The packed cache in `directory` for these weights and the current
    packing code, built first if absent; the file name carries their hash."""
    identity = _identity(weights)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    path = Path(directory) / f"packed-{key.hexdigest()[:16]}.npz"
    if path.exists():
        return path
    return build_cache(weights.draft, weights.qm, path, identity)


class DraftWeights:
    """The drafter's checkpoint. Without target_model only its metadata
    (taps and mask token) is read."""

    tap_slots = list(model_spec.TAP_SLOTS)
    mask_token_id = model_spec.MASK_TOKEN_ID

    def __init__(self, target_model=None, snapshot=None, shared_embedding=None):
        self.snapshot = Path(snapshot or default_draft_snapshot())
        self.qm = target_model
        self.rows = shared_embedding
        self.draft = None
        if target_model is not None:
            self.draft = Draft(self.snapshot / DRAFT_FILE)
            if self.draft.tap_layers != [b - 1 for b in self.tap_slots]:
                raise ValueError("Drafter checkpoint reads other target layers")
            if self.rows is None:
                self.rows = TokenRows(target_model, "model.embed_tokens.weight")

    # What qwen3_4b's TargetDecoder reads in its drafter role.
    def embed_norm_lmhead(self):
        """The scaled embedding table in BF16, the draft's final norm."""
        V, chunk = gw.VOCAB, 16384
        table = np.empty((V, self.rows.q.shape[1]), bfloat16)
        for r0 in range(0, V, chunk):
            ids = np.arange(r0, min(r0 + chunk, V))
            table[ids] = (self.rows(ids) * gw.EMBED_SCALE).astype(bfloat16)
        return table, self.draft.norm, None

    def layer_qk_norm(self, layer):
        lw = self.draft.layers[layer]
        return lw["attn_q_norm"], lw["attn_k_norm"]


class GemmaDrafter:
    """qwen3_4b's TargetDecoder in its drafter role, plus the choice of the
    block's proposals. The device head's top RESCORE candidates per row are
    scored again in float32 against the int8 embedding rows the head is tied
    to: the BFP16 head alone picks a different token often enough to cost a
    few percent of the accepted length."""

    RESCORE = 8

    def __init__(self, artifact, weights, cache_dir):
        from model_paths import default_draft_cache

        cache = packed_cache(cache_dir or default_draft_cache(), weights)
        self.decoder = _qwen_decoder()(
            artifact, packed_cache=cache, model_weights=weights, stream_weights=True
        )
        self.weights = weights

    def __getattr__(self, name):
        if name == "decoder":
            raise AttributeError(name)
        return getattr(self.decoder, name)

    def propose_tokens(self, verification):
        """The block's proposals after its anchor, from a verify call's
        logits and the hidden state that produced them."""
        import torch

        lg = verification.logits[1:]
        top = torch.topk(torch.from_numpy(lg), self.RESCORE, dim=1).indices.numpy()
        d = self.weights.draft
        h = self.decoder.hidden([d.n_layers])[0].view(bfloat16)[1:]
        h = rms(h.astype(np.float32), d.norm, d.eps)
        e = self.weights.rows(top.reshape(-1)).reshape(*top.shape, -1)
        best = np.argmax(np.einsum("rkd,rd->rk", e, h), axis=1)
        return [int(t) for t in top[np.arange(len(top)), best]]

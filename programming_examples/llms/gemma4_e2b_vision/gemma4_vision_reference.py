# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# NumPy reference for the Gemma4 vision tower, plus the HF oracle it is
# checked against.
#
# The NumPy tower computes what the NPU encoder computes and can return each
# layer's output, which `make diagnosis` compares against. Run this file to
# check the NumPy tower against HF.
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gemma4_vision_weights import (  # noqa: E402
    ATTN_SCALE,
    DH,
    NUM_LAYERS,
    N_HEADS,
    VisionWeights,
    apply_rope_2d,
    gelu_tanh,
    pool,
    rmsnorm,
    rope_tables,
)


def _clipped(x, w, bounds):
    """One Gemma4ClippableLinear: clamp, matmul, clamp."""
    (in_lo, in_hi), (out_lo, out_hi) = bounds
    y = np.clip(np.asarray(x, np.float32), in_lo, in_hi) @ np.asarray(w, np.float32)
    return np.clip(y, out_lo, out_hi)


def patch_embed(w, pixel_values, position_ids, taps=None):
    """Gemma4VisionPatchEmbedder: scale, project, add the two position tables."""
    pad = (position_ids == -1).all(axis=-1)
    x = 2.0 * (np.asarray(pixel_values, np.float32) - 0.5)
    hidden = x @ np.asarray(w.patch_proj_T(), np.float32)

    table = w.position_table()  # [2, POS_TABLE, D]
    idx = np.maximum(position_ids, 0)
    pos = table[0][idx[:, 0]] + table[1][idx[:, 1]]
    pos[pad] = 0.0

    if taps is not None:
        taps["patch_proj"] = hidden
        taps["pos_embed"] = pos
    return hidden + pos, pad


def layer(w, h, i, cos, sin, pad, taps=None):
    """One Gemma4VisionEncoderLayer."""
    L = h.shape[0]
    t = (
        (lambda k, v: taps.__setitem__(f"L{i}.{k}", v))
        if taps is not None
        else (lambda k, v: None)
    )

    # ---- attention ----
    residual = h
    x = rmsnorm(h, w.norm(i, "input_ln"))
    q, k, v = (
        _clipped(x, w.wT(i, p), w.clamp(i, p)).reshape(L, N_HEADS, DH)
        for p in ("q", "k", "v")
    )
    # q/k norms carry a scale, v's does not (Gemma4RMSNorm(with_scale=False)).
    q = apply_rope_2d(rmsnorm(q, w.norm(i, "q_norm")), cos, sin)
    k = apply_rope_2d(rmsnorm(k, w.norm(i, "k_norm")), cos, sin)
    v = rmsnorm(v, None)
    t("q", q)
    t("k", k)
    t("v", v)

    # Bidirectional: no causal triangle, only the padding keys are masked.
    scores = np.einsum("lhd,mhd->hlm", q, k) * ATTN_SCALE
    scores[:, :, pad] = -np.inf
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    attn = np.einsum("hlm,mhd->lhd", p, v).reshape(L, N_HEADS * DH)
    t("attn", attn)

    o = _clipped(attn, w.wT(i, "o"), w.clamp(i, "o"))
    h = residual + rmsnorm(o, w.norm(i, "post_attn_ln"))
    t("post_attn", h)

    # ---- mlp ----
    residual = h
    x = rmsnorm(h, w.norm(i, "pre_ffn_ln"))
    g = gelu_tanh(_clipped(x, w.wT(i, "gate"), w.clamp(i, "gate")))
    u = _clipped(x, w.wT(i, "up"), w.clamp(i, "up"))
    d = _clipped(g * u, w.wT(i, "down"), w.clamp(i, "down"))
    h = residual + rmsnorm(d, w.norm(i, "post_ffn_ln"))
    t("out", h)
    return h


def tower(w, pixel_values, position_ids, n_layers=NUM_LAYERS, taps=None):
    """Full vision tower: pixels -> [n_soft, out_dim] soft tokens."""
    h, pad = patch_embed(w, pixel_values, position_ids, taps)
    if taps is not None:
        taps["embeds"] = h

    # Rope takes the positions unclamped, -1 for padding included, as
    # Gemma4VisionEncoder does; only the patch embedder and the pooler clamp.
    cos, sin = rope_tables(position_ids)
    for i in range(n_layers):
        h = layer(w, h, i, cos, sin, pad, taps)

    h = np.asarray(h, np.float32)
    h[pad] = 0.0
    pooled = pool(h, position_ids)
    if taps is not None:
        taps["pooled"] = pooled

    # Gemma4MultimodalEmbedder: RMSNorm without scale, then the projection.
    out = rmsnorm(pooled, None) @ np.asarray(w.embed_proj_T(), np.float32)
    if taps is not None:
        taps["projected"] = out
    return out


# ---------------------------------------------------------------- HF oracle
def hf_tower(pixel_values, position_ids, source=None):
    """HF's own Gemma4VisionModel + Gemma4MultimodalEmbedder, in f32.

    Only the vision tensors are materialised, not the whole checkpoint.
    """
    import json

    import torch
    from safetensors import safe_open
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config,
        Gemma4VisionConfig,
    )
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4MultimodalEmbedder,
        Gemma4VisionModel,
    )

    w = VisionWeights(source) if source else VisionWeights()
    cfg_json = json.loads((w.path.parent / "config.json").read_text())
    vcfg = Gemma4VisionConfig(**w.config)
    vcfg._attn_implementation = "eager"
    full = Gemma4Config(**cfg_json)

    # Built on CPU, not on the meta device: the rotary inv_freq buffer is
    # non-persistent, so it is absent from the checkpoint and a meta-built
    # module cannot materialise it.
    model = Gemma4VisionModel(vcfg).to(torch.float32)
    embedder = Gemma4MultimodalEmbedder(vcfg, full.get_text_config()).to(torch.float32)

    f = safe_open(str(w.path), "pt")
    keys = set(f.keys())

    def _load(module, prefix):
        sd = {}
        for name in list(module.state_dict()):
            full_name = f"{prefix}.{name}"
            if full_name in keys:
                sd[name] = f.get_tensor(full_name).to(torch.float32)
        missing = [n for n in module.state_dict() if n not in sd]
        module.load_state_dict(sd, strict=False)
        return missing

    # strict=False would leave every weight at its random init if the prefix
    # were wrong, so check that only the rope buffer is missing.
    for module, prefix in (
        (model, "model.vision_tower"),
        (embedder, "model.embed_vision"),
    ):
        missing = [n for n in _load(module, prefix) if "inv_freq" not in n]
        assert (
            not missing
        ), f"{prefix}: {len(missing)} tensors missing, e.g. {missing[:3]}"
    model.eval()
    embedder.eval()

    pv = torch.tensor(pixel_values, dtype=torch.float32)[None]
    pid = torch.tensor(position_ids, dtype=torch.long)[None]
    with torch.no_grad():
        # The padding soft tokens are stripped with a boolean index, which also
        # drops the batch axis: last_hidden_state is [n_soft, D].
        pooled = model(pixel_values=pv, pixel_position_ids=pid).last_hidden_state
        return embedder(pooled).numpy()


_ORACLE_DIR = _HERE / "_oracle_cache"


def cached_hf_tower(pixel_values, position_ids, source=None):
    """hf_tower(), memoised on disk, keyed on the inputs and the checkpoint
    rather than the image path, so neither a changed preprocessing config nor
    another checkpoint can hit a stale entry."""
    import hashlib

    from gemma4_vision_weights import checkpoint_identity

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(pixel_values, np.float32))
    h.update(np.ascontiguousarray(position_ids, np.int64))
    h.update(checkpoint_identity(source).encode())
    path = _ORACLE_DIR / f"{h.hexdigest()[:16]}.npy"
    if path.exists():
        return np.load(path)
    out = hf_tower(pixel_values, position_ids, source)
    _ORACLE_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.npy")
    np.save(tmp, out)
    os.replace(tmp, path)
    return out


def cosine(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def make_test_image():
    """Red square upper-left, blue circle lower-right: an image whose
    description can be checked by keyword."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (448, 448), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rectangle([60, 60, 200, 200], fill=(220, 30, 30))
    d.ellipse([250, 250, 400, 400], fill=(30, 80, 220))
    return im


def load_image(path=None):
    from PIL import Image

    return Image.open(path).convert("RGB") if path else make_test_image()


def main():
    import argparse

    from gemma4_vision_weights import preprocess

    ap = argparse.ArgumentParser(description="NumPy vision tower vs the HF oracle")
    ap.add_argument("--image", default=None)
    ap.add_argument("--layers", type=int, default=NUM_LAYERS)
    ap.add_argument("--max-soft-tokens", type=int, default=None)
    ap.add_argument("--min-cosine", type=float, default=0.9999)
    args = ap.parse_args()

    kw = {"max_soft_tokens": args.max_soft_tokens} if args.max_soft_tokens else {}
    pv, pid, n_soft = preprocess(load_image(args.image), **kw)
    pad = int((pid == -1).all(axis=-1).sum())
    print(f"patches {pv.shape[0]} ({pad} padding), soft tokens {n_soft}")

    w = VisionWeights()
    mine = tower(w, pv, pid, n_layers=args.layers)
    print(f"numpy tower -> {mine.shape}")
    assert mine.shape[0] == n_soft, (mine.shape, n_soft)

    if args.layers != NUM_LAYERS:
        print("partial layer count: skipping the HF comparison")
        return 0

    ref = cached_hf_tower(pv, pid)
    assert ref.shape == mine.shape, (ref.shape, mine.shape)
    cos = cosine(mine, ref)
    rel = float(np.abs(mine - ref).max() / (np.abs(ref).max() + 1e-9))
    print(f"vs HF: cosine {cos:.6f}  max_rel {rel:.3e}")
    ok = cos >= args.min_cosine
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

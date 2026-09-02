# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4NX weight loader + CPU reference for the Gemma4-E2B (text) Q4NX example.
#
# Audited against FastFlowLM's own tree, not inferred from the HF config alone:
#   FLM_Xclbin/Gemma4/decoding/{models/gemma4-e2b.h,kernels/model_spec.h}
#   models_simple/simple/gemma4_simple.py   (FLM's golden reference math)
#
# Three things make this model unlike every other q4nx example here, and all
# three are why it gets its own builder (fused_decode_ple) rather than a
# _MODELS entry in fused_decode:
#
#   1. Per-layer embeddings (PLE). A second embedding table, vocab x (35*256),
#      feeds a gated 256-wide input into EVERY layer after the FFN residual.
#   2. Two attention geometries. Sliding layers use head_dim 256 with theta 1e4;
#      the 7 full-attention layers use head_dim 512 with theta 1e6 and only a
#      QUARTER of the dims rotated (partial_rotary_factor 0.25 -> 128 of 512).
#   3. KV sharing. num_kv_shared_layers=20, so layers >= 15 reuse the k/v of the
#      last non-shared layer OF THE SAME TYPE; only 15 caches exist for 35 layers.
#
# The bundle also carries TWO codecs. Projections are Codec B (I8, packed
# (n_chunks,5120), w = scale*q + min) exactly as in the Gemma3 sibling; the two
# embedding tables are int8 with group-32 F32 scales, which is a different codec
# entirely -- reading them with the Codec B path yields garbage, so they have
# their own reader below.
import json
import struct
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Codec B block geometry is shared with every other q4nx example.
_FUSED_DECODE = str(Path(__file__).resolve().parents[2] / "fused_decode")
if _FUSED_DECODE not in sys.path:
    sys.path.insert(0, _FUSED_DECODE)
from proj_qmm_pack import (  # noqa: E402
    ROW_BLOCK,
    COL_BLOCK,
    GROUP,
    N_GROUPS,
    PARALLEL,
    BLOCK_BF16,
)

# ---------------------------------------------------------------- dims
D = 1536  # hidden_size
N_Q_HEADS = 8
N_KV_HEADS = 1  # MQA
DH_SLIDING = 256  # head_dim
DH_GLOBAL = 512  # global_head_dim
INTER = 6144  # mlp intermediate (some layers are double-wide, see below)
NUM_LAYERS = 35
VOCAB = 262144
PLI_D = 256  # hidden_size_per_layer_input
RMS_EPS = 1e-6
SLIDING_WINDOW = 512
FINAL_LOGIT_SOFTCAP = 30.0
ATTN_SCALE = 1.0  # config scaling is 1.0; NOT 1/sqrt(head_dim)
NUM_KV_SHARED_LAYERS = 20
FIRST_KV_SHARED_LAYER = NUM_LAYERS - NUM_KV_SHARED_LAYERS  # 15

# layer_types from config.json: 7 repeats of ssssF.
LAYER_TYPES = [
    "full_attention" if (i % 5) == 4 else "sliding_attention" for i in range(NUM_LAYERS)
]

# RoPE per layer type (config rope_parameters).
ROPE_SLIDING_THETA = 10000.0
ROPE_GLOBAL_THETA = 1000000.0
ROPE_GLOBAL_PARTIAL = 0.25  # only the first 25% of head_dim is rotated

# Gemma scales word embeddings by sqrt(width) -- but FLM's bundle has ALREADY
# folded that in, so applying it here double-scales. Measured against FLM's own
# golden reference (decoding/references/gemma4_e2b_ref.safetensors):
# `input_embedding` has rms 1.171 and the RAW table lookup matches it to a ratio
# of 0.9998, while a sqrt(1536)=39.19x scaled lookup is 39x too large.
# The failure mode is quiet: RMSNorm is scale-invariant, so every normed input
# looks right and only the RESIDUAL stream is wrong -- 39x larger than the
# sublayer outputs added to it, which makes all 35 layers a near-passthrough of
# the embedding and yields fluent, contentless predictions.
EMBED_SCALE = 1.0
PLE_EMBED_SCALE = 1.0
# PLE combine constants. FLM's gemma4-e2b.h SWAPS these two NAMES relative to the
# reference implementation; the values are unambiguous, so key off the values.
PLE_MODEL_PROJ_SCALE = float(D**-0.5)  # 0.025515518  (FLM calls it *_INPUT_SCALE)
PLE_INPUT_SCALE = float(
    2.0**-0.5
)  # 0.707106781  (FLM calls it *_MODEL_PROJECTION_SCALE)

# Row de-interleave within a chunk: row_in_block = g*PARALLEL + 2*k + bit.
_G = ROW_BLOCK // PARALLEL  # 2
_EVEN = np.array(
    [g * PARALLEL + 2 * k for g in range(_G) for k in range(PARALLEL // 2)]
)
_ODD = _EVEN + 1


def is_sliding(layer_idx):
    return LAYER_TYPES[layer_idx] == "sliding_attention"


def head_dim(layer_idx):
    return DH_SLIDING if is_sliding(layer_idx) else DH_GLOBAL


def kv_source_layer(layer_idx):
    """Which layer's k/v this layer actually uses.

    Layers below FIRST_KV_SHARED_LAYER own their cache. At or above it, the layer
    reuses the LAST layer of the same type below the boundary -- so the 7
    full-attention layers and the 28 sliding ones share two different sources.
    """
    if layer_idx < FIRST_KV_SHARED_LAYER:
        return layer_idx
    prev = LAYER_TYPES[:FIRST_KV_SHARED_LAYER]
    want = LAYER_TYPES[layer_idx]
    return len(prev) - 1 - prev[::-1].index(want)


def owns_kv(layer_idx):
    return layer_idx < FIRST_KV_SHARED_LAYER


def _bf(a):
    return a.view(bfloat16).astype(np.float32)


def _rmsnorm(x, w=None, eps=RMS_EPS):
    """Gemma RMSNorm: y = x * rsqrt(mean(x^2)+eps) * (1 + w).

    Gemma4 does NOT use the Gemma2/3 (1+w) convention -- the weight is applied
    directly. Established three independent ways, because the sibling examples
    here all fold +1 and the instinct is to copy them:
      - the bundle ships the RAW HF weights (bundle_w == hf_w exactly, checked
        against the full-precision checkpoint);
      - FLM's own kernels/rms_residual.cc multiplies by w with no add;
      - FLM's reference SimpleRMSNorm.forward is `output * self.weight`.
    Measured: adding the +1 takes layer_0 from cos 0.986 to 0.962 and collapses
    everything downstream (rms_ratio 1.2-1.6), so this is settled empirically too.

    w=None is the weightless value_norm (with_scale=False in the reference).
    """
    y = x / np.sqrt(np.mean(x.astype(np.float32) ** 2, -1, keepdims=True) + eps)
    return y if w is None else y * w


def _gelu_tanh(x):
    return (
        0.5
        * x
        * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))))
    )


class Q4nxModel:
    """Reader for FLM's Gemma4-E2B model.q4nx (both codecs)."""

    def __init__(self, path):
        p = Path(path)
        self.path = p / "model.q4nx" if p.is_dir() else p
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self._hdr = json.loads(f.read(n))
            self._data_off = 8 + n
        self._meta = self._hdr.pop("__metadata__", None)
        if self._meta is not None:
            raise SystemExit(
                f"{self.path}: this reader targets Codec B (I8, packed shape, no "
                "__metadata__). The bundle carries Codec A; re-export or use the "
                "Codec A path in llama32_1b_q4nx."
            )

    def fingerprint(self):
        """Hash of the HEADER (not the payload) so any derived cache invalidates
        when FLM re-exports the bundle."""
        import hashlib

        return hashlib.sha256(
            json.dumps(self._hdr, sort_keys=True).encode()
        ).hexdigest()[:16]

    def has(self, name):
        return name in self._hdr

    def _raw(self, name, dtype):
        e = self._hdr[name]
        s, t = e["data_offsets"]
        return np.fromfile(
            self.path,
            dtype=dtype,
            count=(t - s) // np.dtype(dtype).itemsize,
            offset=self._data_off + s,
        )

    def bf16(self, name):
        return _bf(self._raw(name, np.int16))

    def dequant(self, name, M, K):
        """Codec B tensor -> float32 [M, K], w = scale*q + min.

        The header shape is the PACKED (n_chunks, 5120); the logical (M,K) is not
        recoverable from it, so the caller supplies it and we assert consistency
        rather than reshape blindly.
        """
        nbi, nbj = M // ROW_BLOCK, K // COL_BLOCK
        nb = nbi * nbj
        got = self._hdr[name]["shape"][0]
        if got != nb:
            raise SystemExit(
                f"{name}: header n_chunks={got} but (M/32)*(K/256)=({M}/32)*({K}/256)"
                f"={nb}. The (M,K) handed to dequant() is wrong for this tensor."
            )
        i16 = self._raw(name, np.int16).reshape(nb, BLOCK_BF16)
        sc = _bf(i16[:, 0:256].copy()).reshape(nb, N_GROUPS, ROW_BLOCK)
        mn = _bf(i16[:, 256:512].copy()).reshape(nb, N_GROUPS, ROW_BLOCK)
        qb = (
            i16[:, 512:BLOCK_BF16]
            .copy()
            .view(np.uint8)
            .reshape(nb, _G, COL_BLOCK, PARALLEL // 2)
        )
        lo = (qb & 0xF).transpose(0, 1, 3, 2).reshape(nb, ROW_BLOCK // 2, COL_BLOCK)
        hi = (qb >> 4).transpose(0, 1, 3, 2).reshape(nb, ROW_BLOCK // 2, COL_BLOCK)
        q = np.zeros((nb, ROW_BLOCK, COL_BLOCK), np.float32)
        q[:, _EVEN, :] = lo
        q[:, _ODD, :] = hi
        w = np.repeat(sc.transpose(0, 2, 1), GROUP, axis=2) * q + np.repeat(
            mn.transpose(0, 2, 1), GROUP, axis=2
        )
        return (
            w.reshape(nbi, nbj, ROW_BLOCK, COL_BLOCK)
            .transpose(0, 2, 1, 3)
            .reshape(M, K)
        )

    def embed_rows(self, name, ids):
        """Rows of an int8 group-32 embedding table, dequantized.

        This is NOT Codec B. `model.embed_tokens.weight` is I8 (V, W) with a
        companion `.scale` F32 (V, W/32): one scale per 32 contiguous columns,
        w = scale * q, no additive min. Reading it with dequant() gives garbage.
        Gathers only the requested rows -- the per-layer table is 2.3 GB.
        """
        e, es = self._hdr[name], self._hdr[name + ".scale"]
        V, W = e["shape"]
        Vs, G = es["shape"]
        if Vs != V or G * 32 != W:
            raise SystemExit(
                f"{name}: quant {e['shape']} vs scale {es['shape']} are not a "
                "group-32 pair"
            )
        qs, _ = e["data_offsets"]
        ss, _ = es["data_offsets"]
        ids = np.asarray(ids, np.int64).reshape(-1)
        out = np.empty((ids.size, W), np.float32)
        with open(self.path, "rb") as f:
            for i, tok in enumerate(ids):
                f.seek(self._data_off + qs + int(tok) * W)
                q = np.frombuffer(f.read(W), np.int8).astype(np.float32)
                f.seek(self._data_off + ss + int(tok) * G * 4)
                sc = np.frombuffer(f.read(G * 4), np.float32)
                out[i] = q * np.repeat(sc, 32)
        return out

    # ---- per-layer accessors -------------------------------------------------
    def _proj_out(self, name, in_dim):
        """Logical out dim implied by a Codec B tensor's chunk count."""
        return self._hdr[name]["shape"][0] // (in_dim // COL_BLOCK) * ROW_BLOCK

    def mlp_inter(self, k):
        """This layer's intermediate size; E2B marks some layers DOUBLE_WIDE_MLP,
        so it is read from the bundle rather than assumed to be INTER."""
        return self._proj_out(f"model.layers.{k}.mlp.gate_proj.weight", D)

    def layer_weights(self, k):
        dh = head_dim(k)
        dq, dkv = N_Q_HEADS * dh, N_KV_HEADS * dh
        inter = self.mlp_inter(k)
        p = f"model.layers.{k}."
        w = dict(
            q=self.dequant(p + "self_attn.q_proj.weight", dq, D),
            o=self.dequant(p + "self_attn.o_proj.weight", D, dq),
            up=self.dequant(p + "mlp.up_proj.weight", inter, D),
            gate=self.dequant(p + "mlp.gate_proj.weight", inter, D),
            down=self.dequant(p + "mlp.down_proj.weight", D, inter),
        )
        # Shared-KV layers still ship k/v in the bundle, but the reference never
        # evaluates them -- it takes k/v from the source layer's cache. Load them
        # only where they are actually used, so a wrong sharing map shows up as a
        # KeyError instead of silently-unused weights.
        if owns_kv(k):
            w["k"] = self.dequant(p + "self_attn.k_proj.weight", dkv, D)
            w["v"] = self.dequant(p + "self_attn.v_proj.weight", dkv, D)
        return w

    def layer_norms(self, k):
        p = f"model.layers.{k}."
        return dict(
            input=self.bf16(p + "input_layernorm.weight"),
            post_attn=self.bf16(p + "post_attention_layernorm.weight"),
            pre_ffn=self.bf16(p + "pre_feedforward_layernorm.weight"),
            post_ffn=self.bf16(p + "post_feedforward_layernorm.weight"),
            post_ple=self.bf16(p + "post_layernorm.weight"),
            q_norm=self.bf16(p + "self_attn.q_norm.weight"),
            k_norm=self.bf16(p + "self_attn.k_norm.weight"),
            out_scale=float(self.bf16(p + "layer_output_scale.weight")[0]),
        )

    def layer_ple(self, k):
        """The two PLE matrices for one layer, un-tiled to plain [out, in].

        Both ship pre-tiled bf16: inp_gate (8,1536,32) is [256,1536] as 8 row
        tiles of 32, and per_layer_projection (48,256,32) is [1536,256] the same
        way. per_layer_model_proj is stored per layer as (8,1536,32) = the 256
        rows of the global (35*256, 1536) projection that belong to this layer.
        """
        p = f"model.layers.{k}."
        gate = self.bf16(p + "inp_gate.weight").reshape(8, D, 32)
        proj = self.bf16(p + "per_layer_projection.weight").reshape(48, PLI_D, 32)
        mp = self.bf16(f"model.per_layer_model_proj.weight_layer{k}").reshape(8, D, 32)
        return dict(
            inp_gate=gate.transpose(0, 2, 1).reshape(PLI_D, D),
            per_layer_projection=proj.transpose(0, 2, 1).reshape(D, PLI_D),
            model_proj=mp.transpose(0, 2, 1).reshape(PLI_D, D),
        )

    def lm_head_rows(self, r0, r1, name="lm_head.weight", K=D):
        """Dequantize only output rows [r0, r1) of a Codec B matrix.

        lm_head here is its OWN 4-bit matrix (262144 x 1536, not tied to
        embed_tokens), which is 1.6 GB dequantized in float32. The chunk layout is
        block-major -- index = i*nbj + j over (out/32, in/256) -- so a row range
        that is a multiple of ROW_BLOCK maps to a contiguous chunk range and can
        be read without touching the rest.
        """
        if r0 % ROW_BLOCK or (r1 - r0) % ROW_BLOCK:
            raise SystemExit(f"row range [{r0},{r1}) must be a multiple of {ROW_BLOCK}")
        nbj = K // COL_BLOCK
        i0, i1 = r0 // ROW_BLOCK, r1 // ROW_BLOCK
        nb = (i1 - i0) * nbj
        e = self._hdr[name]
        s, _ = e["data_offsets"]
        i16 = np.fromfile(
            self.path,
            dtype=np.int16,
            count=nb * BLOCK_BF16,
            offset=self._data_off + s + i0 * nbj * BLOCK_BF16 * 2,
        ).reshape(nb, BLOCK_BF16)
        sc = _bf(i16[:, 0:256].copy()).reshape(nb, N_GROUPS, ROW_BLOCK)
        mn = _bf(i16[:, 256:512].copy()).reshape(nb, N_GROUPS, ROW_BLOCK)
        qb = (
            i16[:, 512:BLOCK_BF16]
            .copy()
            .view(np.uint8)
            .reshape(nb, _G, COL_BLOCK, PARALLEL // 2)
        )
        lo = (qb & 0xF).transpose(0, 1, 3, 2).reshape(nb, ROW_BLOCK // 2, COL_BLOCK)
        hi = (qb >> 4).transpose(0, 1, 3, 2).reshape(nb, ROW_BLOCK // 2, COL_BLOCK)
        q = np.zeros((nb, ROW_BLOCK, COL_BLOCK), np.float32)
        q[:, _EVEN, :] = lo
        q[:, _ODD, :] = hi
        w = np.repeat(sc.transpose(0, 2, 1), GROUP, axis=2) * q + np.repeat(
            mn.transpose(0, 2, 1), GROUP, axis=2
        )
        return (
            w.reshape(i1 - i0, nbj, ROW_BLOCK, COL_BLOCK)
            .transpose(0, 2, 1, 3)
            .reshape(r1 - r0, K)
        )

    def rope_freqs(self):
        """The 'proportional' rope divisor table (cached; see rope_lut)."""
        if not hasattr(self, "_rf"):
            self._rf = (
                self.bf16("rope_freqs.weight")
                if self.has("rope_freqs.weight")
                else None
            )
        return self._rf

    def globals(self):
        return dict(
            final_norm=self.bf16("model.norm.weight"),
            ple_proj_norm=self.bf16("model.per_layer_proj_norm.weight"),
        )


def rope_lut(pos, layer_idx, rope_freqs=None):
    """cos/sin for one position, in FLM's convention.

    FLM's kernels/rope.cc rotates over DH_2 = DH/2 pairs and ALWAYS pairs dim i
    with dim i + DH/2 -- there is no partial-rotary special case in the kernel.
    Partial rotary is folded into the frequency table instead, exactly as their
    phi4_rope[128] table does it (a zeroed tail => cos=1/sin=0 => identity).

    For Gemma4 the divisor ships in the bundle as `rope_freqs.weight` (256 = 512/2
    entries): [1.0] * 64 then 1.0003e30 * 192. Dividing by 1e30 zeroes those
    frequencies, and the 64 live entries are exactly rot/2 for rot = 0.25 * 512,
    which is the config's partial_rotary_factor. That makes the live pairing
    (i, i+256) -- NOT (i, i+64) as a naive "rotate the first 128 dims" reading
    gives. The bundle's q/k are stored to match this pairing.

    Weakly exercised by FLM's 16-token reference: at theta=1e6 the angles for
    positions 0..15 are tiny, so a wrong convention costs only ~0.003 cosine
    there. It matters at real context lengths.
    """
    dh = head_dim(layer_idx)
    half = dh // 2
    theta = ROPE_SLIDING_THETA if is_sliding(layer_idx) else ROPE_GLOBAL_THETA
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / dh))
    if not is_sliding(layer_idx) and rope_freqs is not None:
        inv = inv / rope_freqs[:half]  # "proportional" rope: a DIVISOR table
    ang = float(pos) * inv
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32), dh


def apply_rope(x, cos, sin, rot):
    """Half-split rotary on the first `rot` dims of the last axis."""
    out = x.copy()
    h = rot // 2
    a, b = out[..., :h], out[..., h:rot]
    out[..., :h] = a * cos - b * sin
    out[..., h:rot] = b * cos + a * sin
    return out


def per_layer_inputs(model, ids, embeds):
    """The PLE input for EVERY layer, [T, NUM_LAYERS, PLI_D].

    Computed ONCE from the model's input embeddings -- NOT per layer from that
    layer's hidden state. `project_per_layer_inputs` in FLM's reference takes
    `inputs_embeds` and runs before the decoder loop, so all 35 slices come from
    the same tensor.

    Getting this wrong is nearly invisible: the projection still has the right
    shape and scale, the gate path is untouched, and early layers barely move
    (their input is still close to the embedding). It drifts with depth and hurts
    most where the layer's post_per_layer_input_norm is large -- which is exactly
    why layers 8/13/14 were the worst. Feeding the layer's own x scored
    per_layer_projection at cos 0.30/0.78/0.67 on layers 8/7/9; feeding
    inputs_embeds scores 1.000000.

    The bundle's per-layer table is already scaled by sqrt(PLI_D) (measured: the
    bundle/HF ratio is 16.0014), hence PLE_EMBED_SCALE = 1.
    """
    T = len(ids)
    tbl = model.embed_rows("model.per_layer_token_embd.weight", ids)
    tbl = tbl.reshape(T, NUM_LAYERS, PLI_D) * PLE_EMBED_SCALE
    norm_w = model.globals()["ple_proj_norm"]
    out = np.empty((T, NUM_LAYERS, PLI_D), np.float32)
    for L in range(NUM_LAYERS):
        mp = model.layer_ple(L)["model_proj"]
        proj = _rmsnorm((embeds @ mp.T) * PLE_MODEL_PROJ_SCALE, norm_w)
        out[:, L, :] = (proj + tbl[:, L, :]) * PLE_INPUT_SCALE
    return out


def _sliding_mask(t_q, t_k, window):
    """Causal mask, optionally restricted to a left window of `window` keys."""
    qi = np.arange(t_q)[:, None]
    ki = np.arange(t_k)[None, :]
    ok = ki <= qi
    if window:
        ok &= ki > qi - window
    return np.where(ok, 0.0, -np.inf).astype(np.float32)


def forward_prompt(model, ids, want_logits=True, lm_chunk=16384):
    """Full CPU reference forward for a prompt. Returns (logits, kv) .

    Streams the layers: the dequantized weights of all 35 layers would be ~5 GB
    in float32, so each layer is dequantized, used, and dropped. That makes this
    slow but exact, which is what a correctness gate needs.

    kv maps a SOURCE layer index -> (k_embed, v). Shared layers never write it;
    they read the entry their kv_source_layer() points at, which is how the
    20 kv-shared layers of this model avoid carrying a cache at all.
    """
    ids = [int(t) for t in np.asarray(ids).reshape(-1)]
    T = len(ids)
    g = model.globals()

    x = model.embed_rows("model.embed_tokens.weight", ids) * EMBED_SCALE

    # --- PLE inputs, once for the whole prompt, from the EMBEDDINGS --------
    pli_all = per_layer_inputs(model, ids, x)

    kv = {}
    for L in range(NUM_LAYERS):
        w, nm, pw = model.layer_weights(L), model.layer_norms(L), model.layer_ple(L)
        dh, sliding = head_dim(L), is_sliding(L)

        pli = pli_all[:, L, :]

        # ---- attention ----
        r = x
        x1 = _rmsnorm(x, nm["input"])
        q = (x1 @ w["q"].T).reshape(T, N_Q_HEADS, dh)
        q = _rmsnorm(q, nm["q_norm"])
        cos, sin, rot = None, None, None
        qe = np.empty_like(q)
        for t in range(T):
            cos, sin, rot = rope_lut(t, L, rope_freqs=model.rope_freqs())
            qe[t] = apply_rope(q[t], cos, sin, rot)

        src = kv_source_layer(L)
        if owns_kv(L):
            k = (x1 @ w["k"].T).reshape(T, N_KV_HEADS, dh)
            v = (x1 @ w["v"].T).reshape(T, N_KV_HEADS, dh)
            k = _rmsnorm(k, nm["k_norm"])
            v = _rmsnorm(v, None)  # value_norm is weightless
            ke = np.empty_like(k)
            for t in range(T):
                cos, sin, rot = rope_lut(t, L, rope_freqs=model.rope_freqs())
                ke[t] = apply_rope(k[t], cos, sin, rot)
            kv[L] = (ke, v)
        ke, v = kv[src]

        # MQA: a single kv head broadcast over all 8 q heads. Squeeze that axis
        # OFF ke/v up front -- carrying it into the einsum and slicing it out
        # afterwards is how the query axis got collapsed instead, which left the
        # scores constant along keys and every softmax exactly uniform.
        kh, vh = ke[:, 0], v[:, 0]  # (S, dh)
        mask = _sliding_mask(T, kh.shape[0], SLIDING_WINDOW if sliding else 0)
        s = np.einsum("thd,sd->hts", qe, kh) * ATTN_SCALE
        s = s + mask[None]
        s = np.exp(s - s.max(-1, keepdims=True))
        s = s / s.sum(-1, keepdims=True)
        o = np.einsum("hts,sd->thd", s, vh).reshape(T, N_Q_HEADS * dh)
        o1 = r + _rmsnorm(o @ w["o"].T, nm["post_attn"])

        # ---- FFN (GELU-tanh gated) ----
        h = _rmsnorm(o1, nm["pre_ffn"])
        h = _gelu_tanh(h @ w["gate"].T) * (h @ w["up"].T)
        o2 = o1 + _rmsnorm(h @ w["down"].T, nm["post_ffn"])

        # ---- per-layer embedding injection ----
        gate = _gelu_tanh(o2 @ pw["inp_gate"].T) * pli
        o3 = o2 + _rmsnorm(gate @ pw["per_layer_projection"].T, nm["post_ple"])
        x = o3 * nm["out_scale"]

    x = _rmsnorm(x, g["final_norm"])
    if not want_logits:
        return None, kv

    # lm_head is its own 4-bit matrix (NOT tied to embed_tokens) and dequantizing
    # all 262144x1536 at once is 1.6 GB, so walk it in row chunks.
    last = x[-1]
    logits = np.empty(VOCAB, np.float32)
    for r0 in range(0, VOCAB, lm_chunk):
        r1 = min(r0 + lm_chunk, VOCAB)
        blk = model.lm_head_rows(r0, r1)
        logits[r0:r1] = blk @ last
    if FINAL_LOGIT_SOFTCAP:
        logits = FINAL_LOGIT_SOFTCAP * np.tanh(logits / FINAL_LOGIT_SOFTCAP)
    return logits, kv

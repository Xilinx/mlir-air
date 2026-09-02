# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Layer-by-layer check of the CPU reference against FastFlowLM's OWN golden
# activations (FLM_Xclbin/Gemma4/decoding/references/gemma4_e2b_ref.safetensors:
# input_ids, input_embedding, and layer_0..layer_34 for a 16-token prompt).
#
# This is the gate for the whole port. The end-to-end token check is too blunt:
# a wrong residual scale still produces fluent English, and a wrong per-layer
# embedding path still produces plausible logits. Per-layer cosine says exactly
# which layer first diverges, which is the only cheap way to tell the three new
# mechanisms (PLE, sliding/global split, kv sharing) apart when one is broken.
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma4_e2b_q4nx_weights as W  # noqa: E402

FLM = Path("/home/strixminipc/rocm_fastflowlm/FastFlowLM")
REF = FLM / "FLM_Xclbin/Gemma4/decoding/references/gemma4_e2b_ref.safetensors"
BUNDLE = FLM / "models/Gemma4-E2B-IT-NPU2"


def load_ref(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, e in hdr.items():
            if k == "__metadata__":
                continue
            s, t = e["data_offsets"]
            f.seek(base + s)
            raw = f.read(t - s)
            if e["dtype"] == "I32":
                out[k] = np.frombuffer(raw, np.int32).reshape(e["shape"])
            else:
                out[k] = (
                    np.frombuffer(raw, bfloat16).reshape(e["shape"]).astype(np.float32)
                )
    return out


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default=str(BUNDLE))
    ap.add_argument("--ref", default=str(REF))
    ap.add_argument("--layers", type=int, default=W.NUM_LAYERS)
    ap.add_argument(
        "--tol",
        type=float,
        default=0.99,
        help="minimum per-layer cosine to call the layer a match",
    )
    a = ap.parse_args()

    ref = load_ref(a.ref)
    ids = ref["input_ids"].tolist()
    T = len(ids)
    m = W.Q4nxModel(a.bundle)
    g = m.globals()

    x = m.embed_rows("model.embed_tokens.weight", ids) * W.EMBED_SCALE
    c = cos(x, ref["input_embedding"])
    print(
        f"input_embedding      cos={c:.6f}  rms mine={np.sqrt((x**2).mean()):8.4f} "
        f"ref={np.sqrt((ref['input_embedding']**2).mean()):8.4f}"
    )
    # Track the reference stream too: feeding OUR x forward compounds an early
    # error into every later layer and hides where it started. Running each layer
    # on the REFERENCE input isolates the layer under test.
    # From the EMBEDDINGS, once -- not from each layer's hidden state.
    ple = W.per_layer_inputs(m, ids, x)

    # SEPARATE caches per stream. Sharing one dict lets the free-running pass
    # overwrite the teacher's k/v, which silently contaminates every kv-SHARED
    # layer (>=15) -- they read layer 13/14's entry, so they would have been
    # scored against free-running keys while claiming to be teacher-forced.
    kv_t, kv_f = {}, {}
    x_free = x.copy()
    first_bad = None
    teacher_cos, free_cos = [], []
    for L in range(a.layers):
        x_ref_in = ref["input_embedding"] if L == 0 else ref[f"layer_{L-1}"]
        y_teacher = layer(m, g, ple, kv_t, L, x_ref_in, T)
        y_free = layer(m, g, ple, kv_f, L, x_free, T)
        x_free = y_free
        r = ref[f"layer_{L}"]
        ct, cf = cos(y_teacher, r), cos(y_free, r)
        # Cosine CANNOT see a wrong scalar multiplier, and this model applies a
        # per-layer layer_output_scale to the whole residual stream -- so a bad
        # scale reads as a perfect cosine and still destroys the model a few
        # layers later. Report the magnitude ratio alongside.
        rms = lambda a: float(np.sqrt((a.astype(np.float64) ** 2).mean()))
        ratio = rms(y_teacher) / (rms(r) + 1e-30)
        teacher_cos.append(ct)
        free_cos.append(cf)
        flag = (
            ""
            if ct >= a.tol
            else "   <== FIRST DIVERGENCE" if first_bad is None else "   <== bad"
        )
        if ct < a.tol and first_bad is None:
            first_bad = L
        print(
            f"layer_{L:<2} {'s' if W.is_sliding(L) else 'F'} dh={W.head_dim(L):<4} "
            f"kv={'own' if W.owns_kv(L) else f'<-{W.kv_source_layer(L)}':>5} "
            f"cos(t)={ct:.5f} cos(free)={cf:.5f} rms_ratio={ratio:8.4f}{flag}"
        )
    print()
    worst_L = min(range(len(teacher_cos)), key=lambda i: teacher_cos[i])
    print(
        f"min teacher-forced cos = {teacher_cos[worst_L]:.5f} at layer {worst_L}; "
        f"free-running at layer_{len(teacher_cos)-1} = {free_cos[-1]:.5f}"
    )
    if first_bad is None:
        print(f"PASS: all {a.layers} layers >= {a.tol}")
        return 0
    print(
        f"FAIL: layer {first_bad} is the first below {a.tol} "
        f"({'sliding' if W.is_sliding(first_bad) else 'full'}, "
        f"dh={W.head_dim(first_bad)}, "
        f"{'owns kv' if W.owns_kv(first_bad) else 'shares kv'})."
    )
    print(
        "      rms_ratio ~1.0 with a low cosine means a CONTENT bug (a block fed "
        "the wrong tensor);\n      a bad rms_ratio means a scale bug, which cosine "
        "alone cannot see."
    )
    return 1


def layer(m, g, ple, kv, L, x, T):
    """One decoder layer, exactly as gemma4_e2b_q4nx_weights.forward_prompt."""
    w, nm, pw = m.layer_weights(L), m.layer_norms(L), m.layer_ple(L)
    dh, sliding = W.head_dim(L), W.is_sliding(L)
    pli = ple[:, L, :]

    r = x
    x1 = W._rmsnorm(x, nm["input"])
    q = W._rmsnorm((x1 @ w["q"].T).reshape(T, W.N_Q_HEADS, dh), nm["q_norm"])
    qe = np.stack(
        [
            W.apply_rope(q[t], *W.rope_lut(t, L, rope_freqs=m.rope_freqs()))
            for t in range(T)
        ]
    )
    store = kv
    if W.owns_kv(L):
        k = W._rmsnorm((x1 @ w["k"].T).reshape(T, W.N_KV_HEADS, dh), nm["k_norm"])
        v = W._rmsnorm((x1 @ w["v"].T).reshape(T, W.N_KV_HEADS, dh), None)
        ke = np.stack(
            [
                W.apply_rope(k[t], *W.rope_lut(t, L, rope_freqs=m.rope_freqs()))
                for t in range(T)
            ]
        )
        store[L] = (ke, v)
    ke, v = store[W.kv_source_layer(L)]
    kh, vh = ke[:, 0], v[:, 0]
    s = np.einsum("thd,sd->hts", qe, kh) * W.ATTN_SCALE
    s = s + W._sliding_mask(T, kh.shape[0], W.SLIDING_WINDOW if sliding else 0)[None]
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    o = np.einsum("hts,sd->thd", s, vh).reshape(T, W.N_Q_HEADS * dh)
    o1 = r + W._rmsnorm(o @ w["o"].T, nm["post_attn"])

    h = W._rmsnorm(o1, nm["pre_ffn"])
    h = W._gelu_tanh(h @ w["gate"].T) * (h @ w["up"].T)
    o2 = o1 + W._rmsnorm(h @ w["down"].T, nm["post_ffn"])

    gate = W._gelu_tanh(o2 @ pw["inp_gate"].T) * pli
    o3 = o2 + W._rmsnorm(gate @ pw["per_layer_projection"].T, nm["post_ple"])
    return o3 * nm["out_scale"]


if __name__ == "__main__":
    raise SystemExit(main())

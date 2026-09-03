# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Run ONE Gemma4 decoder layer on NPU2 and score it against the numpy reference.
#
# This is the bring-up gate for the fused_decode_ple design, and it is
# deliberately not a chatbot. It feeds a chosen hidden state through a
# single-layer build and compares the layer output element-for-element with
# gemma4_e2b_q4nx_weights, which is itself already gated per layer against
# FastFlowLM's golden activations. One dispatch therefore exercises the whole
# device path at once -- q4 projections, the 4-norm sandwich, GQA attention, the
# GLU, and the per-layer-embedding branch this builder exists for.
#
# --layer defaults to 4, the first FULL-attention layer, on purpose. Layer 0 is
# sliding, and a sliding layer only matches once the padded-head interleave in
# gemma4_e2b_q4nx_requant.py is correct; starting on a full layer separates
# "the design is wrong" from "the padding is wrong". Run --layer 0 second.
#
# Position 0 with a fresh KV cache, so the dispatch appends this token's own K/V
# and attends to itself. That keeps the test independent of KV seeding, which is
# a separate mechanism with its own failure modes.
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_DEC = _HERE / ".." / ".." / "fused_decode_ple"
BUNDLE = "/home/strixminipc/rocm_fastflowlm/FastFlowLM/models/Gemma4-E2B-IT-NPU2"


def _load_fd(uni_dec, attn_maxl):
    """Import the builder with the env its module-level constants read."""
    os.environ.update(
        DECODE_MODEL="gemma4-e2b",
        VOCAB_CHUNK_I2="27",
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        DECODE_GOLDEN_L=str(attn_maxl),
        DECODE_UNI_DEC=str(uni_dec),
    )
    sys.path.insert(0, str(_DEC))
    spec = importlib.util.spec_from_file_location(
        "fdp", str(_DEC / "fused_decode_ple.py")
    )
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    return fd


def ple_embed(gw, qm, L, tok):
    """This layer's slice of the per-layer token-embedding table, for one token.

    The bundle's table is already scaled by sqrt(PLI_D) (PLE_EMBED_SCALE == 1).
    """
    tbl = qm.embed_rows("model.per_layer_token_embd.weight", [tok])
    return (tbl.reshape(1, gw.NUM_LAYERS, gw.PLI_D)[0, L] * gw.PLE_EMBED_SCALE).astype(
        np.float32
    )


def cpu_layer(gw, qm, L, x, x0, emb_L, ple=True):
    """The reference layer, at T=1 and position 0.

    Mirrors check_vs_flm_reference.layer(); kept here rather than imported so
    the device is scored against the SAME expression the per-layer gate uses,
    with no shared mutable state between the two.
    """
    w, nm, pw = qm.layer_weights(L), qm.layer_norms(L), qm.layer_ple(L)
    dh = gw.head_dim(L)
    T = 1

    # The per-layer input, from the token EMBEDDING (never the hidden state).
    pli = (
        gw._rmsnorm(
            (x0 @ pw["model_proj"].T) * gw.PLE_MODEL_PROJ_SCALE,
            qm.globals()["ple_proj_norm"],
        )
        + emb_L
    ) * gw.PLE_INPUT_SCALE

    r = x
    x1 = gw._rmsnorm(x, nm["input"])
    q = gw._rmsnorm((x1 @ w["q"].T).reshape(T, gw.N_Q_HEADS, dh), nm["q_norm"])
    cos, sin, _ = gw.rope_lut(0, L, rope_freqs=qm.rope_freqs())
    qe = np.stack([gw.apply_rope(q[t], cos, sin, dh) for t in range(T)])
    k = gw._rmsnorm((x1 @ w["k"].T).reshape(T, gw.N_KV_HEADS, dh), nm["k_norm"])
    v = gw._rmsnorm((x1 @ w["v"].T).reshape(T, gw.N_KV_HEADS, dh), None)
    ke = np.stack([gw.apply_rope(k[t], cos, sin, dh) for t in range(T)])

    kh, vh = ke[:, 0], v[:, 0]
    s = np.einsum("thd,sd->hts", qe, kh) * gw.ATTN_SCALE
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    o = np.einsum("hts,sd->thd", s, vh).reshape(T, gw.N_Q_HEADS * dh)
    o1 = r + gw._rmsnorm(o @ w["o"].T, nm["post_attn"])

    h = gw._rmsnorm(o1, nm["pre_ffn"])
    h = gw._gelu_tanh(h @ w["gate"].T) * (h @ w["up"].T)
    o2 = o1 + gw._rmsnorm(h @ w["down"].T, nm["post_ffn"])

    if not ple:
        # DECODE_NO_PLE: the design stops at the post-FFN residual.
        return o2.reshape(-1)
    gate = gw._gelu_tanh(o2 @ pw["inp_gate"].T) * pli
    o3 = o2 + gw._rmsnorm(gate @ pw["per_layer_projection"].T, nm["post_ple"])
    return (o3 * nm["out_scale"]).reshape(-1)


def cos_sim(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--cache", default="/tmp/g4_layer.npz")
    ap.add_argument("--xclbin", default=str(_DEC / "decode.xclbin"))
    ap.add_argument("--insts", default=str(_DEC / "decode.insts.bin"))
    ap.add_argument("--attn-maxl", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.99)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument(
        "--dump",
        help="save the device readback and the reference to an .npz, so a "
        "mismatch can be hypothesis-tested offline without re-dispatching.",
    )
    ap.add_argument(
        "--no-ple",
        action="store_true",
        help="score a DECODE_NO_PLE=1 build (no PLE branch). Numerically wrong "
        "as a model, but it separates a base-design failure from a PLE one.",
    )
    a = ap.parse_args()

    sys.path.insert(0, str(_HERE))
    import gemma4_e2b_q4nx_weights as gw
    import gemma4_e2b_q4nx_requant as rq

    if a.no_ple:
        os.environ["DECODE_NO_PLE"] = "1"
    fd = _load_fd(1, a.attn_maxl)
    qm = gw.Q4nxModel(a.bundle)
    L = a.layer
    print(
        f"[validate] layer {L} "
        f"({'sliding' if gw.is_sliding(L) else 'FULL'} dh={gw.head_dim(L)}, "
        f"{'own kv' if gw.owns_kv(L) else 'shared kv'}) on a 1-layer build",
        flush=True,
    )

    if a.rebuild_cache or not os.path.exists(a.cache):
        rq.build_requant_cache(a.bundle, fd, a.cache, layers=[L])
    z = np.load(a.cache)
    assert int(z["layers"][0]) == L, (
        f"cache {a.cache} was packed for layer {int(z['layers'][0])}, not {L}; "
        f"pass --rebuild-cache"
    )

    # The hidden state fed to the layer, and the token embedding the PLE reads.
    # Both are arbitrary but must be the SAME on host and device.
    rng = np.random.default_rng(0)
    tok = 651
    x0 = qm.embed_rows("model.embed_tokens.weight", [tok])[0].astype(np.float32)
    x = x0.copy()

    emb_L = ple_embed(gw, qm, L, tok)
    want = cpu_layer(gw, qm, L, x[None, :], x0[None, :], emb_L, ple=not a.no_ple)

    import pyxrt as xrt

    dev = xrt.device(0)
    TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    xclbin = xrt.xclbin(a.xclbin)
    dev.register_xclbin(xclbin)
    ctx = xrt.hw_context(dev, xclbin.get_uuid())
    kern = xrt.kernel(ctx, "MLIR_AIE")
    g = kern.group_id
    HO = xrt.bo.host_only

    insts = np.fromfile(a.insts, dtype=np.uint32)
    ib = xrt.bo(dev, insts.nbytes, xrt.bo.cacheable, g(1))
    ib.write(insts, 0)
    ib.sync(TO)

    K, DH = fd.K, fd.DH_A
    n_w = fd.UNI_DEC * fd.W_TOTAL_BLOCKS + fd.UNI_LM * fd.VOCAB_W_BLOCKS
    n_w *= fd.BLOCK_BF16
    n_rms = fd.UNI_DEC * fd.RMS_LAYER + fd.UNI_DEC * fd.ROPE_W_LEN + K
    n_y = (
        fd.HOST_ROUNDS + fd.LAYER_RNDS
    ) * fd.PAYLOAD + fd.UNI_LM * fd.VOCAB_SIZE_PADDED
    n_kv = fd.UNI_DEC * fd.KV_LAYER

    x_bo = xrt.bo(dev, K * 2, HO, g(3))
    w_bo = xrt.bo(dev, n_w * 2, HO, g(4))
    r_bo = xrt.bo(dev, n_rms * 2, HO, g(5))
    y_bo = xrt.bo(dev, n_y * 2, HO, g(6))
    kv_bo = xrt.bo(dev, n_kv * 2, HO, g(7))
    pw_bo = px_bo = None
    if fd.PLE:
        pw_bo = xrt.bo(dev, fd.UNI_DEC * fd.PLE_LAYER * 2, HO, g(8))
        px_bo = xrt.bo(dev, K * 2, HO, g(9))

    # --- weights. The vocab region is never READ by this test -- it scores the
    # layer output, not logits -- but it must not be ZERO. The decode wave is
    # followed by UNI_LM vocab waves, and this design routes packets by a header
    # the kernels compute from the data (rms_residual.cc's
    # residual_add_aie_hdr). All-zero weights there can produce a zero header,
    # misroute a packet and hang a dispatch that is otherwise fine. Filling it
    # with a nonzero pattern makes the logits garbage, which is what we want. ---
    Wbuf = np.full(n_w, 0x1111, np.int16)
    Wbuf[: z["W"].size] = z["W"]
    w_bo.write(Wbuf, 0)
    w_bo.sync(TO)

    # --- rms slab: [5 norms | rope_w | final_norm] ---
    rms = np.concatenate(
        [
            z[f"RMS_{n}"][0].view(bfloat16)
            for n in ("in", "post_attn", "pre_ffn", "post_ffn", "post_ple")
        ]
    )
    cos, sin, dh = gw.rope_lut(0, L, rope_freqs=qm.rope_freqs())
    lut = np.zeros(DH, np.float32)
    lut[: dh // 2] = cos
    lut[DH // 2 : DH // 2 + dh // 2] = sin
    rope = np.concatenate(
        [
            lut.astype(bfloat16),
            z["QNORM"][0].view(bfloat16),
            z["KNORM"][0].view(bfloat16),
        ]
    )
    rbuf = np.concatenate([rms, rope, np.asarray(qm.globals()["final_norm"], bfloat16)])
    assert rbuf.size == n_rms, (rbuf.size, n_rms)
    r_bo.write(rbuf.view(np.int16), 0)
    r_bo.sync(TO)

    # --- PLE slab, with this token's per-layer embedding patched in ---
    if fd.PLE:
        ple = z["PLE"].view(bfloat16).copy()
        ple[fd.PLE_EMB_OFF : fd.PLE_NORMW_OFF] = np.asarray(emb_L, bfloat16)
        pw_bo.write(ple.view(np.int16), 0)
        pw_bo.sync(TO)
        px_bo.write(np.asarray(x0, bfloat16).view(np.int16), 0)
        px_bo.sync(TO)
    kv_bo.write(np.zeros(n_kv, np.int16), 0)
    kv_bo.sync(TO)
    x_bo.write(np.asarray(x, bfloat16).view(np.int16), 0)
    x_bo.sync(TO)

    args = [x_bo, w_bo, r_bo, y_bo, kv_bo] + ([pw_bo, px_bo] if fd.PLE else [])
    st = kern(3, ib, insts.nbytes, *args).wait(60000)
    if not str(st).endswith("COMPLETED"):
        raise SystemExit(f"dispatch state={st}")
    x_bo.sync(FROM)
    got = np.frombuffer(x_bo.map(), dtype=bfloat16, count=K).astype(np.float32)

    if a.dump:
        np.savez(a.dump, got=got, want=want, x=x, x0=x0, emb_L=emb_L, L=L)
        print(f"  dumped -> {a.dump}")
    c = cos_sim(got, want)
    rg, rw = float(np.sqrt((got**2).mean())), float(np.sqrt((want**2).mean()))
    print(f"  cos={c:.6f}  rms npu={rg:.4f} cpu={rw:.4f}  ratio={rg/(rw+1e-30):.4f}")
    # NaN FIRST. `nan < tol` is False, so without this a device that returns all
    # NaN scored PASS -- which it did, and it hid a real base-layer bug.
    n_bad = int((~np.isfinite(got)).sum())
    if n_bad or not np.isfinite(c):
        print(
            f"FAIL: layer {L} device output is not finite "
            f"({n_bad}/{got.size} non-finite, cos={c})"
        )
        return 1
    if c < a.tol:
        print(
            f"FAIL: layer {L} cos {c:.6f} < {a.tol}\n"
            "      A near-1.0 rms ratio with a low cosine is a CONTENT bug "
            "(a block fed the wrong tensor);\n"
            "      a bad ratio is a scale bug, which cosine cannot see."
        )
        return 1
    print(f"PASS: layer {L} matches the reference (cos {c:.6f} >= {a.tol})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
#
# --layers a,b runs a MULTI-LAYER build, which is what a KV-SHARED layer needs:
# layers at or past FIRST_KV_SHARED carry no k/v of their own and attend the
# cache of the last layer of their type below the boundary, so there is no
# one-layer expression to score them against. `--layers 14,19` puts the owning
# layer in slab 0 and the shared layer in slab 1, and DECODE_KV_SRC points
# slab 1's readback at slab 0.
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_DEC = _HERE / ".." / ".." / "fused_decode_ple"
# The model.q4nx bundle: an HF repo id, a directory holding it, or the file
# itself (gemma4_e2b_q4nx_weights.resolve_q4nx_model). Override with
# --bundle or Q4NX_MODEL_SOURCE.
BUNDLE = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Gemma4-E2B-IT-NPU2")


def _load_fd(uni_dec, attn_maxl, kv_src=None):
    """Import the builder with the env its module-level constants read."""
    os.environ.update(
        DECODE_MODEL="gemma4-e2b",
        VOCAB_CHUNK_I2="27",
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS=str(uni_dec),
        DECODE_GOLDEN="1",
        DECODE_GOLDEN_L=str(attn_maxl),
        DECODE_UNI_DEC=str(uni_dec),
    )
    if kv_src is not None:
        os.environ["DECODE_KV_SRC"] = ",".join(str(s) for s in kv_src)
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


def cpu_layer(gw, qm, L, x, x0, emb_L, ple=True, attn_l=1, kv=None):
    """The reference layer, at T=1 and position 0. Returns (out, (ke, v)).

    `kv` is the (ke, v) a KV-SHARED layer attends instead of projecting its own.
    Layers at or past FIRST_KV_SHARED have no k/v to project -- layer_weights()
    deliberately omits the tensors so a wrong sharing map raises KeyError rather
    than silently scoring unused weights -- so they must be given one.

    Mirrors check_vs_flm_reference.layer(); kept here rather than imported so
    the device is scored against the SAME expression the per-layer gate uses,
    with no shared mutable state between the two.

    attn_l is the device's ATTN_L. The design applies no validity mask -- it
    attends over all ATTN_L cache slots and appends this token at slot
    ATTN_L-1 -- so at position 0 the other slots are real zero keys competing
    in the softmax, not absent ones. Scoring a 16-slot device against a 1-key
    reference reads as a 0.71 failure when the device is in fact correct to
    0.998, which is exactly what it did until the two 1x4x1 attention bugs were
    fixed. attn_l=1 is the clean case, where the two expressions coincide.
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
    if kv is None:
        k = gw._rmsnorm((x1 @ w["k"].T).reshape(T, gw.N_KV_HEADS, dh), nm["k_norm"])
        v = gw._rmsnorm((x1 @ w["v"].T).reshape(T, gw.N_KV_HEADS, dh), None)
        ke = np.stack([gw.apply_rope(k[t], cos, sin, dh) for t in range(T)])
    else:
        ke, v = kv

    # The cache the device attends over: zeros, with this token at slot L-1.
    kh = np.zeros((attn_l, dh), np.float32)
    vh = np.zeros((attn_l, dh), np.float32)
    kh[attn_l - 1], vh[attn_l - 1] = ke[0, 0], v[0, 0]
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
        return o2.reshape(-1), (ke, v)
    gate = gw._gelu_tanh(o2 @ pw["inp_gate"].T) * pli
    o3 = o2 + gw._rmsnorm(gate @ pw["per_layer_projection"].T, nm["post_ple"])
    return (o3 * nm["out_scale"]).reshape(-1), (ke, v)


def cos_sim(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument(
        "--layers",
        help="comma-separated model layers to build as consecutive slabs, e.g. "
        "'14,19'. The LAST one is the layer being scored; the ones before it "
        "run to put their K/V in the cache, which is the only way to exercise "
        "a KV-shared layer (>= 15). Overrides --layer.",
    )
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--cache", default="/tmp/g4_layer.npz")
    ap.add_argument("--xclbin", default=str(_DEC / "decode.xclbin"))
    ap.add_argument("--insts", default=str(_DEC / "decode.insts.bin"))
    ap.add_argument("--attn-maxl", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.99)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument(
        "--retries",
        type=int,
        default=8,
        help="dispatch attempts before giving up, for the known PLE liveness "
        "bug. Pass 1 to measure the hang rate instead of working around it.",
    )
    ap.add_argument(
        "--prefix-dump",
        help="a --dump .npz from an earlier run of a PREFIX of --layers. Its "
        "device output becomes the hidden state the reference chain starts "
        "from, so the score isolates the last layer instead of compounding "
        "the whole chain's error. Without it a multi-layer run reports the "
        "composition: layer 14 at 0.992663 feeding layer 19 measures 0.975, "
        "of which 0.986 is layer 14's error AMPLIFIED by layer 19's 12288-wide "
        "FFN and only 0.9916 is layer 19's own.",
    )
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
    Ls = [int(t) for t in a.layers.split(",")] if a.layers else [a.layer]
    L = Ls[-1]  # the layer under test; the others only seed its cache

    # Which SLAB each slab attends. A layer that owns its cache reads its own;
    # a shared one reads the slab holding kv_source_layer(). The source has to
    # be present in this build, and earlier in it, or there is nothing to read.
    kv_src = []
    for i, Li in enumerate(Ls):
        src = gw.kv_source_layer(Li)
        if src not in Ls[: i + 1]:
            raise SystemExit(
                f"layer {Li} attends layer {src}'s KV cache, so {src} must be "
                f"built before it: try --layers {src},{Li}"
            )
        kv_src.append(Ls.index(src))

    fd = _load_fd(len(Ls), a.attn_maxl, kv_src=kv_src)
    qm = gw.Q4nxModel(a.bundle)
    for i, Li in enumerate(Ls):
        print(
            f"[validate] slab {i} = layer {Li} "
            f"({'sliding' if gw.is_sliding(Li) else 'FULL'} dh={gw.head_dim(Li)}, "
            f"{'own kv' if gw.owns_kv(Li) else f'kv from slab {kv_src[i]}'})"
            f"{'  <- SCORED' if i == len(Ls) - 1 else ''}",
            flush=True,
        )

    if a.rebuild_cache or not os.path.exists(a.cache):
        rq.build_requant_cache(a.bundle, fd, a.cache, layers=Ls)
    z = np.load(a.cache)
    assert list(z["layers"]) == Ls, (
        f"cache {a.cache} was packed for layers {list(z['layers'])}, not {Ls}; "
        f"pass --rebuild-cache"
    )

    # The hidden state fed to the layer, and the token embedding the PLE reads.
    # Both are arbitrary but must be the SAME on host and device.
    rng = np.random.default_rng(0)
    tok = 651
    x0 = qm.embed_rows("model.embed_tokens.weight", [tok])[0].astype(np.float32)
    x = x0.copy()

    # The device chains the layers in place -- slab i's output is written back to
    # arg0 and read by slab i+1 -- so the reference chains too, and every slab
    # keeps its K/V for whichever later slab shares it.
    embs = [ple_embed(gw, qm, Li, tok) for Li in Ls]
    caches = {}
    h = x[None, :]

    # A prefix dump swaps the reference's hidden state for the DEVICE's at the
    # prefix boundary. The K/V caches are still the reference's: they are only
    # read by a shared layer, they matched the device's to cos 0.999 when
    # measured, and the part that did not match is the weightless v-norm's
    # scale, which the post-attention rmsnorm cancels either way.
    cut = 0
    if a.prefix_dump:
        pz = np.load(a.prefix_dump)
        pre = list(pz["layers"]) if "layers" in pz else [int(pz["L"])]
        if pre != Ls[: len(pre)] or len(pre) >= len(Ls):
            raise SystemExit(
                f"--prefix-dump covers layers {pre}, which is not a proper "
                f"prefix of {Ls}"
            )
        cut = len(pre)
        print(f"  reference chain starts from the device's layer {pre[-1]} output")

    for i, Li in enumerate(Ls):
        if i == cut and cut:
            h = pz["got"].astype(np.float32)[None, :]
        h, caches[i] = cpu_layer(
            gw,
            qm,
            Li,
            h,
            x0[None, :],
            embs[i],
            ple=not a.no_ple,
            attn_l=fd.ATTN_L,
            kv=None if gw.owns_kv(Li) else caches[kv_src[i]],
        )
        h = h[None, :]
    want = h.reshape(-1)

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

    # --- rms slab: [per-slab 5 norms | per-slab rope_w | final_norm] ---
    # The two regions are each UNI_DEC slabs deep and are indexed independently
    # (_lb(RMS_LAYER) and _lut_off + _lb(ROPE_W_LEN)), so they are built as two
    # separate runs rather than interleaved per layer.
    rms = np.concatenate(
        [
            z[f"RMS_{n}"][i].view(bfloat16)
            for i in range(len(Ls))
            for n in ("in", "post_attn", "pre_ffn", "post_ffn", "post_ple")
        ]
    )
    rope = []
    for i, Li in enumerate(Ls):
        cos, sin, dh = gw.rope_lut(0, Li, rope_freqs=qm.rope_freqs())
        lut = np.zeros(DH, np.float32)
        lut[: dh // 2] = cos
        lut[DH // 2 : DH // 2 + dh // 2] = sin
        rope += [
            lut.astype(bfloat16),
            z["QNORM"][i].view(bfloat16),
            z["KNORM"][i].view(bfloat16),
        ]
    rbuf = np.concatenate(
        [rms, np.concatenate(rope), np.asarray(qm.globals()["final_norm"], bfloat16)]
    )
    assert rbuf.size == n_rms, (rbuf.size, n_rms)
    r_bo.write(rbuf.view(np.int16), 0)
    r_bo.sync(TO)

    # --- PLE slabs, with each layer's per-token embedding patched in ---
    if fd.PLE:
        ple = z["PLE"].view(bfloat16).copy().reshape(len(Ls), fd.PLE_LAYER)
        for i in range(len(Ls)):
            ple[i, fd.PLE_EMB_OFF : fd.PLE_NORMW_OFF] = np.asarray(embs[i], bfloat16)
        pw_bo.write(ple.reshape(-1).view(np.int16), 0)
        pw_bo.sync(TO)
        px_bo.write(np.asarray(x0, bfloat16).view(np.int16), 0)
        px_bo.sync(TO)
    kv_bo.write(np.zeros(n_kv, np.int16), 0)
    kv_bo.sync(TO)
    x_bo.write(np.asarray(x, bfloat16).view(np.int16), 0)
    x_bo.sync(TO)

    args = [x_bo, w_bo, r_bo, y_bo, kv_bo] + ([pw_bo, px_bo] if fd.PLE else [])
    # The PLE arm has a known liveness bug: about one dispatch in five times out
    # in the VOCAB drain, with the decode arm already finished -- X written back
    # and KV appended. DECODE_NO_PLE=1 is 20/20, so it takes the PLE feed.
    #
    # A DISPATCH THAT FOLLOWS A TIMEOUT COMPLETES BUT LIES. The correlation is
    # exact: ten runs of layer 14, the one that reported a retry scored 0.431573
    # and the nine that did not all scored 0.992663; layer 9 was 9/9 the same
    # way. Re-uploading every input BO does not help (the loop below already
    # does that) and neither does a fresh hw_context, so whatever the timeout
    # leaves behind is not in these buffers and is not cleared by a PDI reload.
    #
    # So a retry is used only to get the process to a verdict, and the verdict
    # is then INCONCLUSIVE, not PASS or FAIL. Scoring a retried dispatch is how
    # a clean 0.992663 gets reported as a 0.43 regression.
    retried = False
    for attempt in range(1, a.retries + 1):
        st = kern(3, ib, insts.nbytes, *args).wait(60000)
        if str(st).endswith("COMPLETED"):
            if attempt > 1:
                retried = True
                print(f"  (completed on attempt {attempt} of {a.retries})")
            break
        for b in args:
            b.sync(TO)
    else:
        raise SystemExit(f"dispatch state={st} after {a.retries} attempts")
    x_bo.sync(FROM)
    got = np.frombuffer(x_bo.map(), dtype=bfloat16, count=K).astype(np.float32)

    if a.dump:
        # the KV cache too: at pos 0 the correct attention output is just v, so
        # what the device stored there says whether the QKV path or the attend
        # path is at fault.
        kv_bo.sync(FROM)
        kv_out = np.frombuffer(kv_bo.map(), dtype=bfloat16, count=n_kv).astype(
            np.float32
        )
        # Y too. The QKV phase is host-drained, so Y[0 : HOST_ROUNDS*PAYLOAD] is
        # the device's own q|k|v (4096+1024+1024) -- the attention INPUT, and the
        # only direct observable between the projections and the o gather.
        y_bo.sync(FROM)
        y_out = np.frombuffer(y_bo.map(), dtype=bfloat16, count=n_y).astype(np.float32)
        np.savez(
            a.dump,
            got=got,
            want=want,
            x=x,
            x0=x0,
            emb_L=embs[-1],
            embs=np.stack(embs),
            L=L,
            layers=np.asarray(Ls),
            kv=kv_out,
            y=y_out,
        )
        print(f"  dumped -> {a.dump}")
    c = cos_sim(got, want)
    rg, rw = float(np.sqrt((got**2).mean())), float(np.sqrt((want**2).mean()))
    print(f"  cos={c:.6f}  rms npu={rg:.4f} cpu={rw:.4f}  ratio={rg/(rw+1e-30):.4f}")
    # NaN FIRST. `nan < tol` is False, so without this a device that returns all
    # NaN scored PASS -- which it did, and it hid a real base-layer bug.
    n_bad = int((~np.isfinite(got)).sum())
    if retried and not n_bad and np.isfinite(c):
        # Not PASS and not FAIL: a dispatch that followed a timeout completes
        # but does not carry a trustworthy answer, so this reading says nothing
        # about the layer either way. Exit 2 so a sweep can tell "re-run me"
        # from a real regression.
        print(
            f"INCONCLUSIVE: layer {L} needed a retry, and a dispatch after a "
            f"timeout cannot be scored (cos {c:.6f} is not evidence). Re-run."
        )
        return 2
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

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The 24-launch pre-pass (dflash_draft_prepass.py) as something a LOOP can call.
#
# The gates build it, run it once and exit; a decode loop runs it once per
# block, so the parts that cost minutes have to happen exactly once: the AWQ
# quantization of `fc` and of ten projection matrices, and the aircc compile.
# `PrepassRunner.__init__` does both, and `run(taps, positions)` is then a host
# fill plus one dispatch.
#
# THE ARRAY IS NOT SHARED. Three device programs are live in a DFlash loop --
# this ELF, the target's xclbin and the drafter's -- and the pre-pass takes the
# array to itself, so `run()` loads and unloads around each dispatch. That is
# a real per-block cost and it is why this module reports it (`t_load`,
# `t_run`) instead of hiding it: a loop that reloads an ELF every block is a
# correctness vehicle, not the shipping shape.

import os
import tempfile
import time


class PrepassRunner:
    """Compile the pre-pass once; run it per block.

    `run(taps, positions)` takes the target's tap rows for the block's context
    positions -- [n, 12800], TAP_SLOTS order, exactly `extract_context_feature`'s
    concatenation -- and their ABSOLUTE positions, and returns

        (target_hidden [n, 2560], k_ctx [5, n, 1024], v_ctx [5, n, 1024])

    with K already k_norm'd and rotated at those positions, which is the form
    `dflash_draft_decoder.seed_context_kv` wants.
    """

    def __init__(
        self, draft_weights=None, target_source=None, verbose=False, stack_size=16384
    ):
        import numpy as np
        from ml_dtypes import bfloat16

        import dflash_ctxkv_int4_builder as CK
        import dflash_draft_prepass as PP
        import dflash_int4 as I
        import dflash_int4_fc_builder as FC

        self.np, self.bf16 = np, bfloat16
        self.CK, self.PP, self.FC = CK, PP, FC
        self.lay = PP.prepass_arg_layout()
        self.C, self.D = PP.CTX_PAD, PP.D
        self.N, self.KVD, self.HD = PP.N_LAYERS, CK.KV_DIM, CK.HEAD_DIM
        self.P = PP.N_CHUNKS
        self.rows = self.C * CK.N_KV_HEADS

        I.paths()
        I.compile_int4_gemm_kernel()
        from shared.infra.external_kernels import compile_rope

        compile_rope()

        if draft_weights is None:
            from qwen3_4b_draft_weights import DraftWeights

            draft_weights = (
                DraftWeights(target_source=target_source)
                if target_source
                else DraftWeights()
            )
        dw = draft_weights

        # Quantize once. `fc` is 12800 -> 2560 split across N_CHUNKS launches
        # (matmul_int4_packed needs tile_k_l2 == K and K=12800 does not stage in
        # L2 -- see dflash_int4.build_int4_gemm_ir's assert).
        KC = FC.FC_IN // self.P
        self.fc_pk = []
        for W in FC.split_fc_weight(np.asarray(dw.fc()), self.P):
            q, s, z = I.awq_quantize(W)
            self.fc_pk.append(
                np.ascontiguousarray(I.pack_for_device(q, s, z, self.C, KC, self.D))
            )
        self.hn_w = np.asarray(dw.hidden_norm(), bfloat16)
        self.kpk, self.vpk = [], []
        for L in range(self.N):
            kw, vw = CK.layer_kv_weights(dw, L)
            for w, pk in ((kw, self.kpk), (vw, self.vpk)):
                q, s, z = I.awq_quantize(w)
                pk.append(
                    np.ascontiguousarray(
                        I.pack_for_device(q, s, z, self.C, self.D, self.KVD)
                    )
                )
        self.kn = [
            np.asarray(dw.bf16(f"layers.{L}.self_attn.k_norm.weight"), bfloat16)
            for L in range(self.N)
        ]

        from air.backend.xrt import XRTBackend

        self.backend = XRTBackend(
            verbose=verbose,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="dflash_draft_prepass",
            runtime_loop_tiling_sizes=[2, 2],
            stack_size=stack_size,
        )
        t0 = time.time()
        self.compiled = self.backend.compile(PP.build_prepass_module())
        self.t_compile = time.time() - t0
        self.t_load = 0.0
        self.t_run = 0.0
        self.n_run = 0

    def run(self, taps, positions):
        """Chunks over CTX_PAD rows, so a real prompt is not a rebuild.

        The pre-pass is built for a fixed CTX_PAD (32) and block 0's context is
        the WHOLE prompt (_dflash_upstream/model.py:219), which for anything
        past a toy prompt is longer than that. Every stage of this pass is
        ROW-INDEPENDENT, though -- `fc` and the k/v projections are GEMMs whose
        M is the row, `hidden_norm` and `k_norm` are per row, and RoPE reads one
        LUT entry per row -- so ceil(n/CTX_PAD) dispatches concatenated is
        exactly the same answer as one wide dispatch would be. That beats
        raising CTX_PAD, which restages fc's L2 (it already splits K across two
        launches to fit, see dflash_int4.build_int4_gemm_ir's assert).
        """
        np = self.np
        n = len(positions)
        assert np.asarray(taps).shape == (n, self.FC.FC_IN), np.asarray(taps).shape
        if n > self.C:
            parts = [
                self._run_one(taps[i : i + self.C], positions[i : i + self.C])
                for i in range(0, n, self.C)
            ]
            return (
                np.concatenate([p[0] for p in parts], axis=0),
                np.concatenate([p[1] for p in parts], axis=1),
                np.concatenate([p[2] for p in parts], axis=1),
            )
        return self._run_one(taps, positions)

    def _run_one(self, taps, positions):
        np, bf16 = self.np, self.bf16
        lay, C, D, N = self.lay, self.C, self.D, self.N
        n = len(positions)
        assert n <= C, f"pre-pass takes at most {C} context rows, got {n}"

        # Padded rows stay zero and their LUT positions stay 0: they are never
        # read back, and a zero row through fc + a norm is finite.
        t = np.zeros((C, self.FC.FC_IN), bf16)
        t[:n] = np.asarray(taps, bf16)
        As = self.FC.split_taps(t, self.P)
        pos = np.zeros(C, np.int64)
        pos[:n] = np.asarray(positions, np.int64)

        ins = [None] * lay["n_args"]
        for i, a in enumerate(lay["taps"]):
            ins[a] = As[i]
        for i, a in enumerate(lay["fc_w"]):
            ins[a] = self.fc_pk[i]
        for a in lay["fc_partial"] + lay["fc_fold"]:
            ins[a] = np.zeros((C, D), bf16)
        ins[lay["hn_w"]] = self.hn_w
        ins[lay["target_hidden"]] = np.zeros((C, D), bf16)
        for L in range(N):
            ins[lay["k_w"][L]] = self.kpk[L]
            ins[lay["v_w"][L]] = self.vpk[L]
            ins[lay["k_raw"][L]] = np.zeros((C, self.KVD), bf16)
            ins[lay["v_ctx"][L]] = np.zeros((C, self.KVD), bf16)
            ins[lay["k_norm_w"][L]] = self.kn[L]
            ins[lay["k_nrm"][L]] = np.zeros((self.rows, self.HD), bf16)
            ins[lay["k_ctx"][L]] = np.zeros((self.rows, self.HD), bf16)
        ins[lay["rope_lut"]] = self.CK.rope_lut(pos)

        import filelock

        t0 = time.time()
        with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
            fn = self.backend.load(self.compiled)
            t1 = time.time()
            res = fn(*ins)
            t2 = time.time()
        # Unload before the target's or the drafter's xclbin is dispatched
        # again: two resident device programs in one process is not something
        # this backend promises.
        self.backend.unload()
        self.t_load += t1 - t0
        self.t_run += t2 - t1
        self.n_run += 1

        f32 = np.float32
        th = np.asarray(res[lay["target_hidden"]]).reshape(C, D).astype(f32)[:n]
        k = np.stack(
            [
                np.asarray(res[lay["k_ctx"][L]]).reshape(C, self.KVD).astype(f32)[:n]
                for L in range(N)
            ]
        )
        v = np.stack(
            [
                np.asarray(res[lay["v_ctx"][L]]).reshape(C, self.KVD).astype(f32)[:n]
                for L in range(N)
            ]
        )
        return th, k, v

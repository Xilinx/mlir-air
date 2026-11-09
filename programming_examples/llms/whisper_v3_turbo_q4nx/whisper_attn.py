# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The encoder's self-attention: one FlashAttention ELF over all 1536 rows and
20 heads (flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py), on its
own hardware context next to the fused prefill device's.

It reads q | k | v from one [1536, 3840] buffer, the qkv GEMM's output where
it lies, and keys at or past the 1500th get no weight (valid_keys), so the
36 pad rows never enter a softmax. The schedule is the registry's for the
nearest measured non-causal head-dim-64 shapes."""

import os
from pathlib import Path

ROWS, HEADS, DH, FRAMES = 1536, 20, 64, 1500
NAME = "encoder_attn"
SCHEDULE = dict(
    q_in_segment=True,
    num_q_tiles=4,
    lqp=256,
    num_heads_per_unroll=4,
    num_cascade_stages=2,
    heads_rows=2,
    q_launch=2,
    fused_merge=True,
    qk_pretranspose=True,
)
BACKEND = {
    "verbose": False,
    "omit_while_true_loop": False,
    "omit_pingpong": "all",
    # one entry per launch dimension: the head groups' two and q_launch's
    "runtime_loop_tiling_sizes": [1, 1, 1],
    "output_format": "elf",
    "instance_name": "attention_bf16",
}


def cache_dir(build_dir):
    return Path(build_dir).resolve() / "attn"


def build(build_dir):
    """Compile the attention ELF into build_dir/attn (a KernelCache)."""
    out = cache_dir(build_dir)
    out.mkdir(parents=True, exist_ok=True)
    # the kernel object and the compiler's work files go to the working
    # directory
    cwd = os.getcwd()
    os.chdir(out)
    try:
        _build(out)
    finally:
        os.chdir(cwd)


def _build(out):
    from air_examples.flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module,
    )
    from air_examples.llms.shared.infra.cache import KernelCache
    from air_examples.llms.shared.infra.external_kernels import compile_attn_npu2

    cache = KernelCache(cache_dir=str(out))
    s = SCHEDULE
    mod = build_module(
        lk=ROWS,
        lkp=DH,
        lq=ROWS,
        lqp=s["lqp"],
        dk=DH,
        dv=DH,
        num_q_tiles=s["num_q_tiles"],
        num_cascade_stages=s["num_cascade_stages"],
        num_heads=HEADS,
        num_kv_heads=HEADS,
        causal=False,
        num_heads_per_unroll=s["num_heads_per_unroll"],
        fused_qkv=True,
        q_in_segment=s["q_in_segment"],
        fused_merge=s["fused_merge"],
        heads_rows=s["heads_rows"],
        q_launch=s["q_launch"],
        valid_keys=FRAMES,
    )
    compile_attn_npu2(
        head_dim=DH,
        bfp16=True,
        force=True,
        pretranspose_k=s["qk_pretranspose"],
        fused_merge=s["fused_merge"],
    )
    cache.compile_and_cache(NAME, mod, BACKEND)
    cache._save_manifest()


class EncoderAttention:
    """The attention ELF's context, run on caller-owned buffers."""

    def __init__(self, build_dir, device):
        import pyxrt as xrt
        from air.backend.xrt import XRTBackend

        from air_examples.llms.shared.infra.cache import KernelCache

        cache = KernelCache(cache_dir=str(cache_dir(build_dir)))
        if not cache.load_manifest() or NAME not in cache.artifacts:
            raise FileNotFoundError(f"no {NAME} in {cache_dir(build_dir)}; build it")
        self.backend = XRTBackend(**BACKEND, device=device)
        self.backend.load(cache.artifacts[NAME])
        self.kernel = self.backend.kernel
        self.out = xrt.ext.bo(device, ROWS * HEADS * DH * 2)
        self.run = None

    def start(self, qkv_bo):
        """Attention of q | k | v in qkv_bo (bf16 [1536, 3840]) into self.out."""
        import pyxrt as xrt

        if self.run is None or self.qkv is not qkv_bo:
            self.run = xrt.run(self.kernel)
            self.run.set_arg(0, qkv_bo)
            self.run.set_arg(1, self.out)
            self.qkv = qkv_bo
        self.run.start()

    def wait(self):
        import pyxrt as xrt

        self.run.wait2()
        self.out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

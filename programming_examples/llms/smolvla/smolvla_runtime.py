# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single-process NPU runtime for SmolVLA's vision encoder.

Reads top to bottom as the lifecycle it implements:

  1. `ensure_kernels`      compile the ELFs, or reuse the on-disk cache
  2. `VisionRuntime`       weights + ELFs + XRT context, built ONCE
     `.encode(images)`     -> (N, 64, 960) raw connector output
  3. `get_vision_runtime`  the process-wide singleton that makes (2) once-only

`encode` returns exactly what `vlm_with_expert.embed_image` returns: the RAW
connector output. lerobot applies the sqrt(960) scale afterwards -- do not
pre-apply it.

Why single-process
------------------
An earlier design ran the NPU stage in a *second* Python process, on the
assumption that the lerobot venv could not import `air`/`pyxrt`. It can, once
the mlir-air environment is sourced (`utils/env_setup.sh`), so everything fits
in one process. That matters: the two-process design paid, on EVERY inference,
process spawn + interpreter import (~185 ms) + a safetensors weight reload
(~320-390 ms) + ELF and XRT context load (~34 ms) + an npz round-trip. None of
it is NPU cost. Here the weights and the `KernelCache` (ELFs, XRT context,
device buffer objects) are built ONCE per process and reused, which is what a
deployment does.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_LLMS_DIR = _HERE.parent
if str(_LLMS_DIR) not in sys.path:
    sys.path.insert(0, str(_LLMS_DIR))

from shared.infra.cache import KernelCache, Profiler  # noqa: E402

MODEL_ID = "lerobot/smolvla_base"

# Resolved against THIS FILE, not the cwd: VisionRuntime is imported into
# lerobot's process, so where it finds its ELFs must not depend on who called
# it. Under build/ so that `make clean` is `rm -rf build/` and nothing else.
VISION_CACHE_DIR = str(_HERE / "build" / "vision_kernel_cache")
VISION_SEQ_LEN = 1024
VISION_KERNELS = {
    "vit_ln_qkv",
    "vit_o_ffn",
    "flash_attn",
    "layer_norm",
    "gemm_connector",
}


# ---------------------------------------------------------------------------
# 1. Kernel cache policy: compile once, then reuse
# ---------------------------------------------------------------------------


def ensure_kernels(cache, expected, compile_fn, tag="npu"):
    """Load the cached ELFs if the on-disk cache is COMPLETE, else compile.

    ELFs are build artifacts; recompiling them on every process start would
    dominate any end-to-end measurement and is not what a deployment does. The
    cache is reused only when its manifest resolves AND contains every kernel
    name in `expected`, so a cache left over from an older or partial kernel set
    still triggers a full rebuild rather than silently linking stale objects.

    Set SMOLVLA_FORCE_COMPILE=1 to always rebuild -- necessary after editing any
    kernel builder, since the manifest does not track source hashes.

    Returns True if a compile was performed.
    """
    force = os.environ.get("SMOLVLA_FORCE_COMPILE", "0") == "1"
    expected = set(expected)
    if not force and cache.load_manifest() and expected <= set(cache.artifacts):
        print(
            f"[{tag}] reusing {len(cache.artifacts)} cached ELFs from "
            f"{cache.cache_dir} (SMOLVLA_FORCE_COMPILE=1 to rebuild)",
            flush=True,
        )
        return False
    cache.artifacts.clear()
    compile_fn()
    return True


# ---------------------------------------------------------------------------
# 2. The runtime: weights + ELFs + XRT context, built once
# ---------------------------------------------------------------------------


def _to_chw_f32(img):
    """Accept a torch tensor (1,3,H,W) / (3,H,W) or a numpy array; return
    (3,H,W) float32 without importing torch here."""
    if hasattr(img, "detach"):
        arr = img.detach().float().cpu().numpy()
    else:
        arr = np.asarray(img)
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 4:
        assert arr.shape[0] == 1, arr.shape
        arr = arr[0]
    assert arr.ndim == 3 and arr.shape[0] == 3, arr.shape
    return np.ascontiguousarray(arr)


class VisionRuntime:
    """12-layer SigLIP ViT + modality connector on NPU, loaded ONCE.

    The expensive setup — safetensors weight load, ELF cache load, XRT context,
    per-layer static weight BOs — happens in `__init__` / the first `encode`
    and is then reused for every later inference in this process."""

    def __init__(
        self,
        cache_dir=VISION_CACHE_DIR,
        model_id=MODEL_ID,
        verbose=False,
        profile=False,
    ):
        from smolvla_vision_weights import load_vision_weights, SigLIPVisionConfig
        from smolvla_vision_encoder import compile_all_kernels

        t0 = time.perf_counter()
        self.cfg = SigLIPVisionConfig()
        self.weights = load_vision_weights(model_id, dtype=bfloat16, config=self.cfg)
        t_w = time.perf_counter()
        # profile=True records per-ELF XRT times into cache.profiler.kernel_times;
        # off by default so `make run` and `make verify` measure the shipping path.
        self.cache = KernelCache(
            cache_dir, verbose=verbose, profiler=Profiler(enabled=profile)
        )
        self.compiled = ensure_kernels(
            self.cache,
            VISION_KERNELS,
            lambda: compile_all_kernels(
                self.cache, self.cfg, VISION_SEQ_LEN, with_connector=True
            ),
            tag="npu-vision",
        )
        t_k = time.perf_counter()
        self.setup_ms = {
            "weight_load": (t_w - t0) * 1e3,
            "kernels": (t_k - t_w) * 1e3,
        }
        self.warmed = False

    def encode(self, images, attn_mode="flash", timings=None):
        """images: sequence of N camera tensors/arrays, each (1,3,512,512) or
        (3,512,512), ALREADY lerobot-preprocessed (resize_with_pad + [-1,1]).
        Returns (N, 64, 960) f32 RAW connector output."""
        from smolvla_vision_encoder import run_vit_encoder

        from smolvla_cpu_helpers import im2col_patch_embed

        t0 = time.perf_counter()
        arrs = [_to_chw_f32(im) for im in images]
        # The im2col patch-embed is a ONE-TIME host matmul per image, done
        # up front rather than inside the dispatch loop.
        patch_embeds = [
            im2col_patch_embed(
                a,
                self.weights.patch_w,
                self.weights.patch_b,
                self.weights.pos_embed,
                self.cfg.patch_size,
            )
            for a in arrs
        ]
        t_im2col = (time.perf_counter() - t0) * 1e3
        out = np.empty((len(arrs), 64, self.cfg.connector_out), np.float32)
        per_image_ms = []
        t_enc0 = time.perf_counter()
        for i, a in enumerate(patch_embeds):
            ti = time.perf_counter()
            res = run_vit_encoder(
                a,
                self.weights,
                self.cfg,
                self.cache,
                return_per_layer=False,
                do_connector=True,
                verbose=False,
                attn_mode=attn_mode,
            )
            out[i] = res["connector"]
            per_image_ms.append((time.perf_counter() - ti) * 1e3)
        t_enc = (time.perf_counter() - t_enc0) * 1e3
        self.warmed = True
        if timings is not None:
            timings["vision"] = {
                "wall_ms": (time.perf_counter() - t0) * 1e3,
                "t_im2col_ms": t_im2col,
                "t_encode_ms": t_enc,
                "t_image_ms": per_image_ms,
            }
        return out

    def warmup(self):
        """One throwaway encode so the first *measured* inference does not pay
        XRT context creation + BO alloc + the one-time static weight upload
        (measured 335 ms vs 148 ms warm per image)."""
        if not self.warmed:
            self.encode([np.zeros((3, 512, 512), np.float32)])
        return self


# ---------------------------------------------------------------------------
# 3. The singleton that makes step 2 happen exactly once per process
# ---------------------------------------------------------------------------

_vision_rt = None


def get_vision_runtime(**kw):
    """The whole point of single-process: build the vision runtime once and
    reuse its weights + warm KernelCache for every later inference."""
    global _vision_rt
    if _vision_rt is None:
        _vision_rt = VisionRuntime(**kw)
    return _vision_rt

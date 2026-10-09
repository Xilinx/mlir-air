# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA end-to-end inference with the vision encoder on NPU2.

SmolVLA is three stages: a SigLIP vision encoder, a SmolLM2-360M language
backbone, and a flow-matching action expert. All three were ported to NPU2 and
verified; **only the vision encoder ships on the NPU**, because it is the only
one measurably faster there (1.20x per image, 1.11x end to end). The other two
run lerobot's own unmodified CPU path. See README.md for the measurements.

How the splice works
--------------------
`run_hybrid_forward` wraps `policy.model.embed_prefix` and, for the duration of
that call, swaps `vlm_with_expert.embed_image` for one that serves results the
NPU already computed. Everything downstream -- the sqrt(960) scale, the pad and
attention masks, prefix assembly, the backbone, the action expert -- is
untouched lerobot code, so the comparison against the pure-CPU baseline is
honest. The wrapper is always restored in a `finally`.

All camera images are encoded in ONE call into the runtime rather than one call
per image, so the NPU's weights and ELFs are touched once per inference.

Run standalone:
    python3 smolvla_inference.py            # NPU vision (default)
    python3 smolvla_inference.py --cpu      # unmodified CPU model, for comparison
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# torch and lerobot are imported lazily, inside the functions that need them, so
# that `--compile-only` (and the compile lit test) runs with only the mlir-air
# toolchain installed -- no torch, no lerobot, no HuggingFace download. This
# mirrors the siblings, whose inference entry points are numpy-only at module
# scope for the same reason.

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# numpy/ml_dtypes/shared only at module scope -- no torch, no lerobot, so this
# stays importable for --compile-only.
from smolvla_runtime import VISION_CACHE_DIR  # noqa: E402

import types

# programming_examples/ is published as the air_examples package rather than put
# on sys.path: every directory under it would otherwise become a top-level
# module name and shadow any installed package that shares it.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
# CPU stages (SmolLM2 backbone, action expert). Their GEMMs are small (50 action
# tokens against a 241-token prefix) and do not scale with threads: on this
# 16-core / 32-thread machine the backbone plus 10 denoise steps take 273 ms at
# torch's default of 16 unbound threads, 240 ms with the threads bound to physical
# cores, and 215 ms with 8 bound threads.
#   SMOLVLA_CPU_BIND    (default 1) bind OpenMP threads to physical cores. Must be
#                       in the environment before torch loads its OpenMP runtime,
#                       so it is set here, at import, and skipped if torch is
#                       already imported or the user set OMP_PROC_BIND / OMP_PLACES.
#   SMOLVLA_CPU_THREADS (default 8) torch threads for the NPU path's CPU stages;
#                       0 keeps torch's default. The pure-CPU comparison arm always
#                       keeps torch's default thread count.
CPU_THREADS = int(os.environ.get("SMOLVLA_CPU_THREADS", "8"))
# The same cap for numpy's BLAS (the host im2col patch embed, the expert runtimes' matmuls), inside the
# NPU forward only. It defaults to every hardware thread (32 here); those threads sleep while the NPU
# runs and wake late, and next to torch's 8 bound threads they oversubscribe the cores: the 3-camera patch
# embed took 9 ms or 19 ms with a tail to 54 ms, and the whole vision stage 185 ms instead of 160 ms.
# Needs threadpoolctl (requirements.txt); without it nothing is capped and a warning is printed once.


def _blas_limit():
    from contextlib import nullcontext

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        global _WARNED_BLAS
        if not _WARNED_BLAS:
            print(
                "[smolvla] threadpoolctl not installed: numpy BLAS threads are not capped "
                "(the NPU vision stage runs ~25 ms slower); pip install threadpoolctl"
            )
            _WARNED_BLAS = True
        return nullcontext()
    return threadpool_limits(limits=CPU_THREADS, user_api="blas")


_WARNED_BLAS = False
if os.environ.get("SMOLVLA_CPU_BIND", "1") == "1" and "torch" not in sys.modules:
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

#   SMOLVLA_EXPERT_KV_MEMO (default 1) reuse the expert's cross-attention K/V
#                       projections across the 10 denoise steps. Each step
#                       re-projects the same 241-token prefix K/V through the
#                       expert's k_proj/v_proj; the inputs are compared exactly
#                       (torch.equal) so a changed prefix recomputes, and the
#                       output is bit-identical. NPU path only: the pure-CPU
#                       comparison arm stays unmodified.
EXPERT_KV_MEMO = os.environ.get("SMOLVLA_EXPERT_KV_MEMO", "1") == "1"

#   SMOLVLA_NPU_BACKBONE (default 0) EXPERIMENTAL: run the backbone's prefix fill
#                       on the NPU too (experimental/backbone_runtime.py), the
#                       default for run_hybrid_forward's npu_backbone.
#   SMOLVLA_NPU_ALL (default 0) EXPERIMENTAL: all three stages on the NPU: the two switches below
#                       plus the expert without host K/V packing (expert_runtime_v2.py). The
#                       --npu-all flag and `make run-all / verify-all / profile-all` set it.
NPU_ALL = os.environ.get("SMOLVLA_NPU_ALL", "0") == "1"
NPU_BACKBONE = os.environ.get("SMOLVLA_NPU_BACKBONE", "0") == "1" or NPU_ALL
#   SMOLVLA_NPU_EXPERT (default 0) EXPERIMENTAL: run the action expert's ten
#                       denoising calls on the NPU too (experimental/expert_runtime.py).
NPU_EXPERT = os.environ.get("SMOLVLA_NPU_EXPERT", "0") == "1" or NPU_ALL

DEFAULT_MODEL = "lerobot/smolvla_base"
DEFAULT_PROMPT = "pick up the cube"


def normalized_mse(chunk, ref) -> float:
    """MSE relative to the baseline action's own power: mean((chunk-ref)**2) /
    mean(ref**2).

    Magnitude-invariant, so the gate does not depend on the absolute scale of a
    particular prompt's action chunk -- a raw MSE threshold would silently drift
    PASS/FAIL as action magnitude changes.
    """
    chunk = np.asarray(chunk, np.float32)
    ref = np.asarray(ref, np.float32)
    power = float(np.mean(ref**2))
    return float(np.mean((chunk - ref) ** 2) / max(power, 1e-12))


def build_config(npu_vision: bool = True) -> dict:
    """Minimal config dict for the verify adapter and for reporting."""
    return {
        "model": DEFAULT_MODEL,
        "prompt": DEFAULT_PROMPT,
        "execution_model": "single-process (air/pyxrt in the lerobot venv)",
        "npu_stages": (
            (
                "vision"
                + (" + backbone" if NPU_BACKBONE else "")
                + (" + action expert" if NPU_EXPERT else "")
                + (" (experimental)" if NPU_BACKBONE or NPU_EXPERT else "")
            )
            if npu_vision
            else "none (pure CPU)"
        ),
    }


SYNTHETIC_SEED = 0
SYNTHETIC_GRAIN = 0.05


def _synthetic_image(shape, gen):
    """One random value per SigLIP patch, upsampled, plus a little pixel grain.

    Two properties, one from each term.

    Low frequency, drawn at the patch grid, keeps the model in a normal output
    regime. Pure per-pixel noise does not: the model answers it with a near-zero
    action chunk (rms 0.13 against 0.5-1.6 on real frames), and cosine on a
    near-zero reference is hypersensitive enough to fail the gate on arithmetic
    that is correct -- measured at 0.9869, below the 0.99 threshold.

    The grain restores rank. With a flat image every one of the 1024 patches is
    identical, so the patch-embedding GEMM is probed by a rank-1 operand and only
    its per-channel weight sums matter: shuffling weights within a channel leaves
    the action chunk bit-identical. Upsampled coarse noise alone reaches rank 27
    of 768, 5% grain reaches 297, and the shuffle then moves the chunk by 0.40.

    The grid comes from the encoder, not from `shape`: batch images are 256x256
    and lerobot resizes them to the encoder's 512x512, so a grid derived from
    the batch resolution would put 2x2 patches in each cell.
    """
    import torch
    import torch.nn.functional as F

    from smolvla_vision_weights import SigLIPVisionConfig

    grid = int(SigLIPVisionConfig().num_patches ** 0.5)
    coarse = torch.rand((1, shape[1], grid, grid), generator=gen, dtype=torch.float32)
    img = F.interpolate(coarse, size=shape[2:], mode="bilinear", align_corners=False)
    grain = torch.rand(shape, generator=gen, dtype=torch.float32) - 0.5
    return (img + SYNTHETIC_GRAIN * grain).clamp_(0.0, 1.0)


def build_oracle_batch(policy, prompt: str = DEFAULT_PROMPT, n_cameras=None):
    """Synthetic batch: seeded-random images, zero state, tokenized prompt.
    Deterministic, so the gate is reproducible.

    The images are random rather than zero for coverage: a flat image makes all
    1024 patches identical, and a rank-1 operand probes the patch-embedding GEMM
    along a single direction (see `_synthetic_image`). That they are noise rather
    than a photograph is fine -- this gate asks whether the NPU computes the same
    function as the CPU, not whether the model is any good, and `INPUT=real`
    covers the in-distribution case. [0, 1) is the range LeRobot decodes real
    frames into, so the downstream normalizer sees what it expects.

    Only the images are randomized. State feeds the CPU-only `state_proj`, so
    randomizing it would add a variable without covering any NPU kernel.

    n_cameras : keep only the first N camera feeds. None (default) keeps all
        three, which is what the gate runs -- do not change that. Fewer cameras
        is legal for the model (lerobot only rejects a batch with *every*
        camera missing) and needs no recompile: the camera count is how many
        times VisionRuntime.encode loops, and every ELF is built for one
        512x512 image regardless. Used by the camera-count controls.
    """
    import torch
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    cfg = policy.config
    feats = dict(cfg.input_features)
    if n_cameras is not None:
        cams = [k for k in feats if "images" in k]
        for k in cams[n_cameras:]:
            del feats[k]

    # One generator for the whole batch, so the images differ between cameras
    # the way real feeds do, and CAMERAS=1/2 sees the same frames as CAMERAS=3.
    gen = torch.Generator().manual_seed(SYNTHETIC_SEED)
    b = {}
    for k, f in feats.items():
        shape = (1, *tuple(f.shape))
        b[k] = (
            _synthetic_image(shape, gen)
            if "images" in k
            else torch.zeros(shape, dtype=torch.float32)
        )
    tok = policy.model.vlm_with_expert.processor.tokenizer(
        [prompt],
        padding="max_length",
        max_length=cfg.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    b[OBS_LANGUAGE_TOKENS] = tok["input_ids"]
    b[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].bool()
    return b


def fixed_noise(policy):
    """Deterministic zero noise, matching the oracle's action_chunk baseline.

    The action expert is a flow-matching denoiser seeded from noise, so a
    reproducible gate needs the noise pinned.
    """
    import torch

    return torch.zeros(
        (1, policy.config.chunk_size, policy.config.max_action_dim),
        dtype=torch.float32,
    )


_BACKBONE_RT: dict = {}


def get_backbone_runtime(policy, profile=False):
    """The NPU backbone runtime for `policy`, built once per process."""
    rt = _BACKBONE_RT.get(id(policy))
    if rt is None:
        experimental = str(_HERE / "experimental")
        if experimental not in sys.path:
            sys.path.insert(0, experimental)
        from backbone_runtime import BackboneRuntime

        rt = _BACKBONE_RT[id(policy)] = BackboneRuntime(policy, profile=profile)
    return rt


_EXPERT_RT: dict = {}


def get_expert_runtime(policy, profile=False):
    """The NPU action-expert runtime for `policy`, built once per process."""
    rt = _EXPERT_RT.get(id(policy))
    if rt is None:
        experimental = str(_HERE / "experimental")
        if experimental not in sys.path:
            sys.path.insert(0, experimental)
        if NPU_ALL or os.environ.get("SMOLVLA_NPU_EXPERT_V2", "0") == "1":
            # No host K/V packing: a prefix engine + step engine (experimental/expert_runtime_v2.py).
            from expert_runtime_v2 import ExpertRuntimeV2 as ExpertRuntime
        else:
            from expert_runtime import ExpertRuntime

        rt = _EXPERT_RT[id(policy)] = ExpertRuntime(policy, profile=profile)
    return rt


def warmup_npu():
    """Build the vision runtime and run one throwaway encode.

    Without this the first *measured* inference also pays XRT context creation,
    buffer-object allocation and the one-time static weight upload -- measured
    335 ms vs 148 ms warm, per image.
    """
    from smolvla_runtime import get_vision_runtime

    get_vision_runtime().warmup()


def run_hybrid_forward(
    batch,
    policy=None,
    noise=None,
    npu_vision: bool = True,
    timings: dict | None = None,
    npu_backbone: bool | None = None,
    npu_expert: bool | None = None,
):
    """Run one `predict_action_chunk`; return the (1, chunk, action_dim) chunk.

    npu_vision : encode the camera images with the NPU SigLIP ViT + connector
        instead of lerobot's CPU vision tower. False runs the model completely
        unmodified, which is the baseline the gate compares against.
    timings    : optional dict, filled with the NPU stage's phase timings.
    npu_backbone : also run the backbone's prefix fill on the NPU (experimental;
        default SMOLVLA_NPU_BACKBONE, never for the pure-CPU arm). Only the fill call is swapped: the expert's
        ten calls, which read the fill's KV cache, stay lerobot's CPU code.
    npu_expert : also run the action expert's ten denoising calls on the NPU
        (experimental; default SMOLVLA_NPU_EXPERT, never for the pure-CPU arm):
        each call's 16 layers are one launch (experimental/expert_runtime.py).
    """
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if policy is None:
        policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()

    if npu_backbone is None:
        npu_backbone = NPU_BACKBONE and npu_vision
    if npu_expert is None:
        npu_expert = NPU_EXPERT and npu_vision
    vwe = policy.model.vlm_with_expert
    orig_embed_prefix = policy.model.embed_prefix
    orig_embed_image = vwe.embed_image
    orig_forward = vwe.forward
    ref_dtype = next(policy.parameters()).dtype

    def _npu_forward(*a, **kw):
        # sample_actions keeps only the KV cache from the fill (`_, past_key_values
        # = ...`), so the hidden-state output is returned as None.
        embs, pkv = kw.get("inputs_embeds"), kw.get("past_key_values")
        if (
            npu_expert
            and not a
            and pkv is not None
            and embs is not None
            and embs[0] is None
        ):
            return _npu_expert_forward(embs[1], pkv, kw)
        if (
            not npu_backbone
            or a
            or pkv is not None
            or embs is None
            or embs[1] is not None
        ):
            return orig_forward(*a, **kw)
        assert kw.get("use_cache", True), "the NPU fill only produces the KV cache"
        t = time.perf_counter()
        cache = get_backbone_runtime(policy).fill(
            embs[0], kw["attention_mask"], kw["position_ids"]
        )
        if timings is not None:
            timings["backbone_npu_ms"] = (time.perf_counter() - t) * 1e3
        return [None, None], cache

    def _npu_expert_forward(suffix, pkv, kw):
        # lerobot's expert call returns the final-normed suffix states and the
        # unchanged cache (sample_actions' denoise_step reads outputs[1]).
        t = time.perf_counter()
        f32 = lambda x: x.detach().float().numpy()  # noqa: E731

        def kv():
            n = len(pkv.layers)
            k = np.stack([f32(pkv.layers[i].keys[0].transpose(0, 1)) for i in range(n)])
            v = np.stack(
                [f32(pkv.layers[i].values[0].transpose(0, 1)) for i in range(n)]
            )
            return k.reshape(n, k.shape[1], -1), v.reshape(n, v.shape[1], -1)

        out = get_expert_runtime(policy)(
            f32(suffix[0]),
            kv,
            kw["attention_mask"][0].numpy(),
            kw["position_ids"][0].numpy(),
            kv_src=pkv,
        )
        if timings is not None:
            timings["expert_npu_ms"] = (
                timings.get("expert_npu_ms", 0.0) + (time.perf_counter() - t) * 1e3
            )
        return [None, torch.from_numpy(out).to(suffix.dtype)[None]], pkv

    def _wrapped_embed_prefix(*a, **kw):
        # Two swaps, not one, and the outer one is load-bearing for SPEED, not
        # for correctness: all N images are encoded in ONE runtime call.
        # Encoding them one at a time from embed_image alone is simpler -- one
        # swap, no counter, no ordering assumption -- and passes the gate too.
        # It also costs the entire win: 818/834/838 ms batched vs 882/914/963 ms
        # lazy, against a 913 ms pure-CPU baseline (measured before the gate
        # input changed; the comparison stands, the absolute numbers drift).
        #
        # WHY it costs that is NOT established. The obvious suspect at the
        # time -- a host BLAS thread clamp being entered per image instead of
        # once -- was ruled out and the clamp has since been deleted for
        # measuring as a no-op. The largest single delta is im2col, 20 -> 47
        # ms, which would fit a cache-locality story (batched, patch_w stays
        # hot; interleaved, a 12-layer ViT pass evicts it) -- but that is a
        # guess, not a measurement.
        #
        # So: keep the batching because the number is real, and do not trust
        # any explanation of it, including this comment, without re-measuring.
        from smolvla_runtime import get_vision_runtime

        images = kw["images"] if "images" in kw else a[0]
        conn = get_vision_runtime().encode(images, timings=timings)
        served = {"i": 0}

        def _npu_embed_image(image):
            # lerobot's embed_prefix iterates `images` in order, so the i-th
            # call corresponds to conn[i]. That is an assumption about someone
            # else's loop, so it is checked rather than trusted: a reordering
            # or an extra call raises here instead of silently pairing an
            # image with another camera's embedding.
            i = served["i"]
            assert i < len(conn), (
                f"embed_image called {i + 1}x but only {len(conn)} images were "
                "encoded -- lerobot's embed_prefix no longer consumes `images` "
                "one-for-one in order"
            )
            served["i"] += 1
            emb = torch.from_numpy(np.ascontiguousarray(conn[i])).to(ref_dtype)
            return emb[None, ...].expand(image.shape[0], -1, -1)

        vwe.embed_image = _npu_embed_image
        try:
            out = orig_embed_prefix(*a, **kw)
        finally:
            vwe.embed_image = orig_embed_image
        assert served["i"] == len(conn), (
            f"encoded {len(conn)} images on the NPU but embed_prefix consumed "
            f"{served['i']} -- some camera fell back to the CPU tower silently"
        )
        return out

    prev_threads = torch.get_num_threads()
    memos = _install_expert_kv_memo(policy) if EXPERT_KV_MEMO else []
    for m in memos:
        m.enabled = npu_vision
    if npu_vision:
        policy.model.embed_prefix = _wrapped_embed_prefix
        if CPU_THREADS > 0:
            torch.set_num_threads(CPU_THREADS)
    if npu_backbone or npu_expert:
        vwe.forward = _npu_forward
    try:
        policy.reset()
        from contextlib import nullcontext

        with torch.no_grad(), (
            _blas_limit() if npu_vision and CPU_THREADS > 0 else nullcontext()
        ):
            chunk = policy.predict_action_chunk(batch, noise=noise)
    finally:
        policy.model.embed_prefix = orig_embed_prefix
        vwe.embed_image = orig_embed_image
        vwe.forward = orig_forward
        torch.set_num_threads(prev_threads)
        for m in memos:
            m.enabled = False
            m.clear()

    return chunk.detach().float().numpy()  # (1, chunk_size, action_dim)


def _install_expert_kv_memo(policy):
    """Wrap the expert's k_proj / v_proj in a one-entry exact-input memo.

    Returns the wrappers so the caller can switch them on for the NPU path and
    clear them afterwards. Idempotent per policy. Self-attention expert layers
    see a different input every step, so their wrappers never hit; the cost
    there is one shape check and a failed torch.equal.
    """
    import torch

    class _Memo(torch.nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.lin = lin
            self.enabled = False
            self.clear()

        @property
        def weight(self):  # lerobot reads k_proj.weight.dtype
            return self.lin.weight

        def clear(self):
            self._x = None
            self._y = None

        def forward(self, x):
            if not self.enabled:
                return self.lin(x)
            if (
                self._x is not None
                and self._x.shape == x.shape
                and self._x.dtype == x.dtype
                and torch.equal(self._x, x)
            ):
                return self._y
            y = self.lin(x)
            self._x, self._y = x.clone(), y
            return y

    memos = getattr(policy, "_expert_kv_memos", None)
    if memos is None:
        memos = []
        for layer in policy.model.vlm_with_expert.lm_expert.layers:
            for name in ("k_proj", "v_proj"):
                m = _Memo(getattr(layer.self_attn, name))
                setattr(layer.self_attn, name, m)
                memos.append(m)
        policy._expert_kv_memos = memos
    return memos


def compile_only(cache_dir: str = VISION_CACHE_DIR) -> int:
    """Build every vision ELF through AIR -> AIE -> aiecc -> Peano.

    No NPU dispatch and no HuggingFace download, so this runs anywhere the
    toolchain is installed -- it is the compile smoke test the CI lit file
    drives, and it must not need the device, the network, torch or lerobot.

    Writes to VISION_CACHE_DIR, which smolvla_runtime resolves against its own
    __file__ rather than the cwd -- that is what lets `make compile` and
    `make run` share one cache no matter which directory each runs from.
    """
    # smolvla_vision_encoder puts programming_examples/ and llms/ on sys.path at
    # import time, so it has to come before anything under `shared.`.
    from smolvla_vision_encoder import compile_all_kernels
    from smolvla_vision_weights import SigLIPVisionConfig
    from air_examples.llms.shared.infra.cache import KernelCache, Profiler
    from smolvla_runtime import VISION_N_IMAGES

    cfg = SigLIPVisionConfig()
    cache = KernelCache(cache_dir, verbose=False, profiler=Profiler())
    print(f"Compiling SmolVLA vision kernels into {cache_dir}/ ...")
    # VISION_N_IMAGES, not 1: the ELFs are shape-specialized on the batched row
    # count, so building at 1 would leave this check passing on a configuration
    # the runtime never asks for -- and rebuilding everything on the first run.
    compile_all_kernels(
        cache,
        cfg,
        seq_len=cfg.num_patches,
        fused=True,
        with_connector=True,
        n_images=VISION_N_IMAGES,
    )
    cache._save_manifest()
    print(f"Compiled {len(cache.artifacts)} ELFs: {sorted(cache.artifacts)}")
    print("Compilation passed.")
    return 0


def run_profile(
    prompt: str = DEFAULT_PROMPT,
    reps: int = 5,
    n_cameras: int = 3,
    npu_backbone: bool = False,
    npu_expert: bool = False,
) -> int:
    """Pure CPU vs NPU vision, measured so the comparison is worth reporting.

    Four things the naive "run one, then run the other" does wrong, and what
    this does instead:

      one process        both arms share the loaded policy, so neither pays a
                         model load inside its timed region
      both warmed        a discarded forward per arm first. Timing the CPU arm
                         cold while the NPU arm is warm is not a comparison;
                         measured, the first CPU forward is 1.07x the warm one
      interleaved        CPU, NPU, CPU, NPU... so thermal or scheduler drift
                         hits both arms equally instead of whichever ran last
      median of N        process-to-process spread on this machine is ~10-15%;
                         a single reading lands anywhere in it
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from smolvla_runtime import get_vision_runtime

    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()
    batch = build_oracle_batch(
        policy, prompt, n_cameras=None if n_cameras == 3 else n_cameras
    )
    noise = fixed_noise(policy)
    rt = get_vision_runtime(profile=True)
    rt.warmup()

    # Stage timers. `vlm_with_expert.forward` is the boundary for both CPU
    # stages: the fill call (past_key_values=None) is the backbone, the ten
    # later calls are the action expert. It is never swapped, so one hook
    # serves both arms. `embed_image` is the vision boundary, but the NPU arm
    # replaces it, so that arm reads its time from the runtime's own timings.
    vwe = policy.model.vlm_with_expert
    acc = {"cpu_vision": [], "backbone": [], "expert": []}
    orig_embed_image, orig_fwd = vwe.embed_image, vwe.forward
    cur: dict = {}

    def timed_embed_image(img):
        t = time.perf_counter()
        r = orig_embed_image(img)
        cur["vision"] = cur.get("vision", 0.0) + (time.perf_counter() - t) * 1e3
        return r

    def timed_forward(*a, **kw):
        t = time.perf_counter()
        r = orig_fwd(*a, **kw)
        d = (time.perf_counter() - t) * 1e3
        key = "backbone" if kw.get("past_key_values") is None else "expert"
        cur[key] = cur.get(key, 0.0) + d
        return r

    vwe.embed_image, vwe.forward = timed_embed_image, timed_forward

    def once(npu: bool, npu_bb: bool = False, npu_ex: bool = False) -> tuple:
        cur.clear()
        t: dict = {}
        if npu_ex:
            # Every rep feeds the same batch; a robot's next observation is a new prefix.
            get_expert_runtime(policy).forget_prefix()
        t0 = time.perf_counter()
        chunk = run_hybrid_forward(
            batch,
            policy=policy,
            noise=noise,
            npu_vision=npu,
            timings=t,
            npu_backbone=npu_bb,
            npu_expert=npu_ex,
        )
        wall = (time.perf_counter() - t0) * 1e3
        vision = t["vision"]["wall_ms"] if npu else cur.get("vision", 0.0)
        backbone = t["backbone_npu_ms"] if npu_bb else cur.get("backbone", 0.0)
        expert = t["expert_npu_ms"] if npu_ex else cur.get("expert", 0.0)
        return wall, vision, backbone, expert, chunk

    # Experimental arms beyond NPU vision: (label, chunk key, npu_backbone, npu_expert).
    extra = ([("NPU v+bb", "npu_bb", True, False)] if npu_backbone else []) + (
        [("NPU all", "npu_all", True, True)] if npu_expert else []
    )
    xrows = {
        key: {"wall": [], "vision": [], "backbone": [], "expert": []}
        for _, key, _, _ in extra
    }
    try:
        once(False)
        once(True)  # warm both arms, discard
        for _, _, b_, e_ in extra:
            once(True, b_, e_)
        rt.cache.profiler.kernel_times.clear()  # drop the warmup dispatches

        cpu, npu, vis, cvis, bb, ex = [], [], [], [], [], []
        npu_bb_cpu, npu_ex = [], []
        chunks = {}
        for _ in range(reps):
            w, v, b, e, chunks["cpu"] = once(False)
            cpu.append(w)
            cvis.append(v)
            bb.append(b)
            ex.append(e)
            w, v, b, e, chunks["npu"] = once(True)
            npu.append(w)
            vis.append(v)
            npu_bb_cpu.append(b)
            npu_ex.append(e)
            for _, key, b_, e_ in extra:
                row = once(True, b_, e_)
                chunks[key] = row[4]
                for k, x in zip(xrows[key], row[:4]):
                    xrows[key][k].append(x)
    finally:
        vwe.embed_image, vwe.forward = orig_embed_image, orig_fwd

    med = lambda v: float(np.median(v))  # noqa: E731
    n_cam = sum(1 for k in batch if "images" in k)
    W = 36
    print()
    print("=" * 74)
    print(f"SmolVLA profile — {reps} interleaved reps, both arms warmed, one process")
    print("=" * 74)

    print(f"  {'end to end':{W}s} {'median':>9s} {'min':>9s} {'max':>9s}")
    print(f"  {'-' * W} {'-' * 9} {'-' * 9} {'-' * 9}")
    print(
        f"  {'pure CPU (unmodified lerobot)':{W}s} "
        f"{med(cpu):9.1f} {min(cpu):9.1f} {max(cpu):9.1f}"
    )
    print(
        f"  {'NPU vision + CPU backbone/expert':{W}s} "
        f"{med(npu):9.1f} {min(npu):9.1f} {max(npu):9.1f}"
    )
    print(f"\n  speedup (median)  {med(cpu) / med(npu):.3f}x\n")

    # Per stage, both arms. Only the vision row differs -- the backbone and the
    # expert are the same unmodified CPU code in both, so their times are
    # carried across and the table shows what the swap did and did not touch.
    print(f"  {'per stage':{W}s} {'CPU':>9s} {'NPU run':>9s} {'speedup':>9s}")
    print(f"  {'-' * W} {'-' * 9} {'-' * 9} {'-' * 9}")
    print(
        f"  {f'vision: SigLIP + connector (x{n_cam})':{W}s} "
        f"{med(cvis):9.1f} {med(vis):9.1f} {med(cvis) / med(vis):8.2f}x"
    )
    print(
        f"  {'backbone: SmolLM2-360M (x1)':{W}s} {med(bb):9.1f} {med(bb):9.1f}"
        f"{'  CPU both':>10s}"
    )
    print(
        f"  {'action expert (x10 denoise steps)':{W}s} {med(ex):9.1f} {med(ex):9.1f}"
        f"{'  CPU both':>10s}"
    )

    if extra:

        def cmp(name):
            c, r = chunks[name].ravel().astype(np.float64), chunks[
                "cpu"
            ].ravel().astype(np.float64)
            cos = float(c @ r / (np.linalg.norm(c) * np.linalg.norm(r)))
            return (
                cos,
                normalized_mse(chunks[name], chunks["cpu"]),
                float(np.abs(c - r).max()),
            )

        print()
        print(
            "  EXPERIMENTAL arms (interleaved): NPU v+bb = NPU vision + backbone, CPU expert;"
            " NPU all = every stage on the NPU"
        )
        print(
            f"  {'':{W}s} {'CPU':>9s} {'NPU vis':>9s}"
            + "".join(f" {lab:>9s}" for lab, _, _, _ in extra)
        )
        print(f"  {'-' * W} {'-' * 9} {'-' * 9}" + f" {'-' * 9}" * len(extra))
        for label, a, b2, k in (
            ("end to end (median)", cpu, npu, "wall"),
            ("vision", cvis, vis, "vision"),
            ("backbone fill", bb, npu_bb_cpu, "backbone"),
            ("action expert (x10)", ex, npu_ex, "expert"),
        ):
            print(
                f"  {label:{W}s} {med(a):9.1f} {med(b2):9.1f}"
                + "".join(f" {med(xrows[key][k]):9.1f}" for _, key, _, _ in extra)
            )
        print(
            f"\n  speedup vs CPU (median)  NPU vis {med(cpu) / med(npu):.3f}x"
            + "".join(
                f"   {lab} {med(cpu) / med(xrows[key]['wall']):.3f}x"
                for lab, key, _, _ in extra
            )
        )
        for name in ["npu"] + [key for _, key, _, _ in extra]:
            cos, nmse, mx = cmp(name)
            print(
                f"  action chunk vs pure CPU  {name:7s} cosine {cos:.6f}  nMSE {nmse:.6f}  max|d| {mx:.4f}"
            )
        if npu_expert:
            xt = get_expert_runtime(policy).timings
            n_ex = max(xt.get("calls", 1), 1)
            print(
                f"  NPU expert per call: {xt.get('run_ms', 0) / n_ex:.2f} ms dispatch (write + device + read back);"
                f" per chunk: prefix repack {xt.get('prefix_ms', 0) / max(xt.get('prefixes', 1), 1):.1f} ms"
                f" (first includes the full pack), of which buffer sync"
                f" {xt.get('prefix_sync_ms', 0) / max(xt.get('prefixes', 1) - 1, 1):.1f} ms"
            )

    kt = rt.cache.profiler.kernel_times
    if kt:
        n_img = reps * n_cam * (1 + len(extra))
        print()
        print(
            f"  {f'NPU device time, per image (of {n_cam})':{W}s} "
            f"{'calls':>9s} {'ms/image':>9s}"
        )
        print(f"  {'-' * W} {'-' * 9} {'-' * 9}")
        total = 0.0
        for name in sorted(kt, key=lambda k: -sum(kt[k])):
            ms = sum(kt[name]) * 1e3 / n_img
            total += ms
            print(f"  {name:{W}s} {len(kt[name]) // n_img:9d} {ms:9.2f}")
        print(f"  {'TOTAL device / image':{W}s} {'':9s} {total:9.2f}")
        print()
        print(
            f"  x{n_cam} images = {total * n_cam:.1f} ms device, "
            f"of the {med(vis):.1f} ms vision stage "
            f"({total * n_cam / med(vis) * 100:.0f}% device, "
            f"{med(vis) - total * n_cam:.1f} ms host)"
        )

    # Machine-readable line for llms/bench/extract_perf.py, which the nightly
    # profile lit pipes this output through. Kept to one line, in the shape the
    # other models' summaries use, so the shared extractor needs one regex and
    # no model-specific branch. prompt_len is the prefix the backbone attends
    # over (3 x 64 image + 48 language + 1 state), the analogue of the siblings'
    # prefill length.
    print()
    print(
        f"  Action chunk (NPU vision): {med(npu):.1f} ms, "
        f"prompt_len={64 * n_cam + 49}"
    )
    print("=" * 74)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cpu", action="store_true", help="run the unmodified CPU model instead"
    )
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="build every vision ELF and exit; no NPU dispatch, no download",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="CPU vs NPU, interleaved and warmed, with the per-ELF breakdown",
    )
    ap.add_argument("--reps", type=int, default=5, help="reps per arm for --profile")
    ap.add_argument(
        "--npu-backbone",
        action="store_true",
        help="EXPERIMENTAL: also run the backbone fill on the NPU (a third arm under --profile)",
    )
    ap.add_argument(
        "--npu-expert",
        action="store_true",
        help="EXPERIMENTAL: also run the action expert on the NPU (an all-NPU arm under --profile)",
    )
    ap.add_argument(
        "--npu-all",
        action="store_true",
        help="EXPERIMENTAL: vision + backbone + action expert all on the NPU (implies --npu-backbone "
        "--npu-expert, expert without host K/V packing); needs the expert engine ELFs, see the README",
    )
    ap.add_argument(
        "--input",
        choices=("synthetic", "real"),
        default="synthetic",
        help="synthetic = seeded-random images (no download); real = a LeRobot dataset",
    )
    ap.add_argument("--cameras", type=int, default=3, help="camera feeds to supply")
    ap.add_argument("--dataset", default="lerobot/droid_100", help="for --input real")
    args = ap.parse_args()
    npu_vision = not args.cpu
    if args.npu_all:
        if args.cpu:
            ap.error("--npu-all and --cpu are opposites")
        global NPU_ALL, NPU_BACKBONE, NPU_EXPERT
        NPU_ALL = NPU_BACKBONE = NPU_EXPERT = True

    if args.compile_only:
        return compile_only()
    if args.profile:
        # Timing only. Pixel values do not change how much the NPU computes, so
        # --input has no meaning here; camera count does, since it is the host
        # loop count.
        return run_profile(
            args.prompt,
            args.reps,
            args.cameras,
            args.npu_backbone or NPU_BACKBONE,
            args.npu_expert or NPU_EXPERT,
        )

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()
    n_cam = None if args.cameras == 3 else args.cameras
    if args.input == "real":
        import smolvla_dataset

        idx, batch = next(
            smolvla_dataset.batches(
                policy, args.prompt, args.cameras, args.dataset, n_frames=1
            )
        )
        print(f"[smolvla] input        : {args.dataset} frame {idx}")
    else:
        batch = build_oracle_batch(policy, args.prompt, n_cameras=n_cam)
        print("[smolvla] input        : synthetic (seeded random)")
    print(f"[smolvla] cameras      : {args.cameras}")
    if npu_vision:
        warmup_npu()

    timings: dict = {}
    t0 = time.perf_counter()
    chunk = run_hybrid_forward(
        batch,
        policy=policy,
        noise=fixed_noise(policy),
        npu_vision=npu_vision,
        timings=timings,
        npu_backbone=args.npu_backbone or NPU_BACKBONE,
        npu_expert=args.npu_expert or NPU_EXPERT,
    )
    wall_ms = (time.perf_counter() - t0) * 1e3

    cfg = build_config(npu_vision)
    print(f"[smolvla] NPU stages   : {cfg['npu_stages']}")
    print(f"[smolvla] execution    : {cfg['execution_model']}")
    print(f"[smolvla] wall clock   : {wall_ms:.1f} ms")
    print(f"[smolvla] action chunk : {chunk.shape}  |x|max={np.abs(chunk).max():.4f}")
    for k, v in sorted(timings.items()):
        print(f"[smolvla]   {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

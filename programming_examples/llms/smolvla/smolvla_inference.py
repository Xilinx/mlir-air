# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA end-to-end inference with the vision encoder on NPU2.

SmolVLA is three stages: a SigLIP vision encoder, a SmolLM2-360M language
backbone, and a flow-matching action expert. All three were ported to NPU2 and
verified; **only the vision encoder ships on the NPU**, because it is the only
one measurably faster there (1.19x per image, 1.07x end to end). The other two
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
        "npu_stages": "vision" if npu_vision else "none (pure CPU)",
    }


def build_oracle_batch(policy, prompt: str = DEFAULT_PROMPT, n_cameras=None):
    """The same synthetic batch the oracle dumper uses: zero images and state,
    tokenized prompt. Deterministic, so the gate is reproducible.

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

    b = {}
    for k, f in feats.items():
        b[k] = torch.zeros((1, *tuple(f.shape)), dtype=torch.float32)
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
):
    """Run one `predict_action_chunk`; return the (1, chunk, action_dim) chunk.

    npu_vision : encode the camera images with the NPU SigLIP ViT + connector
        instead of lerobot's CPU vision tower. False runs the model completely
        unmodified, which is the baseline the gate compares against.
    timings    : optional dict, filled with the NPU stage's phase timings.
    """
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if policy is None:
        policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()

    vwe = policy.model.vlm_with_expert
    orig_embed_prefix = policy.model.embed_prefix
    orig_embed_image = vwe.embed_image
    ref_dtype = next(policy.parameters()).dtype

    def _wrapped_embed_prefix(*a, **kw):
        # Two swaps, not one, and the outer one is load-bearing for SPEED, not
        # for correctness: all N images are encoded in ONE runtime call.
        # Encoding them one at a time from embed_image alone is simpler -- one
        # swap, no counter, no ordering assumption -- and holds the gate at
        # cosine 0.99900. It also costs the entire win: 818/834/838 ms batched
        # vs 882/914/963 ms lazy, against a 913 ms pure-CPU baseline.
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

    if npu_vision:
        policy.model.embed_prefix = _wrapped_embed_prefix
    try:
        policy.reset()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch, noise=noise)
    finally:
        policy.model.embed_prefix = orig_embed_prefix
        vwe.embed_image = orig_embed_image

    return chunk.detach().float().numpy()  # (1, chunk_size, action_dim)


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
    from shared.infra.cache import KernelCache, Profiler

    cfg = SigLIPVisionConfig()
    cache = KernelCache(cache_dir, verbose=False, profiler=Profiler())
    print(f"Compiling SmolVLA vision kernels into {cache_dir}/ ...")
    compile_all_kernels(
        cache, cfg, seq_len=cfg.num_patches, fused=True, with_connector=True
    )
    cache._save_manifest()
    print(f"Compiled {len(cache.artifacts)} ELFs: {sorted(cache.artifacts)}")
    print("Compilation passed.")
    return 0


def run_profile(prompt: str = DEFAULT_PROMPT, reps: int = 5, n_cameras: int = 3) -> int:
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

    def once(npu: bool) -> tuple:
        cur.clear()
        t: dict = {}
        t0 = time.perf_counter()
        run_hybrid_forward(batch, policy=policy, noise=noise, npu_vision=npu, timings=t)
        wall = (time.perf_counter() - t0) * 1e3
        vision = t["vision"]["wall_ms"] if npu else cur.get("vision", 0.0)
        return wall, vision, cur.get("backbone", 0.0), cur.get("expert", 0.0)

    try:
        once(False)
        once(True)  # warm both arms, discard
        rt.cache.profiler.kernel_times.clear()  # drop the warmup dispatches

        cpu, npu, vis, cvis, bb, ex = [], [], [], [], [], []
        for _ in range(reps):
            w, v, b, e = once(False)
            cpu.append(w)
            cvis.append(v)
            bb.append(b)
            ex.append(e)
            w, v, _, _ = once(True)
            npu.append(w)
            vis.append(v)
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

    kt = rt.cache.profiler.kernel_times
    if kt:
        n_img = reps * n_cam
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
        "--input",
        choices=("synthetic", "real"),
        default="synthetic",
        help="synthetic = all-zero images (no download); real = a LeRobot dataset",
    )
    ap.add_argument("--cameras", type=int, default=3, help="camera feeds to supply")
    ap.add_argument("--dataset", default="lerobot/droid_100", help="for --input real")
    args = ap.parse_args()
    npu_vision = not args.cpu

    if args.compile_only:
        return compile_only()
    if args.profile:
        # Timing only. Pixel values do not change how much the NPU computes, so
        # --input has no meaning here; camera count does, since it is the host
        # loop count.
        return run_profile(args.prompt, args.reps, args.cameras)

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
        print("[smolvla] input        : synthetic (all-zero)")
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

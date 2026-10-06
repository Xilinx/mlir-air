# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build a model's fused prefill: kernels, every op's insts, and an optional
LM-head device. A model's spec module provides

  CFG            device.Config
  gemm_shapes()  (k, n_pad, act, wq) of every GEMM
  ATTN_POINTS    attention calibration points, name -> (heads, nkv, q0, k0)
  lm_build()     optional: the LM-head device module (built as op "lm")

Every op compiles the same device, so all builds must produce the same device
configuration; the CDOs are compared and the build fails if they differ.
"""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import device as D

HERE = Path(__file__).resolve().parent
EX = HERE.parents[2]  # programming_examples

ATTN_FNS = (
    "zero_fill_g_bf16 zero_fill_gp_bf16 zero_fill_sp_bf16 neg_inf_fill_up_bf16 matmul_a_b_bf16 "
    "matmul_g_b_bf16 fused_softmax mul_r_gp accum_sp_r_s vector_copy_32elems div_gp_sp "
    "apply_causal_mask apply_window_mask"
).split()


def attn_points(heads):
    """Calibration points (heads, nkv, q0, k0): a base, one per parameter,
    and a check point that moves all three."""
    return {
        "base": (heads, 1, 0, 0),
        "nkv": (heads, 2, 0, 0),
        "q0": (heads, 1, 1, 0),
        "k0": (heads, 1, 0, 1),
        "check": (heads, 5, 3, 2),
    }


def load_spec(ref):
    """A spec module by name, or "dense:<model>" for a model in models.py."""
    if ref.startswith("dense:"):
        from .models import MODELS, spec

        return spec(MODELS[ref[6:]])
    return importlib.import_module(ref)


def op_name(op, shape):
    return "_".join([op] + [str(v) for v in shape])


def ops(spec):
    """(op, shape) of every op of the spec's device."""
    out = [("g", s) for s in spec.gemm_shapes()]
    out += [(a, p) for a in D.attn_ops(spec.CFG) for p in spec.ATTN_POINTS.values()]
    return out


def module(spec, op, shape):
    if op == "lm":
        return spec.lm_build()
    return D.build(spec.CFG, op, *shape)


def _peano():
    return os.path.join(os.environ["PEANO_INSTALL_DIR"], "bin", "clang++")


def _flags():
    inc = os.environ.get("MLIR_AIE_INSTALL_DIR") or str(
        Path(shutil.which("aie-opt")).parents[1]
    )
    return [
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-DNDEBUG",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-deprecated-declarations",
        f"-I{inc}/include",
        "-D__AIE_API_AIE_ADF_HPP__",
    ]


def build_kernels(out, cfg):
    cc, f = _peano(), _flags()
    kern = HERE / "kernels"
    mm = EX / "matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc"
    attn = EX / "flash_attention/kernel_fusion_based/attn_npu2.cc"
    m, n, k = D.TM, D.TN, D.TK
    jobs = [
        [
            cc,
            *f,
            "-DBIT_WIDTH=8",
            "-c",
            mm,
            "-o",
            out / D.MM_OBJ,
            f"-DDIM_M={m}",
            f"-DDIM_N={n}",
            f"-DDIM_K={k}",
            f"-DDIM_M_DIV_4={m // 4}",
            f"-DDIM_N_DIV_4={n // 4}",
            f"-DDIM_M_DIV_8={m // 8}",
            f"-DDIM_N_DIV_8={n // 8}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            f"-DSYM_SUFFIX={D.MM_SUF}",
        ],
        [
            cc,
            *f,
            "-c",
            kern / "dq4.cc",
            "-o",
            out / D.DQ_OBJ,
            f"-DDQ_TN={n}",
            f"-DDQ_TK={k}",
        ],
        [cc, *f, "-c", kern / "gemv_q4.cc", "-o", out / "gemv_q4.o"],
    ]
    # one attention object per head_dim; groups sharing one must agree on it
    seen = {}
    for g in cfg.attn.values():
        key = (g.lkp, g.dvt)
        if seen.setdefault(g.dh, key) != key:
            raise ValueError(f"attention groups with head_dim {g.dh} differ in tiling")
    for dh, (lkp, dvt) in seen.items():
        ren = [f"-D{fn}={fn}_d{dh}" for fn in ATTN_FNS]
        jobs.append(
            [
                cc,
                *f,
                "-DBIT_WIDTH=8",
                f"-I{attn.parent}",
                "-c",
                attn,
                "-o",
                out / f"attn_d{dh}.o",
                f"-Dlqp={lkp}",
                f"-Dlkp={lkp}",
                f"-Ddk={dh}",
                f"-Ddk_full={dh}",
                f"-Ddv={dvt}",
                f"-Ddv_full={dh}",
                "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
                "-DROUND_CONV_EVEN",
                *ren,
            ]
        )
    jobs.append(
        [
            "gcc",
            "-O3",
            "-march=native",
            "-ffp-contract=off",
            "-fopenmp",
            "-shared",
            "-fPIC",
            kern / "hostops.c",
            "-o",
            out / "libhostops.so",
            "-lm",
        ]
    )
    for j in jobs:
        subprocess.run([str(x) for x in j], check=True)


def _compile(args):
    """One op in its own work dir (aircc writes air_project/ to the cwd)."""
    spec_name, out, op, shape = args
    from air.backend.xrt import XRTBackend

    spec = load_spec(spec_name)
    name = op_name(op, shape)
    work = Path(out) / "work" / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for o in Path(out).glob("*.o"):
        os.symlink(o, work / o.name)
    os.chdir(work)
    mod = module(spec, op, shape).build(target="npu2")
    be = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        kernel_name="MLIR_AIE",
        instance_name=name,
    )
    be.compile(mod, output_binary_name=name, insts=f"{name}.insts.bin")
    be.unload()
    for ext in (".xclbin", ".insts.bin"):
        shutil.move(work / f"{name}{ext}", Path(out) / f"{name}{ext}")
    cdo = b"".join(
        p.read_bytes() for p in sorted((work / "air_project/cdo_seg").glob("*.bin"))
    )
    shutil.rmtree(work)
    return name, hashlib.sha256(cdo).hexdigest()


def main(spec_name, argv=None):
    """python3 build.py BUILD_DIR [-j N], for the spec spec_name (load_spec)."""
    spec = load_spec(spec_name)
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("-j", type=int, default=4)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    build_kernels(out, spec.CFG)
    todo = [(spec_name, str(out), op, s) for op, s in ops(spec)]
    if hasattr(spec, "lm_build"):
        todo.append((spec_name, str(out), "lm", ()))
    with ProcessPoolExecutor(a.j) as ex:
        digests = dict(ex.map(_compile, todo))
    fused = {d for n, d in digests.items() if not n.startswith("lm")}
    if len(fused) != 1:
        raise SystemExit(
            f"fused prefill builds disagree on the device configuration: {digests}"
        )
    manifest = dict(
        gemm=[op_name("g", s) for s in spec.gemm_shapes()],
        attn={
            a: {k: op_name(a, p) for k, p in spec.ATTN_POINTS.items()}
            for a in D.attn_ops(spec.CFG)
        },
        attn_points=spec.ATTN_POINTS,
        lm="lm" if hasattr(spec, "lm_build") else None,
        device=fused.pop(),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"built {len(todo)} ops into {out}")

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build a model's fused prefill: kernels, the device with one runtime
sequence per op, and an optional LM-head device. A model's spec module provides

  CFG            device.Config
  gemm_shapes()  (k, n_pad, act, wq) of every GEMM
  ATTN_POINTS    attention points, name -> (heads, nkv, q0, k0); only the
                 heads are used: attention takes (nkv, q0, k0) at run time
  lm_build()     optional: the LM-head device module (built as op "lm")

Each op is one func whose segment @seg holds only its own herds, all in one
module, so aircc builds the device once with a runtime sequence per func. The
sequences come out as a C++ TXN builder (fused_prefill.txn.h), called with an
attention op's (nkv, q0, k0) at dispatch.
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


def attn_points(heads, even=False):
    """Calibration points (heads, nkv, q0, k0): a base, one per parameter,
    and a check point that moves all three; nkv even if `even`."""
    if even:
        return {
            "base": (heads, 2, 0, 0),
            "nkv": (heads, 4, 0, 0),
            "q0": (heads, 2, 1, 0),
            "k0": (heads, 2, 0, 1),
            "check": (heads, 6, 3, 2),
        }
    return {
        "base": (heads, 1, 0, 0),
        "nkv": (heads, 2, 0, 0),
        "q0": (heads, 1, 1, 0),
        "k0": (heads, 1, 0, 1),
        "check": (heads, 5, 3, 2),
    }


def load_spec(ref):
    """A spec module by name, or "dense:<model>[:<max_len>]" for a model in
    models.py, with its max_len replaced when one is given."""
    if ref.startswith("dense:"):
        from .models import MODELS, spec

        name, _, max_len = ref[6:].partition(":")
        desc = MODELS[name]
        if max_len:
            desc = desc._replace(max_len=int(max_len))
        return spec(desc)
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


def seq_name(op, shape):
    """Name of the runtime sequence for an op. GEMMs get one per shape. An
    attention op's lengths are run-time arguments, so its points share one
    sequence, keyed by the first shape entry only."""
    return op_name(op, shape if op == "g" else shape[:1])


def _link(texts):
    """Merge modules into one, keeping each shared declaration once."""
    from air.ir import Context, Location, Module

    with Context(), Location.unknown():
        mods = [Module.parse(t) for t in texts]
        base, seen = mods[0], {}

        def key(op):
            if "sym_name" in op.attributes:
                return str(op.attributes["sym_name"])

        for op in base.body.operations:
            seen[key(op)] = op.get_asm(print_generic_op_form=True)
        for m in mods[1:]:
            for op in list(m.body.operations):
                k, text = key(op), op.get_asm(print_generic_op_form=True)
                if k in seen:
                    if seen[k] != text:
                        raise ValueError(f"ops declare {k} differently")
                    continue
                seen[k] = text
                base.body.append(op)
        return str(base)


def fused_module(spec):
    """Build one module holding a func for each op in the spec."""
    texts, done = [], set()
    for op, shape in ops(spec):
        name = seq_name(op, shape)
        if name in done:
            continue
        done.add(name)
        lch = D.build(spec.CFG, op, *shape, name=name, own=True)
        texts.append(str(lch.build(target="npu2")))
    return _link(texts)


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
            kern / "gemm_bfp16.cc",
            "-o",
            out / D.GEMM_OBJ,
            f"-DDIM_M={m}",
            f"-DDIM_N={n}",
            f"-DDIM_K={k}",
        ],
        [cc, *f, "-c", kern / "gemv_q4.cc", "-o", out / "gemv_q4.o"],
    ]
    # one attention object per head_dim; groups sharing one must agree on it
    seen = {}
    for g in cfg.attn.values():
        if g.kern == "bfp16":
            continue
        key = (g.lkp, g.dvt)
        if seen.setdefault(g.dh, key) != key:
            raise ValueError(f"attention groups with head_dim {g.dh} differ in tiling")
    for dh in {g.dh for g in cfg.attn.values() if g.kern == "bfp16"}:
        jobs.append(
            [
                cc,
                *f,
                "-c",
                kern / "attn_bfp16.cc",
                "-o",
                out / f"attn_bfp16_d{dh}.o",
                f"-DDH={dh}",
                *[f"-D{fn}={fn}_d{dh}" for fn in ("attn_init", "attn_blk", "attn_fin")],
            ]
        )
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
    """The fused device ("fused_prefill") or the LM head ("lm"), in its own
    work dir (aircc writes air_project/ to the cwd)."""
    spec_name, out, name = args
    from air.backend.xrt import XRTBackend
    from air.ir import Context, Module

    spec = load_spec(spec_name)
    work = Path(out) / "work" / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for o in Path(out).glob("*.o"):
        os.symlink(o, work / o.name)
    os.chdir(work)
    fused = name == "fused_prefill"
    be = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        kernel_name="MLIR_AIE",
        instance_name=name,
        emit_txn_cpp=fused,
    )
    if fused:
        with Context():
            mod = Module.parse(fused_module(spec))
            art = be.compile(mod, output_binary_name=name)
        shutil.move(work / art.txn_header, Path(out) / f"{name}.txn.h")
    else:
        mod = module(spec, "lm", ()).build(target="npu2")
        be.compile(mod, output_binary_name=name, insts=f"{name}.insts.bin")
        shutil.move(work / f"{name}.insts.bin", Path(out) / f"{name}.insts.bin")
    be.unload()
    shutil.move(work / f"{name}.xclbin", Path(out) / f"{name}.xclbin")
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
    todo = [(spec_name, str(out), "fused_prefill")]
    if hasattr(spec, "lm_build"):
        todo.append((spec_name, str(out), "lm"))
    with ProcessPoolExecutor(a.j) as ex:
        digests = dict(ex.map(_compile, todo))
    heads = spec.ATTN_POINTS["base"][0]
    manifest = dict(
        gemm=[op_name("g", s) for s in spec.gemm_shapes()],
        attn={a: seq_name(a, (heads,)) for a in D.attn_ops(spec.CFG)},
        heads=heads,
        lm="lm" if hasattr(spec, "lm_build") else None,
        device=digests["fused_prefill"],
        max_len=getattr(spec, "max_len", None),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    nops = len(manifest["gemm"]) + len(manifest["attn"]) + bool(manifest["lm"])
    print(f"built {nops} ops into {out}")

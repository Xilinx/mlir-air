# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the fused prefill: kernels, every op's insts, the LM-head GEMV.

  python3 build.py BUILD_DIR [-j N]

Every op compiles the same device, so all builds must produce the same device
configuration; the CDOs are compared and the build fails if they differ.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EX = HERE.parents[2]  # programming_examples
sys.path[:0] = [str(HERE), str(HERE.parent)]

import device as D  # noqa: E402
import gemma4_e2b_q4nx_weights as gw  # noqa: E402

ATTN_FNS = (
    "zero_fill_g_bf16 zero_fill_gp_bf16 zero_fill_sp_bf16 neg_inf_fill_up_bf16 matmul_a_b_bf16 "
    "matmul_g_b_bf16 fused_softmax mul_r_gp accum_sp_r_s vector_copy_32elems div_gp_sp "
    "apply_causal_mask apply_window_mask"
).split()
# attention calibration points (heads, nkv, q0, k0): a base, one per
# parameter, and a check point that moves all three
HEADS = gw.N_Q_HEADS
ATTN_POINTS = {
    "base": (HEADS, 1, 0, 0),
    "nkv": (HEADS, 2, 0, 0),
    "q0": (HEADS, 1, 1, 0),
    "k0": (HEADS, 1, 0, 1),
    "check": (HEADS, 5, 3, 2),
}


def gemm_shapes():
    """(k, n_pad, act, wq) of every Gemma4-E2B prefill GEMM."""
    n = lambda v: -(-v // D.NR) * D.NR  # noqa: E731
    d, s = gw.D, set()
    for dh in (gw.DH_SLIDING, gw.DH_GLOBAL):
        dq = gw.N_Q_HEADS * dh
        s |= {(d, n(dq), 0, 1), (d, n(dh * gw.N_KV_HEADS), 0, 1), (dq, d, 0, 1)}
    for inter in (gw.INTER, 2 * gw.INTER):
        s |= {(d, inter, 1, 1), (d, inter, 0, 1), (inter, d, 0, 1)}
    s |= {
        (d, n(gw.PLI_D), 1, 0),
        (gw.PLI_D, d, 0, 0),
        (d, n(gw.NUM_LAYERS * gw.PLI_D), 0, 0),
    }
    return sorted(s)


def op_name(op, shape):
    return "_".join([op] + [str(v) for v in shape])


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


def build_kernels(out):
    cc, f = _peano(), _flags()
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
            HERE / "kernels/dq4.cc",
            "-o",
            out / D.DQ_OBJ,
            f"-DDQ_TN={n}",
            f"-DDQ_TK={k}",
        ],
        [cc, *f, "-c", HERE / "kernels/gemv_q4.cc", "-o", out / "gemv_q4.o"],
    ]
    for a, (_, dh, lkp, dvt) in D.ATTN.items():
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
            "-shared",
            "-fPIC",
            HERE / "kernels/hostops.c",
            "-o",
            out / "libhostops.so",
            "-lm",
        ]
    )
    for j in jobs:
        subprocess.run([str(x) for x in j], check=True)


def _compile(args):
    """One op in its own work dir (aircc writes air_project/ to the cwd)."""
    out, op, shape = args
    from air.backend.xrt import XRTBackend

    name = op_name(op, shape)
    work = Path(out) / "work" / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for o in Path(out).glob("*.o"):
        os.symlink(o, work / o.name)
    os.chdir(work)
    if op == "lm":
        import lm_gemv

        mod = lm_gemv.build().build(target="npu2")
    else:
        mod = D.build(op, *shape).build(target="npu2")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("-j", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    build_kernels(out)
    ops = [(str(out), "g", s) for s in gemm_shapes()]
    ops += [(str(out), at, p) for at in D.ATTN for p in ATTN_POINTS.values()]
    ops.append((str(out), "lm", ()))
    with ProcessPoolExecutor(a.j) as ex:
        digests = dict(ex.map(_compile, ops))
    fused = {d for n, d in digests.items() if not n.startswith("lm")}
    if len(fused) != 1:
        raise SystemExit(
            f"fused prefill builds disagree on the device configuration: {digests}"
        )
    manifest = dict(
        gemm=[op_name("g", s) for s in gemm_shapes()],
        attn={at: {k: op_name(at, p) for k, p in ATTN_POINTS.items()} for at in D.ATTN},
        attn_points=ATTN_POINTS,
        lm="lm",
        device=fused.pop(),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"built {len(ops)} ops into {out}")


if __name__ == "__main__":
    main()

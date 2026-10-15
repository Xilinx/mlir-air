# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""B8 context prepass using the standard BF16/BFP matrix primitive.

Projection layout follows the BF16-in/FP32-out GEMM example, specialized to one
8-row matrix tile. The four-stage ELF follows the llama32_1b multi-launch flow.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
from build_identity import build_identity, check_existing

from air.ir import MemRefType, F32Type, BF16Type, IntegerAttr, StringAttr
from air.dialects.air import (
    module_builder,
    launch,
    segment,
    herd,
    dma_memcpy_nd,
    MemorySpace,
    T,
)
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_ as range_, yield_
from air.dialects.affine import apply as affine_apply
from air.ir import AffineMapAttr

HERE = Path(__file__).resolve().parent
BACKEND = dict(
    omit_while_true_loop=False,
    output_format="elf",
    instance_name="context_prepass",
    omit_pingpong="all",
    use_lock_race_condition_fix_v2=True,
    stack_size=12288,
    target_device="npu2",
)


@module_builder
def build_module(k_tile=256, n_tile=64, cores_count=8):
    f32 = F32Type.get()
    bf16 = BF16Type.get()
    l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)

    def mem(shape, space=None, dtype=None):
        return MemRefType.get(shape, dtype or f32, memory_space=space)

    def idx(expression, *values):
        names = ",".join(f"d{i}" for i in range(len(values)))
        return affine_apply(
            AffineMapAttr.parse(f"affine_map<({names})->({expression})>").value,
            list(values),
        )

    def private(name, shapes, dtypes=None):
        dtypes = dtypes or [f32] * len(shapes)
        op = FuncOp(
            name,
            ([mem(s, l1, t) for s, t in zip(shapes, dtypes)], []),
            visibility="private",
        )
        op.attributes["link_with"] = StringAttr.get("context_prepass_bfp.o")
        return op

    zero = private("bfp_prepass_zero", [[8, n_tile]])
    project = private(
        "bfp_prepass_project",
        [[8, k_tile], [k_tile, n_tile], [8, n_tile]],
        [bf16, bf16, f32],
    )
    flush = private("bfp_prepass_flush", [[8, n_tile], [8, n_tile]])
    norm = private("bfp_prepass_norm", [[2560], [2560], [2560]], [f32, f32, bf16])
    key = private("prepass_key_row", [[10240], [768]])

    def alloc(shape, dtype=None):
        return AllocOp(mem(shape, l1, dtype), [], [])

    def projection(name, x, w, y, k, n):
        @launch(sizes=[1, 1], operands=[x, w, y])
        def launched(_a, _b, _c, _d, X, W, Y):
            @segment(name=name, operands=[X, W, Y])
            def segmented(X, W, Y):
                @herd(
                    name=name + "_cores",
                    sizes=[8, cores_count // 8],
                    operands=[X, W, Y],
                    link_with="context_prepass_bfp.o",
                )
                def cores(tx, ty, sx, sy, X, W, Y):
                    a, b, c, d = (
                        alloc([8, k_tile], bf16),
                        alloc([k_tile, n_tile], bf16),
                        alloc([8, n_tile]),
                        alloc([8, n_tile]),
                    )
                    core = idx("d0+8*d1", tx, ty)
                    for wave in range_(n // (cores_count * n_tile)):
                        column = idx(f"(d0*{cores_count}+d1)*{n_tile}", wave, core)
                        CallOp(zero, [c])
                        for chunk in range_(k // k_tile):
                            off = idx(f"d0*{k_tile}", chunk)
                            # W layout is [wave, core, Kchunk, K, N].
                            woff = idx(
                                f"((d0*{cores_count}+d1)*{k//k_tile}+d2)*{k_tile*n_tile}",
                                wave,
                                core,
                                chunk,
                            )
                            dma_memcpy_nd(
                                a,
                                X,
                                src_offsets=[0, off],
                                src_sizes=[8, k_tile],
                                src_strides=[k, 1],
                            )
                            dma_memcpy_nd(
                                b,
                                W,
                                src_offsets=[woff],
                                src_sizes=[k_tile * n_tile],
                                src_strides=[1],
                            )
                            CallOp(project, [a, b, c])
                            yield_([])
                        CallOp(flush, [c, d])
                        dma_memcpy_nd(
                            Y,
                            d,
                            dst_offsets=[0, column],
                            dst_sizes=[8, n_tile],
                            dst_strides=[n, 1],
                        )
                        yield_([])
                    for buf in (a, b, c, d):
                        DeallocOp(buf)

    @FuncOp.from_py_func(
        mem([8, 12800], dtype=bf16),
        mem([12800 * 2560], dtype=bf16),
        mem([3200]),
        mem([2560 * 10240], dtype=bf16),
        mem([8, 768]),
        mem([8, 2560]),
        mem([8, 2560], dtype=bf16),
        mem([8, 10240]),
        mem([8, 5120]),
    )
    def context_prepass(X, FC, NORM, KV, LUT, RAW, F, RAWKV, KOUT):
        projection("feature_projection", X, FC, RAW, 12800, 2560)

        @launch(sizes=[1, 1], operands=[RAW, NORM, F])
        def norm_launch(_a, _b, _c, _d, X, W, Y):
            @segment(name="feature_norm", operands=[X, W, Y])
            def norm_segment(X, W, Y):
                @herd(
                    name="feature_norm_cores",
                    sizes=[8, 1],
                    operands=[X, W, Y],
                    link_with="context_prepass_bfp.o",
                )
                def norm_herd(tx, ty, sx, sy, X, W, Y):
                    a, b, c = alloc([2560]), alloc([2560]), alloc([2560], bf16)
                    dma_memcpy_nd(
                        a,
                        X,
                        src_offsets=[tx, 0],
                        src_sizes=[1, 2560],
                        src_strides=[2560, 1],
                    )
                    dma_memcpy_nd(
                        b, W, src_offsets=[0], src_sizes=[2560], src_strides=[1]
                    )
                    CallOp(norm, [a, b, c])
                    dma_memcpy_nd(
                        Y,
                        c,
                        dst_offsets=[tx, 0],
                        dst_sizes=[1, 2560],
                        dst_strides=[2560, 1],
                    )
                    for buf in (a, b, c):
                        DeallocOp(buf)

        projection("context_projection", F, KV, RAWKV, 2560, 10240)

        @launch(sizes=[1, 1], operands=[RAWKV, NORM, LUT, KOUT])
        def key_launch(_a, _b, _c, _d, X, W, L, Y):
            @segment(name="key_norm_rope", operands=[X, W, L, Y])
            def key_segment(X, W, L, Y):
                @herd(
                    name="key_norm_rope_cores",
                    sizes=[8, 1],
                    operands=[X, W, L, Y],
                    link_with="context_prepass_bfp.o",
                )
                def key_herd(tx, ty, sx, sy, X, W, L, Y):
                    # Complete row fits in L1; one transfer per input avoids
                    # interleaving five K blocks with five norm blocks on one
                    # physical packet stream. The microkernel compacts K in-place.
                    a, b = alloc([10240]), alloc([768])
                    dma_memcpy_nd(
                        a,
                        X,
                        src_offsets=[tx, 0],
                        src_sizes=[1, 10240],
                        src_strides=[10240, 1],
                    )
                    dma_memcpy_nd(
                        b,
                        L,
                        src_offsets=[tx, 0],
                        src_sizes=[1, 768],
                        src_strides=[768, 1],
                    )
                    CallOp(key, [a, b])
                    dma_memcpy_nd(
                        Y,
                        a,
                        dst_offsets=[tx, 0],
                        dst_sizes=[1, 5120],
                        dst_strides=[5120, 1],
                        src_offsets=[0],
                        src_sizes=[5120],
                        src_strides=[1],
                    )
                    DeallocOp(a)
                    DeallocOp(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-existing", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--projection-k", type=int, choices=(128, 256, 512), default=256)
    ap.add_argument("--projection-n", type=int, choices=(32, 64, 128), default=64)
    ap.add_argument("--cores", type=int, choices=(8, 16), default=8)
    args = ap.parse_args()
    assert 2560 % (args.projection_n * args.cores) == 0
    out = args.output.resolve()
    identity = build_identity("prepass")
    if out.exists() and args.check_existing:
        check_existing(out, identity)
        old = json.loads((out / "manifest.json").read_text())
        if (old["projection_k"], old["projection_n"], old["projection_cores"]) != (
            args.projection_k,
            args.projection_n,
            args.cores,
        ):
            raise ValueError("Prepass configuration differs; choose a fresh BUILD_DIR")
        return
    out.mkdir(parents=True, exist_ok=False)
    source = HERE / "kernels/context_prepass_bfp.cc"
    command = [
        os.environ["PEANO_INSTALL_DIR"] + "/bin/clang++",
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-attributes",
        "-Wno-parentheses",
        "-Wno-macro-redefined",
        "-D__AIE_API_AIE_ADF_HPP__",
        "-DNDEBUG",
        f"-DBFP_PREPASS_K={args.projection_k}",
        f"-DBFP_PREPASS_N={args.projection_n}",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        "-I" + os.environ["MLIR_AIE_INSTALL_DIR"] + "/include",
        "-c",
        str(source),
        "-o",
        str(out / "context_prepass_bfp.o"),
    ]
    record = dict(
        build_identity=identity,
        status="building",
        backend=BACKEND,
        command=command,
        compiler_sha256=hashlib.sha256(
            Path(shutil.which("air-opt")).read_bytes()
        ).hexdigest(),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), source, HERE / "kernels/context_prepass.cc")
        },
        precision="BF16 weights/input, BFP16 8x8x8 matrix products, FP32 accumulators; BF16 normalized features",
        interface="bfp-prepass-v1",
        weight_source="official_checkpoint_bf16",
        rows=8,
        projection_k=args.projection_k,
        projection_n=args.projection_n,
        projection_cores=args.cores,
        host_dispatches_per_chunk=1,
        device_stages=4,
    )

    def save():
        (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        subprocess.run(command, check=True)
        module = build_module(args.projection_k, args.projection_n, args.cores)
        assert module.operation.verify()
        (out / "input.mlir").write_text(str(module))
        from air.backend.xrt import XRTBackend

        os.chdir(out)
        XRTBackend(**BACKEND).compile(module, output_binary_name="prepass")
        record.update(
            status="compiled",
            artifacts={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.glob("prepass.*")
            },
        )
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

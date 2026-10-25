# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compile the selected target or drafter and save its ABI.

Use the standard AIR environment. Each build keeps its own
compiler intermediates; sources and artifacts are hashed in manifest.json.
"""

from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import ModuleType

from air.backend.compile_cache import run_compiler
from common.build_identity import build_identity, check_existing

COMMON = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main(
    target_config: ModuleType,
    selected_config,
    kernel_defines,
):
    """Build with a model's `target_config` module and selected configurations.

    `kernel_defines(cfg)` gives the model's kernel macros for a configuration:
    the model the fused-decode headers select, and any layout or arithmetic
    settings beyond the kernels' defaults.
    """
    HERE, EXAMPLES, FUSED, REPO = (
        target_config.HERE,
        target_config.EXAMPLES,
        target_config.FUSED,
        target_config.REPO,
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-existing", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--role", choices=("target", "drafter"), default="target")
    ap.add_argument("--capacity", type=int, default=4096)
    ap.add_argument("--block-size", type=int, choices=(8, 16), default=8)
    ap.add_argument(
        "--layers", type=int, help="Reduced layer count for focused validation"
    )
    ap.add_argument(
        "--reuse-objects",
        type=Path,
        help="Reuse matching target objects for the drafter",
    )
    args = ap.parse_args()
    cfg = selected_config(args.role, args.block_size)
    cfg = replace(cfg, capacity=args.capacity, layers=args.layers or cfg.layers)
    dest = args.output.resolve()
    identity = build_identity("decoder", HERE)
    if dest.exists() and args.check_existing:
        check_existing(dest, identity)
        if json.loads((dest / "manifest.json").read_text())["config"] != asdict(cfg):
            raise ValueError("Build configuration differs; choose a fresh BUILD_DIR")
        return
    dest.mkdir(parents=True, exist_ok=False)
    fd = target_config.load_geometry(cfg)
    inputs = (
        list(HERE.glob("*.py"))
        + list(COMMON.glob("*.py"))
        + list((COMMON / "kernels").glob("*"))
    )
    inputs += (
        list(FUSED.glob("*.py"))
        + list((FUSED / "kernels").glob("*"))
        + list((FUSED / "models").glob("*"))
    )
    manifest = {
        "build_identity": identity,
        "config": asdict(cfg),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip(),
        "source_hashes": {
            str(p.relative_to(EXAMPLES)): sha(p) for p in sorted(inputs) if p.is_file()
        },
        "toolchain": {
            k: os.environ[k]
            for k in (
                "MLIR_AIR_INSTALL_DIR",
                "MLIR_AIE_INSTALL_DIR",
                "PEANO_INSTALL_DIR",
                "LLVM_INSTALL_DIR",
            )
        },
        "status": "building",
        "commands": [],
        "artifacts": {},
    }
    air_opt = Path(shutil.which("air-opt")).resolve()
    manifest["air_build_manifest_sha256"] = sha(air_opt)
    compiler_source = REPO / "mlir/lib/Conversion/AIRToAIEPass.cpp"
    manifest["compiler"] = {
        "air_opt": str(air_opt),
        "air_opt_sha256": sha(air_opt),
        "air_to_aie_source_sha256": sha(compiler_source),
        "air_to_aie_schedule_source_sha256": sha(
            compiler_source.with_name("AIRToAIESchedulingUtils.cpp")
        ),
        "air_schedule_source_sha256": sha(
            compiler_source.parent.parent / "Transform/AIRDependencyScheduleOpt.cpp"
        ),
        "dependency_util_source_sha256": sha(
            compiler_source.parent.parent / "Util/Dependency.cpp"
        ),
        "air_rt_to_npu_source_sha256": sha(
            compiler_source.with_name("AIRRtToNpuPass.cpp")
        ),
    }
    import tarfile

    with tarfile.open(dest / "source_snapshot.tar.gz", "w:gz") as archive:
        for p in sorted(inputs):
            if p.is_file():
                archive.add(p, arcname=str(p.relative_to(EXAMPLES)))
    write_json(dest / "manifest.json", manifest)
    clang = Path(os.environ["PEANO_INSTALL_DIR"]) / "bin/clang++"
    flags = [
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-deprecated-declarations",
        "-DNDEBUG",
        *kernel_defines(cfg),
        "-D__AIE_API_AIE_ADF_HPP__",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        "-DGLU_SLICE_EXPECTED=" + str(fd.GLU_SLICE),
        "-I" + str(Path(os.environ["MLIR_AIE_INSTALL_DIR"]) / "include"),
        "-I" + str(FUSED / "kernels"),
        "-I" + str(FUSED / "models"),
        "-I" + str(EXAMPLES),
        "-I" + str(COMMON / "kernels"),
    ]
    flags += [
        "-DATTN_NORM_VECTOR",
        f"-DPROJ_MM_BATCH={cfg.batch}",
        "-DDECODE_ROUND_NEAREST",
        "-DATTN_QUERY_TILE=4",
        "-DATTN_NATIVE_KEY=" + str(int(cfg.attention_native_key)),
        "-DATTN_PV_REUSE=" + str(int(cfg.attention_pv_reuse)),
    ]
    objdir = dest / "objects"
    objdir.mkdir()
    reuse = None
    if args.reuse_objects:
        reuse = json.loads((args.reuse_objects / "manifest.json").read_text())
        if reuse["air_build_manifest_sha256"] != manifest["air_build_manifest_sha256"]:
            raise ValueError("Object reuse toolchain mismatch")
        for name, digest in manifest["source_hashes"].items():
            if (
                Path(name).suffix in (".cc", ".h", ".hpp")
                and reuse["source_hashes"].get(name) != digest
            ):
                raise ValueError("Object reuse source mismatch: " + name)
        manifest["object_reuse_manifest"] = {
            "path": str(args.reuse_objects / "manifest.json"),
            "sha256": sha(args.reuse_objects / "manifest.json"),
        }
    mapping = {
        "proj_qmm": "proj",
        "rms_residual": "rms",
        "glu": "glu",
    }
    try:
        # The ABI of Peano's lifetime intrinsics requires llvm-link <23.
        subprocess.run(["make", "-C", str(FUSED), "preflight-llvm-link"], check=True)
        for name in (
            "proj_qmm",
            "rms_residual",
            "glu",
            "rope",
            "attn_fused",
            "norm_cache",
        ):
            src = (
                COMMON / "kernels" / (mapping.get(name, name) + "_batch.cc")
                if name != "rope"
                else FUSED / "kernels" / (name + ".cc")
            )
            if name == "norm_cache":
                src = COMMON / "kernels/norm_cache.cc"
            if name == "rms_residual" and cfg.batch == 16:
                src = COMMON / "kernels/rms_b16.cc"
            attn = name.startswith("attn_")
            out = objdir / (name + (".ll" if attn else ".o"))
            cmd = [str(clang), *flags, "-O1" if attn else "-O2"]
            cmd += ["-DDECODE_INLINE_ATTN", "-S", "-emit-llvm"] if attn else ["-c"]
            cmd += [str(src), "-o", str(out)]
            manifest["commands"].append(cmd)
            print("compiling", name, flush=True)
            old = (
                [c for c in reuse["commands"] if Path(c[-1]).name == out.name]
                if reuse
                else []
            )
            if len(old) == 1 and old[0][:-1] == cmd[:-1]:
                source = args.reuse_objects / "objects" / out.name
                shutil.copy2(source, out)
                manifest.setdefault("reused_object_hashes", {})[out.name] = sha(out)
            else:
                # Without a reusable object, e.g. when the roles select
                # different models in the kernel headers, compile it here.
                if reuse:
                    manifest.setdefault("recompiled_objects", []).append(out.name)
                run_compiler(cmd, check=True)
        from air.backend.xrt import XRTBackend

        length = cfg.build_length
        work = dest / f"compile_L{length}"
        work.mkdir()
        for p in objdir.iterdir():
            shutil.copy2(p, work / p.name)
        module, interface = target_config.build_module(cfg, length)
        assert module.operation.verify()
        (work / "input.mlir").write_text(str(module))
        old = Path.cwd()
        try:
            os.chdir(work)
            backend = XRTBackend(
                omit_while_true_loop=False,
                output_format="xclbin",
                kernel_name="MLIR_AIE",
                stack_size=cfg.stack_bytes,
                omit_pingpong="" if cfg.l2_pingpong else "L2",
                use_lock_race_condition_fix_v2=cfg.batch == 8,
                use_lock_race_condition_fix_auto=True if cfg.batch == 16 else None,
                coalesce_shim_dma=bool(fd.COALESCE),
                emit_txn_cpp=cfg.dynamic_context,
                target_device="npu2",
            )
            print("compiling template", length, "capacity", cfg.capacity, flush=True)
            artifact = backend.compile(
                module, output_binary_name="decode", insts="decode.insts.bin"
            )
        finally:
            os.chdir(old)
        files = {
            "xclbin": work / "decode.xclbin",
            "txn.h": work / artifact.txn_header,
        }
        prefix = f"decode_dyn_L{cfg.capacity}"
        for suffix, source in files.items():
            out = dest / f"{prefix}.{suffix}"
            shutil.copy2(source, out)
            manifest["artifacts"][out.name] = sha(out)
        manifest["abi"] = interface
        if "physical_abi" in interface:
            manifest["physical_abi"] = interface["physical_abi"]
        write_json(dest / "manifest.json", manifest)
        manifest["status"] = "compiled"
        write_json(dest / "manifest.json", manifest)
        print("compiled", dest, flush=True)
    except BaseException as e:
        manifest["status"] = "failed"
        manifest["error"] = repr(e)
        write_json(dest / "manifest.json", manifest)
        raise

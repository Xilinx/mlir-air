# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the single-token (B1) Qwen3-4B megakernel in an isolated directory."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from paths import HERE, COMMON
from common.build_identity import build_identity, check_existing, digest
from target_config import FUSED


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    out = a.output.resolve()
    identity = build_identity("baseline", HERE)
    if out.exists():
        check_existing(out, identity)
        return
    out.mkdir(parents=True)
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("DECODE_", "PROJ_", "RMS_", "ATTN_", "Q4K_", "UNI_")):
            del env[name]
    env.update(
        AIR_TARGET_DEVICE="npu2",
        DECODE_MODEL="qwen3-4b",
        VOCAB_CHUNK_I2="30",
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        DECODE_GOLDEN_L="4096",
        W_DUAL_CHAN="1",
    )
    record = dict(
        status="building",
        build_identity=identity,
        capacity=4096,
        batch=1,
        source="upstream fused_decode.py / qwen3-4b",
        commands=[],
        artifacts={},
    )

    def save():
        (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    def run(cmd, **kwargs):
        record["commands"].append(cmd)
        save()
        subprocess.run(cmd, check=True, env=env, **kwargs)

    save()
    try:
        run(["make", "-C", str(FUSED), "preflight-llvm-link"])
        glu = subprocess.check_output(
            [os.sys.executable, str(FUSED / "fused_decode.py")],
            env=dict(env, FUSED_DECODE_PRINT_CONST="GLU_SLICE"),
            text=True,
        ).strip()
        flags = [
            os.environ["PEANO_INSTALL_DIR"] + "/bin/clang++",
            "-std=c++20",
            "--target=aie2p-none-unknown-elf",
            "-DNDEBUG",
            "-DMODEL_TYPE=QWEN3_4B",
            "-D__AIE_API_AIE_ADF_HPP__",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DGLU_SLICE_EXPECTED=" + glu,
            "-I" + os.environ["MLIR_AIE_INSTALL_DIR"] + "/include",
            "-I" + str(FUSED / "kernels"),
            "-I" + str(FUSED / "models"),
        ]
        objects = out / "objects"
        objects.mkdir()
        for name in ("proj_qmm", "rms_residual", "glu", "rope", "attn_qk", "attn_kv"):
            attn = name.startswith("attn_")
            dest = objects / (name + (".ll" if attn else ".o"))
            run(
                flags
                + (
                    ["-O1", "-DDECODE_INLINE_ATTN", "-S", "-emit-llvm"]
                    if attn
                    else ["-O2", "-c"]
                )
                + [str(FUSED / "kernels" / (name + ".cc")), "-o", str(dest)]
            )
        for length in (4096, 4095):
            work = out / f"compile_L{length}"
            work.mkdir()
            for file in objects.iterdir():
                shutil.copy2(file, work / file.name)
            env["DECODE_GOLDEN_L"] = str(length)
            run([os.sys.executable, str(FUSED / "fused_decode.py")], cwd=work)
            for suffix in ("xclbin", "insts.bin"):
                dest = out / f"decode_L{length}.{suffix}"
                shutil.copy2(work / f"decode.{suffix}", dest)
                record["artifacts"][dest.name] = digest(dest)
        record["status"] = "compiled"
    except BaseException as e:
        record.update(status="failed", error=repr(e))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

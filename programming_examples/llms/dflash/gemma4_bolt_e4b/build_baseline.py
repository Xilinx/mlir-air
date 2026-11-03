# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the single-token (B1) Gemma4-E4B decoder (fused_decode_ple's full
ELF, as llms/gemma4_e4b_q4nx builds it) for a 4096-position cache, and keep
it in its own directory.

  python3 build_baseline.py --output DIR
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from paths import EXAMPLES, HERE
from common.build_identity import build_identity, check_existing, digest

PLE = EXAMPLES / "fused_decode_ple"
CAPACITY = 4096
OUTPUTS = ("decode_scratchpad.elf", "decode_scratchpad.params.txt")
OUTPUTS += ("decode_scratchpad.args", "decode_scratchpad.maxl")


def identity():
    """The shared baseline identity (fused_decode sources, tools), extended
    with fused_decode_ple's build and the kernel makefile it includes."""
    ident = build_identity("baseline", HERE)
    extra = [PLE / "Makefile", PLE / "fused_decode_ple.py"]
    extra += sorted((PLE / "kernels").glob("*"))
    extra += sorted((EXAMPLES / "fused_decode").glob("*.mk"))
    ident["sources"].update(
        {str(p.relative_to(EXAMPLES)): digest(p) for p in extra if p.is_file()}
    )
    return ident


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    out = a.output.resolve()
    manifest = out / "manifest.json"
    ident = identity()
    if manifest.exists():
        check_existing(out, ident)
        return
    sys.path.insert(0, str(HERE.parents[1] / "gemma4_e4b_q4nx"))
    import gemma4_e4b_q4nx_weights as gw

    kv_src = ",".join(str(gw.kv_source_layer(i)) for i in range(gw.NUM_LAYERS))
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "make",
        "-C",
        str(PLE),
        "compile-decode-elf",
        "DECODE_MODEL=gemma4-e4b",
        "MODEL_TYPE=GEMMA4_E4B",
        "VOCAB_CHUNK_I2=5",
        f"LBUILD={CAPACITY}",
        f"UNI_DEC={gw.NUM_LAYERS}",
        f"KV_SRC={kv_src}",
        "PEANO_INSTALL_DIR=" + os.environ["PEANO_INSTALL_DIR"],
    ]
    # The build writes into fused_decode_ple's directory; one at a time.
    with open("/tmp/mlir-air-fused-decode-ple.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(cmd, check=True)
        for name in OUTPUTS:
            shutil.move(PLE / name, out / name)
        (PLE / "params.txt").unlink(missing_ok=True)
    manifest.write_text(
        json.dumps(
            dict(
                status="compiled",
                capacity=CAPACITY,
                batch=1,
                command=cmd,
                build_identity=ident,
                artifacts={n: digest(out / n) for n in OUTPUTS},
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

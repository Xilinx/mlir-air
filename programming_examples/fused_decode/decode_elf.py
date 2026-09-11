# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Full-ELF decode: ONE artifact for every context length.
#
# The xclbin path specializes each token by patching the instruction stream it
# hands the kernel as an argument. An ELF has no such argument -- its stream
# lives in .ctrltext and is uploaded when the hw_context is built -- so the
# context length reaches the device as mlir-aie scratchpad parameters instead:
# the KV append slot address and the attention mask threshold, both declared by
# AIR (see AIRRtToNpuPass.cpp) and listed in the build's params.txt.
#
# That removes the template pair, the L-slope calibration and the staircase
# windows: there is nothing per-L to build or calibrate.
#
# Every model in programming_examples/llms shares this, because they all share
# the fused_decode engine; only the artifact directory differs.
#
# Needs `make compile-decode-elf` and a pyxrt exposing run.get_ctrl_scratchpad_bo()
# (XRT >= the 2026-05-19 binding; xdna-driver 1.7 carries it).

import os
import re
from pathlib import Path

# The runtime sequence's own name, hardcoded in fused_decode.py's builder, so it
# is the same for every model. XRT resolves an ELF kernel as main:<name>.
KERNEL_NAME = "main:q4nx_decode"


def parse_params(path):
    """Read params.txt into (append_name, scale, addend, mask_name, arg_index).

    The names are NOT fixed across models. AIR names a BD-offset parameter after
    the sequence argument it is affine in and puts the coefficients in the
    suffix, so llama's REGION_W=256 gives `..._x256_m256` (L*256 - 256) while
    gemma's 512 gives `..._x512_m512`. Hardcoding either writes the wrong KV
    address for the other, silently. Classify by the kind column instead --
    `addr` is the BD offset, `core` the herd RTP -- and take the arithmetic from
    the suffix rather than assuming it.
    """
    lines = [ln.split() for ln in Path(path).read_text().split("\n") if ln.strip()]
    entries = [ln for ln in lines[1:] if len(ln) >= 4]
    addr = [ln[0] for ln in entries if ln[3] == "addr"]
    core = [ln[0] for ln in entries if ln[3] == "core"]
    if len(addr) != 1 or len(core) != 1:
        raise RuntimeError(
            f"{path}: expected exactly one 'addr' and one 'core' parameter, "
            f"got addr={addr} core={core}"
        )
    m = re.search(r"_argoff_(\d+)_x(-?\d+)_([mp])(\d+)$", addr[0])
    if not m:
        raise RuntimeError(f"{path}: cannot read affine coefficients from {addr[0]}")
    # AIR keys the name on the SEQUENCE ARGUMENT NUMBER, which is also the index
    # the host must set L at -- and it is not the same for every model (llama is
    # 5, qwen3-8b is 9). Take it from the name rather than assuming.
    arg_index = int(m.group(1))
    scale = int(m.group(2))
    addend = int(m.group(4)) * (-1 if m.group(3) == "m" else 1)
    return addr[0], scale, addend, core[0], arg_index


def elf_on():
    """The full ELF is the default decode path; DECODE_ELF=0 selects the xclbin.

    Env, so every entry point -- CLI, verify adapter, lit -- selects it the same
    way. The xclbin path stays in the tree because it is the only one that runs
    on an XRT without the scratchpad binding (pre 2026-05-19), and because it is
    the reference an ELF is checked against.
    """
    return os.environ.get("DECODE_ELF", "1") == "1"


class ElfDecode:
    """One ELF + one persistent run, with L written to the scratchpad per token."""

    def __init__(self, art_dir, dev, xrt, region_w):
        self.xrt = xrt
        self.region_w = region_w
        art_dir = Path(art_dir)
        self.elf_path = art_dir / "decode_scratchpad.elf"
        self.params_path = art_dir / "decode_scratchpad.params.txt"
        stamp = art_dir / "decode_scratchpad.maxl"
        missing = [
            p.name for p in (self.elf_path, self.params_path, stamp) if not p.exists()
        ]
        if missing:
            raise RuntimeError(
                f"DECODE_ELF needs {', '.join(missing)} in {art_dir}; run "
                "`make compile-decode-elf`. (DECODE_ELF=0 selects the xclbin path.)"
            )
        # ATTN_MAXL comes from the build, not a default: a driver guessing 2048
        # against an ELF built at another L mis-sizes the KV cache and the mask
        # threshold together, and both failures look like bad numerics.
        self.attn_maxl = int(stamp.read_text().split()[0])
        (
            self.append_param,
            self.append_scale,
            self.append_addend,
            self.mask_param,
            self.scalar_arg,
        ) = parse_params(self.params_path)
        # The scale IS the model's REGION_W. If they disagree, the driver and the
        # build were made from different geometries and every KV append would land
        # in the wrong place -- which reads as bad numerics, not as a mismatch.
        if self.append_scale != region_w:
            raise RuntimeError(
                f"{self.params_path.name} encodes scale {self.append_scale} but the "
                f"driver's REGION_W is {region_w}; the ELF and the driver disagree."
            )
        self.ctx = xrt.hw_context(dev, xrt.elf(str(self.elf_path)))
        self.kern = xrt.ext.kernel(self.ctx, KERNEL_NAME)
        self.run = None
        self.params = None

    def bind(self, bos):
        """Bind the buffer arguments once. The run is persistent because the
        scratchpad BO belongs to it -- a run per token would rebuild the
        parameter buffer with it."""
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad,
        )

        self.run = self.xrt.run(self.kern)
        for i, b in enumerate(bos):
            self.run.set_arg(i, b)
        self.params = ParameterScratchpad(self.run, str(self.params_path))

    def dispatch(self, L, np):
        """One token at context length L. Returns the XRT run state."""
        self.params.write(
            self.append_param, np.int32(L * self.append_scale + self.append_addend)
        )
        self.params.write(self.mask_param, np.int32(L))
        self.params.sync()
        # Still required even though the hardware acts on the scratchpad copies:
        # XRT patches every declared argument, and the sequence declares L.
        self.run.set_arg(self.scalar_arg, L)
        self.run.start()
        self.run.wait2()
        return self.run.state()

    def close(self):
        self.params = None
        self.run = None
        self.kern = None
        self.ctx = None

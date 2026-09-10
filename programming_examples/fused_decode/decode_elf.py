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
from pathlib import Path

# The runtime sequence's own name, hardcoded in fused_decode.py's builder, so it
# is the same for every model. XRT resolves an ELF kernel as main:<name>.
KERNEL_NAME = "main:q4nx_decode"

# AIR names a BD-offset parameter after the sequence argument it is affine in,
# with the coefficients in the suffix -- the host writes the WHOLE affine value.
# `_x256_m256` is (L * 256) - 256, i.e. this token's slot at (L-1)*REGION_W.
APPEND_PARAM = "__air_param_argoff_5_x256_m256"
MASK_PARAM = "__air_param_attn_blk_0"


def elf_on():
    """Opt into (or out of) the full-ELF decode. Env, so every entry point --
    CLI, verify adapter, lit -- selects it the same way."""
    return os.environ.get("DECODE_ELF", "0") == "1"


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
        self.params.write(APPEND_PARAM, np.int32((L - 1) * self.region_w))
        self.params.write(MASK_PARAM, np.int32(L))
        self.params.sync()
        # Still required even though the hardware acts on the scratchpad copies:
        # XRT patches every declared argument, and the sequence declares L.
        self.run.set_arg(5, L)
        self.run.start()
        self.run.wait2()
        return self.run.state()

    def close(self):
        self.params = None
        self.run = None
        self.kern = None
        self.ctx = None

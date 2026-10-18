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

# Match the xclbin path's .wait(60000). Without a bound a wedged dispatch hangs
# the host instead of surfacing as a non-COMPLETED state the caller already
# raises on. Use wait(ms), not wait2's timeout overload: that one takes a
# std::chrono::duration with no pybind caster registered, so it cannot be called
# from Python at all (every argument type raises TypeError).
DISPATCH_TIMEOUT_MS = 60000


# AIR names a length parameter after the sequence argument it holds.
_LEN_PARAM = re.compile(r"_arglen_\d+$")


def _entries(path):
    """The parameter lines of params.txt: a count, then one line per parameter
    (joint bounds follow them)."""
    lines = [ln.split() for ln in Path(path).read_text().split("\n") if ln.strip()]
    return lines[1 : 1 + int(lines[0][0])]


def parse_length_param(path):
    """Name of the KV readback's block-count parameter, or None.

    A DECODE_RT_ROUNDS build reads back ceil(L/16) KV blocks; the host writes
    that count per token.
    """
    found = [ln[0] for ln in _entries(path) if _LEN_PARAM.search(ln[0])]
    if len(found) > 1:
        raise RuntimeError(f"{path}: more than one length parameter: {found}")
    return found[0] if found else None


def parse_params(path):
    """Read params.txt into (append_name, scale, addend, mask_name).

    The names are NOT fixed across models. AIR names a BD-offset parameter after
    the sequence argument it is affine in and puts the coefficients in the
    suffix, so llama's REGION_W=256 gives `..._x256_m256` (L*256 - 256) while
    gemma's 512 gives `..._x512_m512`. Hardcoding either writes the wrong KV
    address for the other, silently. Classify by the kind column instead --
    `addr` is the BD offset, `core` the herd RTP -- and take the arithmetic from
    the suffix rather than assuming it. A readback length parameter is `addr`
    kind too; see parse_length_param.
    """
    entries = _entries(path)
    addr = [ln[0] for ln in entries if ln[3] == "addr" and not _LEN_PARAM.search(ln[0])]
    core = [ln[0] for ln in entries if ln[3] == "core"]
    if len(addr) != 1 or len(core) != 1:
        raise RuntimeError(
            f"{path}: expected exactly one 'addr' and one 'core' parameter, "
            f"got addr={addr} core={core}"
        )
    m = re.search(r"_argoff_(\d+)_x(-?\d+)_([mp])(\d+)$", addr[0])
    if not m:
        raise RuntimeError(f"{path}: cannot read affine coefficients from {addr[0]}")
    scale = int(m.group(2))
    addend = int(m.group(4)) * (-1 if m.group(3) == "m" else 1)
    return addr[0], scale, addend, core[0]


# A build that names its scalars (decode_scratchpad.args) has every parameter
# written from them: an offset is `scale * arg + addend`, a length the argument
# itself, and a core parameter L.
_ARG_PARAM = re.compile(r"_arg(off|len)_(\d+)(?:_x(-?\d+)_([mp])(\d+))?$")


def parse_args(path):
    """Read decode_scratchpad.args into ({name: arg_index}, align, parts, window)."""
    args, align, parts, window = {}, 1, 1, 0
    for ln in Path(path).read_text().split("\n"):
        f = ln.split()
        if not f:
            continue
        if f[0] == "arg":
            args[f[2]] = int(f[1])
        elif f[0] == "align":
            align = int(f[1])
        elif f[0] == "parts":
            parts = int(f[1])
        elif f[0] == "window":
            window = int(f[1])
    return args, align, parts, window


def param_writers(params_path, args):
    """[(param name, fn(values by arg index) -> int)] for every parameter."""
    writers = []
    for ln in _entries(params_path):
        name, kind = ln[0], ln[3]
        if kind == "core":
            writers.append((name, lambda v, n=args["L"]: v[n]))
            continue
        m = _ARG_PARAM.search(name)
        if not m:
            raise RuntimeError(f"{params_path}: no rule for parameter {name}")
        n = int(m.group(2))
        if m.group(1) == "len":
            writers.append((name, lambda v, n=n: v[n]))
            continue
        scale = int(m.group(3) or 1)
        addend = int(m.group(5) or 0) * (-1 if m.group(4) == "m" else 1)
        writers.append((name, lambda v, n=n, s=scale, a=addend: s * v[n] + a))
    return writers


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
        args_path = art_dir / "decode_scratchpad.args"
        self.args = None
        if args_path.exists():
            self.args, self.align, self.parts, self.window = parse_args(args_path)
            self.writers = param_writers(self.params_path, self.args)
            self._check_append_scale(region_w)
        else:
            self._legacy_params(region_w)
        self.ctx = xrt.hw_context(dev, xrt.elf(str(self.elf_path)))
        self.kern = xrt.ext.kernel(self.ctx, KERNEL_NAME)
        self.run = None
        self.params = None

    def _check_append_scale(self, region_w):
        """The KV append's scale is the model's REGION_W; a mismatch means the
        ELF and the driver were built from different geometries."""
        for ln in _entries(self.params_path):
            m = _ARG_PARAM.search(ln[0])
            if m and m.group(1) == "off" and int(m.group(2)) == self.args["L"]:
                if int(m.group(3) or 1) != region_w:
                    raise RuntimeError(
                        f"{self.params_path.name}: {ln[0]} scales L by "
                        f"{m.group(3)}, the driver's REGION_W is {region_w}"
                    )

    def _legacy_params(self, region_w):
        (
            self.append_param,
            self.append_scale,
            self.append_addend,
            self.mask_param,
        ) = parse_params(self.params_path)
        self.length_param = parse_length_param(self.params_path)
        # The scale IS the model's REGION_W. If they disagree, the driver and the
        # build were made from different geometries and every KV append would land
        # in the wrong place -- which reads as bad numerics, not as a mismatch.
        if self.append_scale != region_w:
            raise RuntimeError(
                f"{self.params_path.name} encodes scale {self.append_scale} but the "
                f"driver's REGION_W is {region_w}; the ELF and the driver disagree."
            )

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
        if self.args is not None:
            # The cores' range: ceil(L/16) blocks rounded up to the alignment,
            # and on a sliding wave from the block its window opens in.
            blocks = -(-((L + 15) // 16) // self.align) * self.align
            skip = 0
            if self.window and L > self.window:
                skip = (L - self.window) // 16
                skip -= skip % self.align
            part, swpart = blocks // self.parts, (blocks - skip) // self.parts
            scalars = {
                "L": L,
                "kvb": part,
                "swb": swpart,
                "sws": skip,
                "swm": skip + swpart,
            }
            v = {n: scalars[name] for name, n in self.args.items()}
            for name, fn in self.writers:
                self.params.write(name, np.int32(fn(v)))
        else:
            self._write_legacy(L, np)
        self.params.sync()
        # L is not a kernel argument: the ELF takes only the sequence's buffers,
        # and the scratchpad writes above are all the hardware reads.
        self.run.start()
        self.run.wait(DISPATCH_TIMEOUT_MS)
        return self.run.state()

    def _write_legacy(self, L, np):
        self.params.write(
            self.append_param, np.int32(L * self.append_scale + self.append_addend)
        )
        self.params.write(self.mask_param, np.int32(L))
        if self.length_param:
            self.params.write(self.length_param, np.int32((L + 15) // 16))

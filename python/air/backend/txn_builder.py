# ./python/air/backend/txn_builder.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Call the C++ TXN builder that `aircc --emit-txn-cpp` emits.

A runtime-valued access pattern leaves a scalar in the runtime sequence, so the
instruction stream cannot be frozen into an insts.bin -- it has to be assembled
at dispatch from that scalar. aie-translate emits a header of
`generate_txn_*(...)` functions that do the assembling; this wraps one in a
shared library and calls it from Python, returning the words to upload to the
instruction buffer.

Size that buffer from the returned words, not from the companion insts.bin: the
static build constant-folds writes the dynamic one has to keep separate, so its
stream is the shorter of the two. The length does not depend on the argument
values -- it follows the sequence's op structure -- so one call sizes it for all.
"""

import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

__all__ = ["TxnBuilder", "TxnBuilderError"]


class TxnBuilderError(Exception):
    pass


# aie-translate emits `int64_t`/`size_t` scalars; anything else would be a
# translator change, so fail loudly rather than guess a marshalling.
_CTYPE_OF = {
    "int64_t": ctypes.c_int64,
    "uint64_t": ctypes.c_uint64,
    "size_t": ctypes.c_size_t,
    "int32_t": ctypes.c_int32,
    "uint32_t": ctypes.c_uint32,
}

_SIGNATURE = re.compile(
    r"generate_txn_(?P<name>\w+)\s*\((?P<params>[^)]*)\)", re.MULTILINE
)


def _parse_signatures(header_text):
    """Map each generate_txn_* function to its parameter ctypes."""
    sigs = {}
    for m in _SIGNATURE.finditer(header_text):
        params = []
        for p in m.group("params").split(","):
            p = p.strip()
            if not p:
                continue
            ty = p.rsplit(" ", 1)[0].strip()
            if ty not in _CTYPE_OF:
                raise TxnBuilderError(
                    f"generate_txn_{m.group('name')} takes an unsupported "
                    f"parameter type '{ty}'"
                )
            params.append((ty, _CTYPE_OF[ty]))
        sigs[m.group("name")] = params
    return sigs


def _default_include_dirs():
    dirs = []
    aie_install = os.environ.get("MLIR_AIE_INSTALL_DIR")
    if not aie_install:
        aie_opt = shutil.which("aie-opt")
        if aie_opt:
            aie_install = os.path.dirname(os.path.dirname(os.path.realpath(aie_opt)))
    if aie_install:
        dirs.append(os.path.join(aie_install, "include"))
    return dirs


class TxnBuilder:
    """Compile a `--emit-txn-cpp` header and call its builders.

    >>> b = TxnBuilder("build/air_project/npu.decode.txn.h")
    >>> words = b("main_task", 37)      # 37 = the runtime context length
    """

    def __init__(self, header, workdir=None, include_dirs=None, verbose=False):
        header = os.path.abspath(header)
        if not os.path.exists(header):
            raise TxnBuilderError(f"no TXN builder header at {header}")
        self.header = header
        self.verbose = verbose
        with open(header) as f:
            text = f.read()
        self.signatures = _parse_signatures(text)
        if not self.signatures:
            raise TxnBuilderError(f"{header} declares no generate_txn_* function")

        self._include_dirs = list(include_dirs or _default_include_dirs())
        self._workdir = workdir
        self._lib = None
        self._build()

    @property
    def function_names(self):
        return sorted(self.signatures)

    def _shim_source(self):
        lines = [f'#include "{self.header}"', "#include <cstring>", ""]
        for name, params in self.signatures.items():
            args = ", ".join(f"{ty} a{i}" for i, (ty, _) in enumerate(params))
            # Two-call protocol: pass cap=0 to size the stream, then again with
            # a buffer. The builder is cheap enough that sizing twice is fine.
            lines += [
                f"extern \"C\" long air_txn_{name}({args}{', ' if args else ''}"
                "unsigned int *out, unsigned long cap) {",
                f"  auto r = generate_txn_{name}("
                + ", ".join(f"a{i}" for i in range(len(params)))
                + ");",
                "  if (!r) return -1;",
                "  if (r->size() <= cap) memcpy(out, r->data(), r->size() * 4);",
                "  return (long)r->size();",
                "}",
            ]
        return "\n".join(lines) + "\n"

    def _build(self):
        src = self._shim_source()
        workdir = self._workdir
        if workdir is None:
            # Key the cache on the source so a recompiled model picks up a new
            # builder without the caller having to clean anything.
            with open(self.header, "rb") as f:
                digest = hashlib.sha256(src.encode() + f.read())
            workdir = os.path.join(
                tempfile.gettempdir(), f"air_txn_{digest.hexdigest()[:16]}"
            )
        os.makedirs(workdir, exist_ok=True)
        cpp = os.path.join(workdir, "shim.cpp")
        so = os.path.join(workdir, "shim.so")
        with open(cpp, "w") as f:
            f.write(src)
        if not os.path.exists(so):
            cmd = ["g++", "-O2", "-fPIC", "-shared", "-std=c++17", cpp, "-o", so]
            for d in self._include_dirs:
                cmd += ["-I", d]
            if self.verbose:
                print("TxnBuilder:", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise TxnBuilderError(
                    f"building the TXN builder failed:\n{result.stderr}"
                )
        self.so_path = so
        self._lib = ctypes.CDLL(so)
        for name, params in self.signatures.items():
            fn = getattr(self._lib, f"air_txn_{name}")
            fn.restype = ctypes.c_long
            fn.argtypes = [c for _, c in params] + [
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_ulong,
            ]

    def __call__(self, name, *args):
        """Assemble the instruction stream, returning it as uint32 words."""
        if name not in self.signatures:
            raise TxnBuilderError(
                f"no generate_txn_{name}; have {', '.join(self.function_names)}"
            )
        params = self.signatures[name]
        if len(args) != len(params):
            raise TxnBuilderError(
                f"generate_txn_{name} takes {len(params)} argument(s), got {len(args)}"
            )
        fn = getattr(self._lib, f"air_txn_{name}")
        null = ctypes.POINTER(ctypes.c_uint32)()
        n = fn(*args, null, 0)
        if n < 0:
            raise TxnBuilderError(f"generate_txn_{name}{args} returned no stream")
        words = np.zeros(n, dtype=np.uint32)
        got = fn(*args, words.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), n)
        if got != n:
            raise TxnBuilderError(
                f"generate_txn_{name}{args} is not deterministic: {n} then {got} words"
            )
        return words

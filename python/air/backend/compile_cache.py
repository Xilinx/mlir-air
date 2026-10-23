# ./python/air/backend/compile_cache.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Content-addressed cache for aircc builds.

Enabled by pointing AIR_COMPILE_CACHE at a directory. A build is reused only
when everything it was made from is identical: the module text, aircc's
options, the bytes of every object the module links, and the toolchain. Each
entry's manifest.json records all of those components, so a hit can be traced
back to exactly what it matched on.

    python -m air.backend.compile_cache prune --max-gb 50
"""

import argparse
import datetime
import functools
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import socket
import sys
import tempfile
from pathlib import Path

FORMAT = 1
ENV = "AIR_COMPILE_CACHE"


@functools.lru_cache(maxsize=None)
def _sha256_cached(path, size, mtime_ns):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path):
    st = os.stat(path)
    return _sha256_cached(os.path.realpath(path), st.st_size, st.st_mtime_ns)


def _file_identity(path):
    if not path or not os.path.isfile(path):
        return None
    return _sha256(path)


def _tree_identity(root):
    if not root or not os.path.isdir(root):
        return None
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(_sha256(p).encode())
    return h.hexdigest()


def _which(*names):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _peano_identity(ctx):
    peano = ctx["peano"]
    if not peano:
        return None
    return {
        tool: _file_identity(os.path.join(peano, "bin", tool))
        for tool in ("clang", "clang++", "opt", "llc", "ld.lld")
    }


def _aie_runtime_lib(ctx):
    aiecc = _which("aiecc", "aiecc.py")
    if not aiecc:
        return None
    root = os.path.join(os.path.dirname(os.path.realpath(aiecc)), "..")
    return _tree_identity(os.path.join(root, "aie_runtime_lib"))


# Everything outside the module text and aircc's options that can change what
# aircc produces. Each entry maps a component name to a function of the build
# context returning that component's identity, or None when it is absent. A new
# dependency of the compile flow is made part of the key by adding it here.
TOOLCHAIN = {
    "aircc": lambda ctx: _file_identity(ctx["aircc"]),
    "aiecc": lambda ctx: _file_identity(_which("aiecc", "aiecc.py")),
    "aie-opt": lambda ctx: _file_identity(_which("aie-opt")),
    "aie-translate": lambda ctx: _file_identity(_which("aie-translate")),
    "aie_runtime_lib": _aie_runtime_lib,
    "llvm-link": lambda ctx: _file_identity(_which("llvm-link")),
    "xclbinutil": lambda ctx: _file_identity(
        os.environ.get("AIE_XCLBINUTIL") or _which("xclbinutil")
    ),
    "peano": _peano_identity,
    "mlir_aie": lambda ctx: _package_version("mlir_aie"),
    "llvm-aie": lambda ctx: _package_version("llvm-aie"),
}

_LINK_ATTR = re.compile(r'link_(?:with|files|merge_files)\s*=\s*(\[[^\]]*\]|"[^"]*")')


def _linked_files(module_str):
    """Every object the module names for linking, by its bytes in each place
    aiecc may pick it up from. None if one of them cannot be found."""
    names = sorted(
        {
            n
            for m in _LINK_ATTR.finditer(module_str)
            for n in re.findall(r'"([^"]+)"', m.group(1))
        }
    )
    found = {}
    for name in names:
        where = {
            "air_project": _file_identity(os.path.join("air_project", name)),
            "cwd": _file_identity(name),
        }
        if os.path.isabs(name):
            where = {"path": _file_identity(name)}
        if not any(where.values()):
            return None
        found[name] = where
    return found


def _normalize_options(options):
    """aircc's options as they bear on the output. None if one names a file
    that does not exist."""
    out, i = [], 0
    while i < len(options):
        opt = options[i]
        if opt == "-v":
            i += 1
            continue
        if opt == "--peano":
            out += [opt, "<peano>"]
            i += 2
            continue
        if opt in ("--xclbin-input", "--lower-linalg-to-func"):
            sha = _file_identity(options[i + 1])
            if sha is None:
                return None
            out += [opt, f"sha256:{sha}"]
            i += 2
            continue
        out.append(opt)
        i += 1
    return out


class Entry:
    """One build's slot in the cache: its key, the components the key was made
    from, and the files the build is expected to produce."""

    def __init__(self, root, components, outputs, optional_dirs):
        self.components = components
        blob = json.dumps(components, sort_keys=True).encode()
        self.key = hashlib.sha256(blob).hexdigest()
        self.dir = Path(root) / self.key[:2] / self.key
        self.outputs = outputs
        self.optional_dirs = optional_dirs

    def restore(self):
        """Copy a stored build into place. False if there is none, or if any
        stored file no longer matches the hash recorded for it."""
        manifest = self.dir / "manifest.json"
        if not manifest.is_file():
            return False
        try:
            files = json.loads(manifest.read_text())["files"]
        except (OSError, KeyError, json.JSONDecodeError):
            return False
        for f in files.values():
            blob = self.dir / "blobs" / f["blob"]
            if not blob.is_file() or _sha256(blob) != f["sha256"]:
                return False
        for d in self.optional_dirs:
            if any(Path(path).is_relative_to(d) for path in files):
                shutil.rmtree(d, ignore_errors=True)
        for path, f in files.items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.dir / "blobs" / f["blob"], path)
        os.utime(manifest)
        print(f"XRTBackend: compile cache hit {self.key[:12]}")
        return True

    def store(self):
        """Save the files the build just produced. Best effort: a failure here
        never fails the compile."""
        if not all(os.path.isfile(p) for p in self.outputs):
            return
        paths = list(self.outputs)
        for d in self.optional_dirs:
            if os.path.isdir(d):
                paths += [str(p) for p in sorted(Path(d).rglob("*")) if p.is_file()]
        try:
            self.dir.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(dir=self.dir.parent, prefix=".tmp-"))
            (tmp / "blobs").mkdir()
            files = {}
            for i, path in enumerate(paths):
                shutil.copyfile(path, tmp / "blobs" / str(i))
                files[path] = {
                    "blob": str(i),
                    "sha256": _sha256(tmp / "blobs" / str(i)),
                }
            manifest = {
                "format": FORMAT,
                "key": self.key,
                "components": self.components,
                "files": files,
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "host": socket.gethostname(),
            }
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            try:
                tmp.rename(self.dir)
            except OSError:
                # Another process stored the same build first.
                shutil.rmtree(tmp, ignore_errors=True)
        except OSError as e:
            print(f"XRTBackend: compile cache store skipped: {e}")


def entry_for(
    module_str, options, aircc, peano, outputs, optional_dirs=(), cacheable=True
):
    """The cache entry for this build, or None if caching is off or the build
    cannot be keyed reliably."""
    root = os.environ.get(ENV, "").strip()
    if not root or not cacheable:
        return None
    opts = _normalize_options(options)
    links = _linked_files(module_str)
    if opts is None or links is None:
        return None
    ctx = {"aircc": str(aircc), "peano": peano}
    components = {
        "format": FORMAT,
        "module": hashlib.sha256(module_str.encode()).hexdigest(),
        "options": opts,
        "link_files": links,
        "toolchain": {name: fn(ctx) for name, fn in sorted(TOOLCHAIN.items())},
    }
    return Entry(root, components, list(outputs), list(optional_dirs))


def prune(root, max_bytes):
    """Drop the least recently used entries until the cache fits max_bytes."""
    entries = []
    for manifest in Path(root).glob("??/*/manifest.json"):
        d = manifest.parent
        size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        entries.append((manifest.stat().st_mtime, size, d))
    total = sum(e[1] for e in entries)
    for _, size, d in sorted(entries):
        if total <= max_bytes:
            break
        shutil.rmtree(d, ignore_errors=True)
        total -= size
    for tmp in Path(root).glob("??/.tmp-*"):
        shutil.rmtree(tmp, ignore_errors=True)
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prune", help="evict least recently used entries")
    p.add_argument("--dir", default=os.environ.get(ENV))
    p.add_argument("--max-gb", type=float, required=True)
    args = ap.parse_args(argv)
    if not args.dir:
        ap.error(f"no cache directory: pass --dir or set {ENV}")
    left = prune(args.dir, int(args.max_gb * 2**30))
    print(f"{args.dir}: {left / 2**30:.1f} GiB after pruning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

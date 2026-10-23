# ./python/air/backend/compile_cache.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Content-addressed cache for aircc builds and the kernels they link.

Enabled by pointing AIR_COMPILE_CACHE at a directory. A build is reused only
when everything it was made from is identical:

  - an aircc build: the module text, aircc's options, the bytes of every object
    the module links, and the toolchain (TOOLCHAIN);
  - a kernel compile: the preprocessed source, the compiler's arguments, and
    the compiler (OBJECT_TOOLCHAIN).

Each entry's manifest.json records all of those components, so a hit can be
traced back to exactly what it matched on.

    python compile_cache.py cc <compiler> <args>...   # a cached compile
    python compile_cache.py prune --max-gb 30
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
import subprocess
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


# The toolchain parts of the keys. Each entry maps a component name to a
# function of the build context returning that component's identity, or None
# when it is absent. A new dependency is made part of the key by adding it to
# the registry of the step it affects.
#
# TOOLCHAIN: everything outside the module text and aircc's options that can
# change what aircc produces.
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

# OBJECT_TOOLCHAIN: everything outside the preprocessed source and the
# compiler's arguments that can change a compiled kernel.
OBJECT_TOOLCHAIN = {
    "compiler": lambda ctx: _file_identity(ctx["compiler"]),
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

    def __init__(self, root, components, outputs, optional_dirs, label):
        self.components = components
        self.label = label
        blob = json.dumps(components, sort_keys=True).encode()
        self.key = hashlib.sha256(blob).hexdigest()
        self.dir = Path(root) / self.key[:2] / self.key
        self.outputs = outputs
        self.optional_dirs = optional_dirs

    def _destination(self, f):
        """Where a stored file goes for this build. Stored files are recorded
        by role, not path: a key can leave the output's name out of it."""
        if "output" in f:
            return self.outputs[f["output"]]
        return os.path.join(self.optional_dirs[f["dir"]], f["path"])

    def _discard(self):
        """Remove a damaged entry so the rebuild can be stored in its place.
        Entries are published whole by a rename, so one that exists and does
        not check out is damaged, never half-written."""
        aside = self.dir.parent / f".tmp-{os.getpid()}-{self.key[:12]}"
        try:
            self.dir.rename(aside)
        except OSError:
            return
        shutil.rmtree(aside, ignore_errors=True)

    def restore(self):
        """Copy a stored build into place. False if there is none, or if any
        stored file no longer matches the hash recorded for it."""
        if not self.dir.exists():
            return False
        try:
            files = json.loads((self.dir / "manifest.json").read_text())["files"]
            dests = [self._destination(f) for f in files]
            for f in files:
                blob = self.dir / "blobs" / f["blob"]
                if _sha256(blob) != f["sha256"]:
                    raise ValueError(blob)
        except (OSError, KeyError, IndexError, ValueError):
            self._discard()
            return False
        for i, d in enumerate(self.optional_dirs):
            if any(f.get("dir") == i for f in files):
                shutil.rmtree(d, ignore_errors=True)
        for f, dest in zip(files, dests):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.dir / "blobs" / f["blob"], dest)
        os.utime(self.dir / "manifest.json")
        print(f"compile cache hit {self.key[:12]}: {self.label}", flush=True)
        return True

    def store(self):
        """Save the files the build just produced. Best effort: a failure here
        never fails the compile."""
        if not all(os.path.isfile(p) for p in self.outputs):
            return
        files = [({"output": i}, p) for i, p in enumerate(self.outputs)]
        for i, d in enumerate(self.optional_dirs):
            if os.path.isdir(d):
                files += [
                    ({"dir": i, "path": str(p.relative_to(d))}, p)
                    for p in sorted(Path(d).rglob("*"))
                    if p.is_file()
                ]
        try:
            self.dir.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(dir=self.dir.parent, prefix=".tmp-"))
            (tmp / "blobs").mkdir()
            for n, (f, src) in enumerate(files):
                shutil.copyfile(src, tmp / "blobs" / str(n))
                f.update(blob=str(n), sha256=_sha256(tmp / "blobs" / str(n)))
            manifest = {
                "format": FORMAT,
                "key": self.key,
                "components": self.components,
                "files": [f for f, _ in files],
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
            print(f"compile cache store skipped: {e}", flush=True)


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
    return Entry(root, components, list(outputs), list(optional_dirs), "aircc")


# Options whose value names an output of the compile, not an input.
_OUTPUT_OPTS = ("-o", "-MF", "-MT", "-MQ")


def _object_entry(cmd, cwd, env):
    """The cache entry for one compiler invocation, or None if caching is off
    or the invocation is not a single compile to one output."""
    root = os.environ.get(ENV, "").strip()
    args = [str(a) for a in cmd[1:]]
    if not root or args.count("-o") != 1 or not ({"-c", "-S"} & set(args)):
        return None
    out = args[args.index("-o") + 1]
    compiler = str(cmd[0])
    if os.sep not in compiler:
        path = (env or os.environ).get("PATH")
        compiler = shutil.which(compiler, path=path) or compiler

    # The preprocessed source stands in for the source and every header it
    # includes. Preprocessing also writes the dependency file a build asked
    # for, so a hit leaves the same files behind as a compile.
    with tempfile.TemporaryDirectory() as tmp:
        pre = [a for a in args if a not in ("-c", "-S", "-emit-llvm")]
        pre[pre.index("-o") + 1] = os.path.join(tmp, "pre")
        pre.append("-E")
        if "-MF" in args and not {"-MT", "-MQ"} & set(args):
            pre += ["-MT", out]
        r = subprocess.run([compiler] + pre, cwd=cwd, env=env, capture_output=True)
        if r.returncode != 0:
            return None
        source = _sha256(os.path.join(tmp, "pre"))

    normalized = []
    for i, a in enumerate(args):
        if i and args[i - 1] in _OUTPUT_OPTS:
            normalized.append("<output>")
        elif a not in ("-MD", "-MMD", "-MP"):
            normalized.append(a)
    ctx = {"compiler": compiler}
    components = {
        "format": FORMAT,
        "kind": "object",
        "args": normalized,
        "source": source,
        "toolchain": {name: fn(ctx) for name, fn in sorted(OBJECT_TOOLCHAIN.items())},
    }
    path = os.path.join(cwd, out) if cwd else out
    return Entry(root, components, [path], [], os.path.basename(out))


def run_compiler(cmd, **kwargs):
    """subprocess.run for one kernel compile, reusing the output from
    AIR_COMPILE_CACHE when the compiler, its arguments and the preprocessed
    source all match. Without the cache it is subprocess.run."""
    entry = _object_entry(cmd, kwargs.get("cwd"), kwargs.get("env"))
    if entry is not None and entry.restore():
        empty = "" if kwargs.get("text") or kwargs.get("universal_newlines") else b""
        return subprocess.CompletedProcess(cmd, 0, empty, empty)
    result = subprocess.run(cmd, **kwargs)
    if entry is not None and result.returncode == 0:
        entry.store()
    return result


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
    p = sub.add_parser("cc", help="run a compiler through the cache")
    p.add_argument("compiler_cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if args.cmd == "cc":
        return run_compiler(args.compiler_cmd).returncode
    if not args.dir:
        ap.error(f"no cache directory: pass --dir or set {ENV}")
    left = prune(args.dir, int(args.max_gb * 2**30))
    print(f"{args.dir}: {left / 2**30:.1f} GiB after pruning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

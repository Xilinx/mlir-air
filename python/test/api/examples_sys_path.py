# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root | FileCheck %s
"""programming_examples/ and llms/ must stay off sys.path.

Both are directories of short, generic names -- bottleneck, conv2d, softmax,
gelu, shared, verify -- and putting either on sys.path makes every one of them
a top-level module name. An example then shadows any installed package that
shares its name, for the whole interpreter. That is not hypothetical: pandas
probes for `bottleneck` as an optional dependency, imported
programming_examples/bottleneck/ instead, found no __version__ on it, and
raised out of `import pandas` inside smolvla's CPU reference.

The examples and llms/shared are reached through the air_examples package
instead, so nothing here needs a naming convention to stay safe.

Every sys.path argument is resolved symbolically rather than matched as source
text: a check that recognises only the shapes it was written against passes on
the shape it has not seen.
"""

import ast
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
ROOT = SRC / "programming_examples"
BANNED = {ROOT.resolve(), (ROOT / "llms").resolve()}

# The api tests drive the example modules, so they publish the same names if
# they set sys.path the old way.
SCANNED = [ROOT, SRC / "python" / "test" / "api"]


def resolve(node, env, this):
    """Value of a path expression built from __file__, or None if not static."""
    if isinstance(node, ast.Name):
        return this if node.id == "__file__" else env.get(node.id)
    if isinstance(node, ast.Constant):
        return None
    # The api tests take the source root as sys.argv[1].
    if isinstance(node, ast.Subscript) and ast.unparse(node.value).replace(
        " ", ""
    ).endswith("sys.argv"):
        return SRC
    if isinstance(node, ast.Call):
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name in ("str", "Path") and node.args:
            return resolve(node.args[0], env, this)
        if name == "resolve" and isinstance(fn, ast.Attribute):
            return resolve(fn.value, env, this)
        if name == "abspath" and node.args:
            return resolve(node.args[0], env, this)
        if name == "dirname" and node.args:
            p = resolve(node.args[0], env, this)
            return p.parent if p else None
        if name == "join" and node.args:
            head = resolve(node.args[0], env, this)
            if head is None:
                return None
            for a in node.args[1:]:
                if not (isinstance(a, ast.Constant) and isinstance(a.value, str)):
                    return None
                head = head / a.value
            return head
        return None
    if isinstance(node, ast.Attribute):
        if node.attr == "parent":
            p = resolve(node.value, env, this)
            return p.parent if p else None
        return None
    if isinstance(node, ast.Subscript):  # Path(__file__).resolve().parents[N]
        v = node.value
        if isinstance(v, ast.Attribute) and v.attr == "parents":
            p = resolve(v.value, env, this)
            n = node.slice.value if isinstance(node.slice, ast.Constant) else None
            if p is not None and isinstance(n, int):
                for _ in range(n + 1):
                    p = p.parent
                return p
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
        # `SRC / "programming_examples"` and `sys.argv[1] + "/programming_examples"`
        # are the same statement; the tests use the second form, which this
        # missed until one of them published llms/ past the check.
        left = resolve(node.left, env, this)
        right = node.right.value if isinstance(node.right, ast.Constant) else None
        if left is None or not isinstance(right, str):
            return None
        return left / right.lstrip("/\\")
    return None


def imported_constants(tree, f):
    """Module-level path constants this file imports from a sibling example.

    `from shared.infra.external_kernels import _PROJ_ROOT` then
    `sys.path.insert(0, str(_PROJ_ROOT))` publishes programming_examples/ just
    as surely as computing the path inline, but the value lives in another
    file. One of those reached review because this only looked at local
    assignments.
    """
    env = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        rel = node.module.replace("air_examples.", "").replace(".", "/")
        for cand in (ROOT / f"{rel}.py", ROOT / "llms" / f"{rel}.py"):
            if not cand.is_file():
                continue
            try:
                sub = ast.parse(cand.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            wanted = {a.name for a in node.names}
            subenv = {}
            for n in ast.walk(sub):
                if (
                    isinstance(n, ast.Assign)
                    and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)
                ):
                    v = resolve(n.value, subenv, cand.resolve())
                    if v is not None:
                        subenv[n.targets[0].id] = v
                        if n.targets[0].id in wanted:
                            env[n.targets[0].id] = v
            break
    return env


def offenders(f):
    """Paths this module puts on sys.path that are in BANNED."""
    try:
        tree = ast.parse(f.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    this = f.resolve()
    env, loops, out = imported_constants(tree, f), {}, []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            v = resolve(node.value, env, this)
            if v is not None:
                env[node.targets[0].id] = v
        # `for p in (A, B, C): sys.path.insert(0, p)` -- p takes every value
        elif isinstance(node, ast.For) and isinstance(node.iter, (ast.Tuple, ast.List)):
            if isinstance(node.target, ast.Name):
                vals = [resolve(e, env, this) for e in node.iter.elts]
                loops[node.target.id] = [v for v in vals if v is not None]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("insert", "append", "extend")
            and ast.unparse(node.func.value).replace(" ", "").endswith("sys.path")
        ):
            arg = node.args[-1]
            cands = (
                loops[arg.id]
                if isinstance(arg, ast.Name) and arg.id in loops
                else [resolve(arg, env, this)]
            )
            out += [c for c in cands if c is not None and c.resolve() in BANNED]
    # `sys.path[:0] = [...]` -- a slice assignment, not a method call
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Subscript)
            and ast.unparse(t.value).replace(" ", "").endswith("sys.path")
            for t in node.targets
        ):
            elts = (
                node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else []
            )
            for e in elts:
                v = resolve(e, env, this)
                if v is not None and v.resolve() in BANNED:
                    out.append(v)
    return out


REGISTER = 'sys.modules.setdefault("air_examples"'
IMPORTS = re.compile(r"^(from|import) air_examples\.")


def registers_too_late(f):
    """True if a module-scope air_examples import runs before the registration.

    Nothing fails at import of the file that got this wrong -- it fails in
    whichever model imports it, at run time, which is how it reached a green
    local test run once already.
    """
    lines = f.read_text(encoding="utf-8", errors="replace").split("\n")
    first = next((i for i, l in enumerate(lines) if IMPORTS.match(l)), None)
    if first is None:
        return False
    reg = next((i for i, l in enumerate(lines) if REGISTER in l), None)
    return reg is None or reg > first


# Directory names that only resolve as top-level modules when one of the banned
# roots is on sys.path. Importing one is the other half of the same defect:
# removing the inserts without rewriting these leaves a ModuleNotFoundError
# that no module-level import mentions, so an import sweep stays green.
PUBLISHED = {d.name for p in (ROOT, ROOT / "llms") for d in p.iterdir() if d.is_dir()}
PUBLISHED -= {"__pycache__"}


def stale_imports(f):
    """Names imported from a directory that is no longer on sys.path."""
    try:
        tree = ast.parse(f.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    own = f.resolve().parent.name  # a model importing its own package is fine
    out = []
    for node in ast.walk(tree):
        root = None
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            root = node.module.split(".")[0]
        elif isinstance(node, ast.Import):
            for a in node.names:
                if "." in a.name:
                    root = a.name.split(".")[0]
        # importlib.import_module("qwen3_1_7b.verify_adapter") and __import__:
        # resolved from a string, so nothing above sees them.
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ("import_module", "__import__") and node.args:
                a = node.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    root = a.value.split(".")[0]
        if root and root != own and root in PUBLISHED:
            out.append(root)
    return out


bad, late, stale = [], [], []
for scan in SCANNED:
    for f in sorted(scan.rglob("*.py")):
        if "__pycache__" in f.parts or f.name == "examples_sys_path.py":
            continue
        for p in offenders(f):
            bad.append(f"{f.relative_to(SRC).as_posix()} -> {p.name}/")
        if registers_too_late(f):
            late.append(f.relative_to(SRC).as_posix())
        for name in stale_imports(f):
            stale.append(f"{f.relative_to(SRC).as_posix()} imports {name}")

for line in bad:
    print("PUBLISHES:", line)
print(f"{len(bad)} files put programming_examples/ or llms/ on sys.path")
# CHECK: 0 files put programming_examples/ or llms/ on sys.path

for line in late:
    print("LATE:", line)
print(f"{len(late)} files import air_examples before registering it")
# CHECK: 0 files import air_examples before registering it

for line in stale:
    print("STALE:", line)
print(f"{len(stale)} files import a name that needs the removed paths")
# CHECK: 0 files import a name that needs the removed paths

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Sweep fused-decode throughput against context length into JSON.

The nightly benchmark's other capture (`extract_perf.py`) profiles one point,
and that point is the 6-token prompt the profile lit passes -- i.e. an empty KV
cache. Decode throughput is dominated by KV streaming, so that single number
sits at the flattering end of a wide range (llama-3.2-1B measures 68.97 tok/s at
1k context and 5.77 at 128k) and would not move if the KV path regressed. This
walks the axis instead.

Decode-only on purpose. A context is set by sizing the KV cache and telling the
device to attend over it, not by prefilling a real prompt of that length, so a
32k point costs a template build and a few seconds rather than a 32k prefill.
Contents are synthetic: this is LATENCY ONLY and never a correctness gate --
`make verify` is the only correctness gate.

TTFT is deliberately not swept here. It is a prefill measurement and its x-axis
is prompt length, not KV depth, so it needs a real prompt and belongs in its own
table.

Per context the run is: build the decode template pair at that ATTN_MAXL, guard
that the build actually produced a *new* xclbin, then dispatch. Contexts listed
in --expect-fail may fail without failing the run: whether a model reaches
64k/128k depends on how much host memory XRT will pin for its KV BO, which has
been observed to differ between two otherwise similar Krackan boxes. That makes
it a property of the machine, not of the design, so the point is recorded with
its status rather than either hidden or allowed to red the job.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FUSED_DECODE = HERE.parents[1] / "fused_decode"
BENCH_EXE = FUSED_DECODE / "bench_decode.exe"

# "[bench] n=20  mean 14.488 ms  sd ...  (69.02 tok/s)"
BENCH_RE = re.compile(r"^\[bench\].*?mean\s+([\d.]+)\s*ms.*?\(([\d.]+)\s*tok/s\)", re.M)
# bench_qwen.py's summary line: "CSV,<ctx>,<ms>,...,<tok_s>"
QWEN_CSV_RE = re.compile(r"^CSV,[^,]*,([\d.]+),[^,]*,[^,]*,([\d.]+)", re.M)

# Device-side "the dispatch did not complete" evidence. Deliberately specific:
# an earlier version matched a bare "timeout", which also matches the shell's
# own "timeout: failed to run command ..." when the bench binary is missing, and
# three models were logged as device failures when the real fault was a deleted
# executable. A missing binary is now caught up front instead.
XRT_FAIL_RE = re.compile(
    r"ERT_CMD_STATE|did not complete|not COMPLETED|command timeout|xrt::error", re.I
)


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def _run(cmd, cwd=None, timeout=5400, env=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _make_vars(args):
    """Toolchain overrides to hand `make`, in the same form the other lits use.

    lit supplies these as substitutions rather than in the environment, and the
    runner shells out to make itself, so without forwarding them every build
    fails at preflight-peano. The profile/verify lits pass PEANO_INSTALL_DIR on
    each make line for exactly this reason.
    """
    return [f"PEANO_INSTALL_DIR={args.peano_dir}"] if args.peano_dir else []


def build_templates(makefile, ctx, workdir, template_dir, log, args):
    """Build the decode_L{ctx} / decode_L{ctx-1} pair and stage it in workdir.

    Uses _compile_decode_build, not compile-decode: the latter is idempotent on
    a build stamp that does not include the context, so it would skip and leave
    the previous context's templates in place -- a sweep that benches the same
    binary eight times.
    """
    ref = ctx - 1
    for l in (ctx, ref):
        for ext in ("xclbin", "insts.bin"):
            for d in (workdir, template_dir):
                (d / f"decode_L{l}.{ext}").unlink(missing_ok=True)
    shutil.rmtree(FUSED_DECODE / "air_project", ignore_errors=True)

    r = _run(
        ["make", "-f", str(makefile), "_compile_decode_build", f"LBUILD={ctx}"]
        + _make_vars(args)
    )
    log.write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        err = re.search(
            r"error: .{0,90}|Killed|Cannot allocate|out of memory", r.stdout + r.stderr
        )
        if err:
            return None, f"build_fail: {err.group(0)}"
        # No recognised pattern: carry the tail of the output, because build.log
        # stays in the lit working directory and never reaches the artifact.
        tail = " / ".join(
            l.strip()
            for l in (r.stdout + r.stderr).strip().splitlines()[-3:]
            if l.strip()
        )
        return None, f"build_fail: {tail[:300] or 'no output'}"

    for l in (ctx, ref):
        for ext in ("xclbin", "insts.bin"):
            src = template_dir / f"decode_L{l}.{ext}"
            if not src.exists():
                return None, f"no_template: {src.name} not produced"
            shutil.copy2(src, workdir / src.name)
    return workdir / f"decode_L{ctx}.xclbin", None


def bench_decode(workdir, ctx, args):
    from decode_geometry import as_flags, geometry

    geom = geometry(args.bench_model, args.vocab_chunk_i2, ctx, args.w_elems)
    cmd = (
        f"{BENCH_EXE} --dir . --base-l {ctx} --ref-l {ctx - 1} --l {ctx} "
        f"--iters {args.iters} --warmup {args.warmup} {as_flags(geom)}"
    )
    r = _run(cmd, cwd=workdir, timeout=1800)
    return r.stdout + r.stderr, BENCH_RE.search(r.stdout + r.stderr)


def bench_qwen(workdir, ctx, args):
    env = dict(os.environ, QWEN_NLAYERS=str(args.n_layers), W_DUAL_CHAN="1")
    r = _run(
        [
            sys.executable,
            str(HERE / "bench_qwen_decode.py"),
            str(ctx),
            str(workdir),
            str(args.iters),
            str(args.warmup),
        ],
        cwd=FUSED_DECODE,
        timeout=1800,
        env=env,
    )
    return r.stdout + r.stderr, QWEN_CSV_RE.search(r.stdout + r.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-name", required=True, help="example dir name, e.g. llama32_1b_q4nx"
    )
    p.add_argument("--makefile", required=True, type=Path)
    p.add_argument(
        "--template-dir",
        required=True,
        type=Path,
        help="where _compile_decode_build leaves decode_L<N>.*",
    )
    p.add_argument(
        "--contexts",
        required=True,
        help="comma-separated, e.g. 1024,2048,4096,8192,16384,32768,65536,131072",
    )
    p.add_argument(
        "--expect-fail",
        default="",
        help="contexts allowed to fail without failing the run",
    )
    p.add_argument(
        "--driver", choices=("bench_decode", "bench_qwen"), default="bench_decode"
    )
    p.add_argument("--bench-model", help="DECODE_MODEL for bench_decode geometry")
    p.add_argument("--vocab-chunk-i2", help="VOCAB_CHUNK_I2 for bench_decode geometry")
    p.add_argument("--w-elems", type=int, help="weight element count")
    p.add_argument("--n-layers", type=int, default=36, help="bench_qwen only")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument(
        "--workdir", type=Path, default=Path("sweep_out"), help="per-context scratch"
    )
    p.add_argument(
        "--peano-dir",
        default="",
        help="Peano install (lit's %PEANO_INSTALL_DIR), forwarded to make.",
    )
    p.add_argument(
        "--xrt-dir",
        default="",
        help="XRT install (lit's %XRT_DIR). bench_decode.exe's build rule needs "
        "XILINX_XRT set, and the lit environment does not export it.",
    )
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    sys.path.insert(0, str(HERE))
    contexts = [int(c) for c in a.contexts.split(",") if c.strip()]
    expect_fail = {int(c) for c in a.expect_fail.split(",") if c.strip()}

    if a.driver == "bench_decode":
        if not BENCH_EXE.exists():
            # `make clean` in any model dir removes this; rebuild rather than
            # letting every cell fail with a shell "No such file" that reads
            # like a device fault.
            env = dict(os.environ)
            if not env.get("XILINX_XRT") and a.xrt_dir:
                env["XILINX_XRT"] = a.xrt_dir
            r = _run(
                ["make", "-f", str(FUSED_DECODE / "Makefile"), "bench-decode-exe"]
                + _make_vars(a),
                env=env,
            )
            if not BENCH_EXE.exists():
                print(
                    f"FATAL: cannot build {BENCH_EXE}\n{r.stdout}{r.stderr}",
                    file=sys.stderr,
                )
                return 2
        for f in ("bench_model", "vocab_chunk_i2", "w_elems"):
            if getattr(a, f) is None:
                p.error(f"--driver bench_decode needs --{f.replace('_', '-')}")

    # Absolute: bench_qwen runs with cwd=fused_decode and bench_decode with
    # cwd=workdir, so a relative path resolves differently for the two drivers.
    a.workdir = a.workdir.resolve()
    a.workdir.mkdir(parents=True, exist_ok=True)
    points, seen_md5, hard_fail = [], {}, False

    for ctx in contexts:
        wd = a.workdir / f"L{ctx}"
        wd.mkdir(parents=True, exist_ok=True)
        rec = {
            "context_len": ctx,
            "decode_tokens_per_sec": None,
            "ms_per_token": None,
            "status": "ok",
        }

        xclbin, err = build_templates(
            a.makefile, ctx, wd, a.template_dir, wd / "build.log", a
        )
        if err:
            rec.update(status=err.split(":")[0], detail=err)
        else:
            h = _md5(xclbin)
            rec["xclbin_md5"] = h
            if h in seen_md5:
                # Two contexts producing the same binary means a stale copy got
                # benched; that silently reports the wrong context's number.
                rec.update(
                    status="stale_template",
                    detail=f"xclbin md5 {h} == context {seen_md5[h]}",
                )
            else:
                seen_md5[h] = ctx
                runner = bench_decode if a.driver == "bench_decode" else bench_qwen
                out, m = runner(wd, ctx, a)
                (wd / "bench.log").write_text(out)
                if m:
                    rec["ms_per_token"] = float(m.group(1))
                    rec["decode_tokens_per_sec"] = float(m.group(2))
                elif XRT_FAIL_RE.search(out):
                    rec.update(
                        status="xrt_incomplete", detail="dispatch did not complete"
                    )
                else:
                    hint = re.search(
                        r"mmap.{0,60}|err=-\d+|std::bad_alloc|host.mem.{0,40}", out
                    )
                    rec.update(
                        status="run_fail",
                        detail=hint.group(0) if hint else "no bench line",
                    )

        if rec["status"] != "ok":
            if ctx in expect_fail:
                rec["status"] = "expected_fail"
            else:
                hard_fail = True
        points.append(rec)
        tps = rec["decode_tokens_per_sec"]
        # `is not None`, not truthiness: a genuine 0.00 tok/s is a measurement,
        # and printing it as the status would hide a real (if pathological)
        # result behind what looks like a failure.
        print(
            f"[sweep] {a.model_name} ctx={ctx:<7} "
            f"{f'{tps:.2f} tok/s' if tps is not None else rec['status']}",
            flush=True,
        )

    a.out.write_text(
        json.dumps(
            {
                "model": a.model_name,
                "axis": "context_len",
                "metric": "decode_tokens_per_sec",
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "runner": os.environ.get("RUNNER_NAME", ""),
                "run_params": {
                    "iters": a.iters,
                    "warmup": a.warmup,
                    "driver": a.driver,
                },
                "points": points,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[sweep] wrote {a.out}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Full-output B8 attention check on the graph and object the runtime uses.

Reference: FP32 SDPA cast BF16, rtol=.016 / atol=.1, zero outliers.
Repeat, row-isolation and visibility probes compare all applicable BF16 words.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from ml_dtypes import bfloat16
from target_config import HERE, TargetConfig, load_layout
from fused_attention_graph import ParallelAttentionGraph


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, j):
    Path(p).write_text(json.dumps(j, indent=2) + "\n")


def module(cfg):
    f = load_layout(cfg)
    f.BATCH = 8
    f.CAUSAL = cfg.causal
    f.DYNSEQ_TRIP = True
    f.EARLY_KEY_RELEASE = cfg.attention_early_key_release

    @f.module_builder
    def build():
        bf, i32 = f.BF16Type.get(), f.IntegerType.get_signless(32)

        def mem(n):
            return f.MemRefType.get([n], bf)

        def ix(n):
            return f.arith.ConstantOp.create_index(n)

        def ci(n):
            return f.arith.ConstantOp(f.IntegerAttr.get(i32, n), None).result

        pa = ParallelAttentionGraph(f)
        for kind in ("K", "V"):
            f.channel_decl("inKV_" + kind, size=[2])

        @f.FuncOp.from_py_func(
            mem(8 * 4096),
            mem(2 * cfg.capacity * 512),
            mem(2 * cfg.capacity * 512),
            mem(8 * 4096),
            i32,
        )
        def attention(Q, K, V, O, L):
            @f.launch(sizes=[1, 1], operands=[Q, K, V, O, L])
            def launch(*args):
                q, k, v, o, length = args[4:]
                pa.query(q)
                rounds = pa.rounds(length)
                for group in range(2):
                    for kind, buf in (("K", k), ("V", v)):
                        f.ChannelPut(
                            "inKV_" + kind,
                            buf,
                            indices=[ix(group)],
                            offsets=[ix(0), ix(0), ix(group * cfg.capacity * 512)],
                            sizes=[rounds, ix(16), ix(512)],
                            strides=[ix(8192), ix(512), ix(1)],
                        )
                pa.gather(o)

                @f.segment(name="seg", operands=[length])
                def seg(Ls):
                    pa.emit(Ls, ci(1))

    return build()


def build(a):
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    seed = json.loads((a.source / "manifest.json").read_text())
    cfg = replace(
        TargetConfig(**seed["config"]),
        causal=a.causal,
        role="target" if a.causal else "drafter",
        layers=36 if a.causal else 5,
    )
    assert cfg.batch == 8
    obj = a.source / "objects/attn_fused.ll"
    m = dict(
        status="building",
        batch=8,
        capacity=cfg.capacity,
        causal=a.causal,
        pv_reuse=cfg.attention_pv_reuse,
        source_artifact=str(a.source.resolve()),
        source_manifest_sha256=sha(a.source / "manifest.json"),
        object_sha256=sha(obj),
        source_sha256=sha(__file__),
        graph_sha256=sha(HERE / "fused_attention_graph.py"),
        gate=dict(
            reference="FP32 SDPA cast BF16", rtol=0.016, atol=0.1, allowed_outliers=0
        ),
    )
    write(out / "manifest.json", m)
    shutil.copy2(obj, out / "attn_fused.ll")
    shutil.copy2(__file__, out / "source_probe.py")
    try:
        graph = module(cfg)
        assert graph.operation.verify()
        (out / "input.mlir").write_text(str(graph))
        from air.backend.xrt import XRTBackend

        old = Path.cwd()
        try:
            os.chdir(out)
            art = XRTBackend(
                omit_while_true_loop=False,
                kernel_name="MLIR_AIE",
                stack_size=6080,
                omit_pingpong="all",
                target_device="npu2",
                use_lock_race_condition_fix_v2=True,
                emit_txn_cpp=True,
            ).compile(
                graph, output_binary_name="attention", insts="attention.insts.bin"
            )
            shutil.copy2(out / art.txn_header, out / "attention.txn.h")
        finally:
            os.chdir(old)
        m.update(
            status="compiled",
            artifacts={p.name: sha(p) for p in out.glob("attention.*") if p.is_file()},
        )
    except BaseException as e:
        m.update(status="failed", error=repr(e))
        raise
    finally:
        write(out / "manifest.json", m)


def reference(q, k, v, p, causal):
    k = np.repeat(k.astype(np.float32), 4, axis=1)
    v = np.repeat(v.astype(np.float32), 4, axis=1)
    score = np.einsum("mhd,lhd->mhl", q.astype(np.float32), k) / np.sqrt(
        np.float32(128)
    )
    visible = p + (np.arange(8) + 1 if causal else np.full(8, 8))
    score = np.where(
        np.arange(len(k))[None, None, :] < visible[:, None, None], score, -np.inf
    )
    prob = np.exp(score - score.max(axis=-1, keepdims=True))
    prob /= prob.sum(axis=-1, keepdims=True)
    return np.einsum("mhl,lhd->mhd", prob, v).astype(bfloat16)


def run(a):
    import pyxrt as xrt
    from air.backend.txn_builder import TxnBuilder

    out = a.output.resolve()
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "compiled" and m["batch"] == 8
    for name, h in m["artifacts"].items():
        assert sha(out / name) == h
    assert not a.result.exists()
    cap = m["capacity"]
    dev = xrt.device(0)
    xb = xrt.xclbin(str(out / "attention.xclbin"))
    dev.register_xclbin(xb)
    context = xrt.hw_context(dev, xb.get_uuid())
    kernel = xrt.kernel(context, "MLIR_AIE")
    builder = TxnBuilder(str(out / "attention.txn.h"))
    assert len(builder.function_names) == 1
    fn = builder.function_names[0]
    ins = builder(fn, cap - 7)
    ib = xrt.bo(dev, ins.nbytes, xrt.bo.cacheable, kernel.group_id(1))
    sizes = [8 * 4096, 2 * cap * 512, 2 * cap * 512, 8 * 4096]
    bos = [
        xrt.bo(dev, n * 2, xrt.bo.host_only, kernel.group_id(i + 3))
        for i, n in enumerate(sizes)
    ]
    to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    fr = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    def call(q, k, v, p):
        inst = builder(fn, p + 1)
        assert inst.nbytes <= ib.size()
        ib.write(inst, 0)
        ib.sync(to)
        arrays = [
            q.reshape(-1),
            k.reshape(cap, 2, 512).transpose(1, 0, 2).copy().reshape(-1),
            v.reshape(cap, 2, 512).transpose(1, 0, 2).copy().reshape(-1),
            np.full(sizes[3], 0x7FC1, np.uint16),
        ]
        for bo, data in zip(bos, arrays):
            bo.write(data.view(np.int16), 0)
            bo.sync(to)
        status = kernel(3, ib, inst.size, *bos, p + 1).wait(10000)
        assert str(status).endswith("COMPLETED"), str(status)
        bos[3].sync(fr)
        return np.frombuffer(bos[3].map(), np.uint16).copy().reshape(8, 32, 128)

    result = dict(
        passed=False,
        status="running",
        manifest_sha256=sha(out / "manifest.json"),
        source_sha256=sha(__file__),
        gate=m["gate"],
        cases=[],
    )
    try:
        for p in a.prefixes:
            assert 0 <= p <= cap - 8
            for seed in (917, 2027):
                rng = np.random.default_rng(seed + p)
                q = rng.normal(0, 0.7, (8, 32, 128)).astype(bfloat16)
                k = rng.normal(0, 0.7, (cap, 8, 128)).astype(bfloat16)
                v = rng.normal(0, 0.7, (cap, 8, 128)).astype(bfloat16)
                expected = reference(q, k, v, p, m["causal"]).astype(np.float32)
                first = None
                for rep in range(3):
                    actual = call(q, k, v, p)
                    f = actual.view(bfloat16).astype(np.float32)
                    delta = np.abs(f - expected)
                    bad = int(np.count_nonzero(delta > 0.1 + 0.016 * np.abs(expected)))
                    assert np.isfinite(f).all() and bad == 0, (
                        p,
                        seed,
                        rep,
                        bad,
                        float(delta.max()),
                    )
                    if first is not None:
                        assert np.array_equal(first, actual), "repeat"
                    else:
                        first = actual
                altered = q.copy()
                altered[4] = -altered[4]
                got = call(altered, k, v, p)
                assert np.array_equal(
                    got[np.arange(8) != 4], first[np.arange(8) != 4]
                ), "row isolation"
                altered = v.copy()
                altered[p + 7] += np.asarray(4, bfloat16)
                got = call(q, k, altered, p)
                if m["causal"]:
                    assert np.array_equal(got[:7], first[:7]) and not np.array_equal(
                        got[7], first[7]
                    ), "causal visibility"
                else:
                    assert all(
                        not np.array_equal(got[i], first[i]) for i in range(8)
                    ), "bidirectional visibility"
                altered = v.copy()
                altered[p + 8 :] = np.asarray(32, bfloat16)
                assert np.array_equal(
                    call(q, k, altered, p), first
                ), "stale future visibility"
                file = a.result.with_name(a.result.stem + f"_p{p}_s{seed}.npy")
                np.save(file, first)
                result["cases"].append(
                    dict(
                        prefix=p,
                        seed=seed,
                        compared_elements=first.size,
                        repeats=3,
                        out_of_tolerance=bad,
                        max_abs_error=float(delta.max()),
                        repeat_exact=True,
                        row_isolation=True,
                        visibility=True,
                        stale_future_exact=True,
                        output_sha256=sha(file),
                    )
                )
                write(a.result, result)
        result.update(passed=True, status="passed")
    except BaseException as e:
        result.update(status="failed", error=repr(e))
        raise
    finally:
        write(a.result, result)
    print(
        json.dumps(dict(status=result["status"], cases=len(result["cases"]))),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--result", type=Path)
    p.add_argument(
        "--prefixes",
        type=int,
        nargs="+",
        default=[0, 1, 7, 8, 15, 16, 31, 64, 112, 512, 1024, 4088],
    )
    a = p.parse_args()
    if a.build:
        build(a)
    if a.run:
        run(a)

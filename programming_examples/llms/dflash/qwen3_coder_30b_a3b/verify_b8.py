# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The target's verify pass over a block of 8 tokens: every layer in one
# dispatch (layers_b8), routed on the device, then the 8-row LM head
# (lm_head_b8). The layers' build is the one with the fewest cache reads that
# cover the block.
import argparse
import contextlib
import functools
import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layers_b8 as LS  # noqa: E402
import lm_head_b8 as LH  # noqa: E402
import model_b8 as M  # noqa: E402
import q4k_layout as QL  # noqa: E402

W, TOK, D, E = M.W, M.TOK, M.D, M.E
FIX = dict(stack_size=4096)


@contextlib.contextmanager
def _cwd(d):
    old = os.getcwd()
    os.makedirs(d, exist_ok=True)
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


SOURCES = {
    "lm_head": (
        "verify_b8.py",
        "lm_head_b8.py",
        "kernels/q8_b8.cc",
        "kernels/q4k_b8.cc",
    ),
    "layers": (
        "verify_b8.py",
        "layers_b8.py",
        "model_b8.py",
        "kernels/layers_b8.cc",
        "kernels/attn_b8.cc",
        "kernels/moe_b8.cc",
        "kernels/q4k_b8.cc",
    ),
}


@functools.lru_cache(maxsize=None)
def _tools():
    """The compilers' identities: a new toolchain is a new build."""
    import hashlib
    import shutil

    from air.tools import resolve_tool

    h = hashlib.sha256()
    peano = os.environ.get("PEANO_INSTALL_DIR")
    for name, tool in (
        ("air-opt", resolve_tool("air-opt")),
        ("aircc", resolve_tool("aircc")),
        ("aiecc", shutil.which("aiecc")),
        ("clang++", peano and Path(peano) / "bin/clang++"),
    ):
        h.update(name.encode())
        if tool and Path(tool).is_file():
            with open(tool, "rb") as f:
                h.update(hashlib.file_digest(f, "sha256").digest())
    return h.hexdigest()


def _weights_stamp(w):
    """What the tiled weights are made from: the layouts and the bundle."""
    import hashlib

    h = hashlib.sha256()
    for f in (
        HERE / "q4k_layout.py",
        HERE / "model_b8.py",
        M.CODER / "lm_head.py",
        M.LLMS / "shared" / "moe" / "lm_head.py",
        M.CODER / "moe_block.py",
        M.CODER / "attn_layer.py",
    ):
        h.update(f.read_bytes())
    st = Path(w.path).stat()
    h.update(f"{w.path} {st.st_size} {st.st_mtime_ns}".encode())
    return h.hexdigest()


def _stamp(module, *key):
    """A build's sources, module text, toolchain and arguments, hashed."""
    import hashlib

    h = hashlib.sha256()
    for f in SOURCES[key[0]]:
        h.update((HERE / f).read_bytes())
    h.update(str(module).encode())
    h.update(_tools().encode())
    h.update(repr(key).encode())
    return h.hexdigest()


def _compile(d, module, kernel, *key, opts=FIX):
    """Build module() in d, unless d holds this build already: (ELF path,
    kernel name, params.txt path, host)."""
    import json

    from air.backend.xrt import XRTBackend, _HostSide

    d = Path(d).resolve()
    m = module()
    st, sf = _stamp(m, *key), d / "stamp.json"
    if (
        sf.is_file()
        and json.loads(sf.read_text()).get("stamp") == st
        and (d / "air.elf").is_file()
    ):
        kname = json.loads(sf.read_text())["kernel"]
    else:
        sf.unlink(missing_ok=True)
        with _cwd(d):
            kernel()
            art = XRTBackend(instance_name=key[0], output_format="elf", **opts).compile(
                m
            )
        kname = art.kernel
        sf.write_text(json.dumps(dict(stamp=st, kernel=kname)))
    return dict(
        elf=str(d / "air.elf"),
        kernel=kname,
        params=str(d / "params.txt"),
        host=_HostSide(str(d / "air_project" / "host")),
    )


def compile_layers(root, nl, reads, cap):
    return _compile(
        Path(root) / f"layers_n{nl}_r{reads}",
        lambda: LS.build_module(nl, reads, cap),
        LS.compile_kernel,
        "layers",
        nl,
        reads,
        cap,
    )


def compile_head(root):
    return _compile(
        Path(root) / "lm_head",
        LH.build_module,
        LH.compile_kernel,
        "lm_head",
        opts=LH.OPTS,
    )


class VerifyB8:
    """The device buffers of the verify pass and its two dispatches."""

    def __init__(self, w, root, cap, reads, layers=None, log=print, tiled=None):
        """The weights and the cache are one buffer each, a layer's part a
        view of it; every read bucket is built up front. tiled: the cache of
        the weights in the device's layout, default <root>/weights_tiled."""
        import pyxrt as xrt

        self.xrt, self.w, self.root, self.cap = xrt, w, Path(root), cap
        self.layers = list(range(W.N_LAYERS if layers is None else layers))
        nl = len(self.layers)
        self.dev = xrt.device(0)
        from air_examples.llms.shared.fused_prefill import hostops as H

        os.makedirs(root, exist_ok=True)
        H.build(str(root), 4)
        H.load(str(root))
        self.reads = sorted(reads)
        self.ctx, self.runs, self.pads = {}, {}, {}
        # host work to do while a dispatch runs
        self.work = []
        t0 = time.time()
        cache = Path(tiled) if tiled else self.root / "weights_tiled"
        cache.mkdir(parents=True, exist_ok=True)
        # tiled weights made from other layouts or another bundle are stale
        ws, wf = _weights_stamp(w), cache / "stamp.txt"
        if not wf.is_file() or wf.read_text() != ws:
            for f in cache.glob("*.npy"):
                f.unlink()
            wf.write_text(ws)

        def cached(name, make):
            f = cache / f"{name}.npy"
            if not f.is_file():
                np.save(cache / f"{name}.tmp.npy", make())
                os.replace(cache / f"{name}.tmp.npy", f)
            return np.load(f, mmap_mode="r")

        sizes = dict(
            we=E * M.G_EXP * M.R * M.BB,
            wd=M.NCOL * M.G_DENSE * M.R * M.BB,
            kv=2 * cap * M.DK * 2,
        )
        big = {k: xrt.ext.bo(self.dev, nl * n) for k, n in sizes.items()}
        self.WE, self.WD, self.KV = big["we"], big["wd"], big["kv"]
        maps = {k: np.frombuffer(b.map(), np.uint8) for k, b in big.items()}
        maps["kv"][:] = 0
        for l in self.layers:
            for k, make in (
                ("wd", lambda: QL.tiled(M.dense_weights(w, l))),
                ("we", lambda: QL.tiled(M.expert_weights(w, l))),
            ):
                n = sizes[k]
                a = cached(f"{'wdf' if k == 'wd' else 'we'}{l}", make)
                maps[k][l * n : (l + 1) * n] = np.asarray(a).view(np.uint8).ravel()
        for b in big.values():
            b.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.kv = {
            l: xrt.bo(self.KV, sizes["kv"], l * sizes["kv"]) for l in self.layers
        }
        self._layer_bos(w)
        log(f"weights of {nl} layers on the device in {time.time() - t0:.0f} s")
        self.xt = self._bo(np.zeros(M.XB, np.uint8))
        self.lm = compile_head(self.root)
        self.wl = self._bo(cached("lm_head", lambda: LH.LM.pack_weights(w)))
        self.yl = self._bo(np.zeros((LH.NCOL, LH.ROUNDS, LH.NROW * LH.YB), bfloat16))
        self.fn = np.asarray(w.final_norm(), np.float32)

    def _layer_bos(self, w):
        """The routers, the norms and h's way in, RW and each layer's K block
        0, the residual out; and every read's build, as one compiled mid-run
        would stall it minutes."""
        nl = len(self.layers)
        f = lambda a: np.asarray(a, np.float32)  # noqa: E731
        self.WR = self._bo(
            np.stack([M.router_blocks(f(w.router(l))) for l in self.layers])
        )
        hin = np.zeros((nl, 2, LS.NSTG * LS.STG), np.float32)
        zh = np.zeros((D // 8, 4, 8), np.float32)
        qkn = []
        for l in self.layers:
            n1, n2 = w.norms(l)
            for c in range(2):
                hin[l, c] = LS.h_way_in(zh, f(n1), f(n2))
            qkn.append(
                np.concatenate(
                    [f(w.vec(l, f"self_attn.{n}_norm.weight")) for n in "qk"]
                )
            )
        self.HIN = self._bo(hin)
        self.qkn = np.stack(qkn).astype(bfloat16)
        self.RWL = self._bo(np.zeros((nl, LS.BLKW), bfloat16))
        self.RWB = self._bo(np.zeros((nl, 16, M.DK), bfloat16))
        self.ZB = self._bo(np.zeros((16, M.DK), bfloat16))
        self.HO = self._bo(np.zeros((nl, 2, LS.HT), np.float32))
        self.builds = {
            r: compile_layers(self.root, nl, r, self.cap) for r in self.reads
        }

    def _bo(self, a):
        xrt = self.xrt
        b = np.ascontiguousarray(a).view(np.uint8).ravel()
        o = xrt.ext.bo(self.dev, b.nbytes)
        np.frombuffer(o.map(), np.uint8)[:] = b
        o.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return o

    def _put(self, bo, a):
        b = np.ascontiguousarray(a).view(np.uint8).ravel()
        np.frombuffer(bo.map(), np.uint8)[: b.nbytes] = b
        bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def _get(self, bo, dt, shape):
        bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        return np.frombuffer(bo.map(), np.uint8)[:n].view(dt).reshape(shape).copy()

    def _run(self, art, bos):
        """The ELF's run, its hardware context opened and bos bound on first
        use."""
        xrt, key = self.xrt, art["elf"]
        if key not in self.runs:
            ctx = xrt.hw_context(self.dev, xrt.elf(key))
            try:
                kern = xrt.ext.kernel(ctx, art["kernel"])
            except RuntimeError:  # a repeated launch's kernel sits in the segment
                kern = xrt.ext.kernel(ctx, art["kernel"].replace("main:", "seg:"))
            run = xrt.run(kern)
            for i, b in enumerate(bos):
                run.set_arg(i, b)
            self.ctx[key] = ctx
            self.runs[key] = run
        return self.runs[key]

    def _go(self, run, what, poll=None):
        from air.backend.xrt import _check_run_state

        run.start()
        while self.work:
            self.work.pop(0)()
        try:
            # poll rather than block: a blocked host wakes late and clocks down
            done = __import__("pyxrt").ert_cmd_state.ERT_CMD_STATE_COMPLETED
            t_end = time.perf_counter() + 30
            while run.state() != done:
                if time.perf_counter() > t_end:
                    raise TimeoutError(f"not done after 30 s, state {run.state()}")
                if poll is not None:
                    poll()
                    while self.work:
                        self.work.pop(0)()
            _check_run_state(run.wait2())
        except Exception as e:
            raise RuntimeError(f"{what}: {e}") from e

    def prefill(self, ids, taps=()):
        """The prompt through the verify pass 8 tokens at a time from position
        0: the cache filled, (the first token, {layer: its outputs [n, D]})
        for the layers in taps. A last short block's padding rows leave
        cache rows the next block overwrites."""
        n = len(ids)
        got = {l: [] for l in taps}
        for i in range(0, n, TOK):
            blk = list(ids[i : i + TOK])
            k = len(blk)
            blk += [blk[-1]] * (TOK - k)
            tp = dict.fromkeys(taps)
            h = self.forward(self.w.embed(blk).astype(np.float32), i, taps=tp)
            for l in taps:
                got[l].append(tp[l][:k])
        first = int(self.head(h)[k - 1])
        return first, {l: np.concatenate(v_) for l, v_ in got.items()}

    def seed(self, kv, n):
        """kv[layer] = (K, V [n, 512]) of the context's n positions."""
        for l in self.layers:
            m = np.frombuffer(self.kv[l].map(), bfloat16).reshape(2, self.cap, M.DK)
            m[0, :n], m[1, :n] = kv[l][0], kv[l][1]
            self.kv[l].sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def head(self, x, norm=True):
        """The LM head's argmax of each of 8 rows; norm: the final norm first."""
        return self.logits(x, norm).argmax(-1)

    def logits(self, x, norm=True):
        """The LM head's logits [8, vocab] of the rows; norm: as head's."""
        import torch

        x = np.asarray(x, np.float32)
        if norm:
            x = M.rmsnorm(x, self.fn)
        self._put(self.xt, M.to_tiles(x.astype(bfloat16)))
        run = self._run(self.lm, [self.xt, self.wl, self.yl])
        self._go(run, "lm head")
        y = self._get(self.yl, bfloat16, (LH.NCOL, LH.ROUNDS, LH.NROW * LH.YB))
        lg = torch.from_numpy(LH.unpack_logits(y.view(np.int16)).copy())
        return (lg.to(torch.int32) << 16).view(torch.float32).numpy()

    def forward(self, h, pos, times=None, taps=None):
        """h [8, D] float32 at positions pos ..; returns the last layer's.
        times: seconds of host and device work, added to; taps: {layer: its
        output}, for the layers it names, each read as soon as it is out."""
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad,
        )

        xrt = self.xrt
        clock = time.perf_counter
        t = dict.fromkeys(("host", "layers"), 0.0) if times is None else times
        t0 = clock()
        nl = len(self.layers)
        r = next((r for r in self.reads if r >= pos + TOK), None)
        if r is None:
            raise ValueError(f"block at {pos} is past the longest read")
        art = self.builds[r]
        # RW at pos, the same rotation for every layer, each its own norms
        cs = M.rope_table(pos, np.zeros(M.DH), np.zeros(M.DH))[: TOK * M.DH]
        rw = np.zeros((nl, LS.BLKW), bfloat16)
        rw[:, : cs.size] = cs
        rw[:, cs.size : cs.size + 2 * M.DH] = self.qkn
        self._put(self.RWL, rw)
        rb = rw.reshape(nl, 16, 16, 8).transpose(0, 2, 1, 3).reshape(nl, 16, M.DH)
        self._put(self.RWB, np.tile(rb, (1, 1, M.NKV)))
        # h into the first layer's way in, of both residual cores
        ht = M.to_res(h).reshape(D // 8, TOK, 8)
        hm = np.frombuffer(self.HIN.map(), np.float32).reshape(nl, 2, -1)
        for c in range(2):
            hm[0, c, : LS.HT] = ht[:, 4 * c : 4 * c + 4].ravel()
        nin = 2 * LS.NSTG * LS.STG * 4
        self.HIN.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nin, 0)
        bos = [self.WE, self.WD, self.WR, self.RWL, self.KV, self.RWB, self.ZB]
        run = self._run(art, bos + [self.HIN, self.HO])
        if art["elf"] not in self.pads:
            self.pads[art["elf"]] = ParameterScratchpad(run, art["params"])
        sp = self.pads[art["elf"]]
        for name, v in art["host"].parameters(art["params"], {9: pos}):
            sp.write(name, np.int32(v))
        sp.sync()
        ho = np.frombuffer(self.HO.map(), np.float32).reshape(nl, 2, D // 8, 4, 8)
        per = 2 * LS.HT * 4
        sync = self.HO.sync
        dev, host = (
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
        )

        def h_after(l):
            sync(dev, per, l * per)
            return (
                np.concatenate([ho[l, 0], ho[l, 1]], axis=1)
                .transpose(1, 0, 2)
                .reshape(TOK, D)
                .copy()
            )

        # each tapped layer's last 64 bytes per core start as NaN, so a poll
        # sees when the layer is out
        tail = 64
        pending = sorted(taps) if taps else []
        for l in pending:
            for c in range(2):
                ho[l, c, -1, 2:, :] = np.nan
                sync(host, tail, l * per + (c + 1) * per // 2 - tail)

        def written(l):
            for c in range(2):
                sync(dev, tail, l * per + (c + 1) * per // 2 - tail)
            return not np.isnan(ho[l, :, -1, -1, -1]).any()

        # a cache sync per spin slows the device: look every half millisecond
        last = [0.0]

        def poll():
            now = clock()
            if pending and now - last[0] > 5e-4:
                last[0] = now
                if written(pending[0]):
                    l = pending.pop(0)
                    taps[l] = h_after(l)

        t1 = clock()
        self._go(run, f"{nl} layers at {pos}", poll=poll)
        t2 = clock()
        for l in pending:
            taps[l] = h_after(l)
        out = h_after(nl - 1)
        t["host"] += (t1 - t0) + (clock() - t2)
        t["layers"] += t2 - t1
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="def quicksort(arr):\n    ")
    ap.add_argument("--layers", type=int, default=W.N_LAYERS, help="the first n")
    ap.add_argument("--reads", default="256")
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--root", default="builds")
    ap.add_argument("--tiled", help="the tiled weights' cache, default the target's")
    ap.add_argument("--bench", type=int, default=0, help="then time n passes")
    ap.add_argument(
        "--compile-only", action="store_true", help="build every read and the head"
    )
    a = ap.parse_args()
    reads = [int(r) for r in a.reads.split(",")]
    if a.compile_only:
        for r in reads:
            compile_layers(a.root, a.layers, r, a.cap)
        compile_head(a.root)
        return 0
    import torch

    sys.path.insert(0, str(M.CODER))
    import reference as RF

    torch.set_num_threads(12)
    w = W.Qwen3CoderQ4k()
    tok = w.tokenizer()
    ids = tok.encode(a.prompt).ids
    n = len(ids)
    layers = list(range(a.layers))
    full = a.layers == W.N_LAYERS
    ref = RF.Reference(w, layers)
    # the context, then 8 tokens of the reference's own greedy continuation;
    # row i of the verify predicts token i + 1
    t0 = time.time()
    xr = ref.forward(ids, lm=False)
    kv = {
        l: tuple(
            np.asarray(c.reshape(n, -1), np.float32).astype(bfloat16) for c in ref.kv[l]
        )
        for l in layers
    }
    fn = w.final_norm()
    lm = w.lm_head if full else None

    def head(x):
        return M.rmsnorm(np.asarray(x, np.float32), fn) @ lm.T

    nxt = int(head(xr[-1:])[0].argmax()) if full else None
    block, rows = [], []
    for i in range(TOK):
        tk = nxt if full else ids[i % n]
        block.append(tk)
        xi = np.asarray(ref.forward([tk], lm=False))
        rows.append(xi[-1])
        if full:
            nxt = int(head(xi[-1:])[0].argmax())
    want = np.stack(rows)
    print(f"reference: {time.time() - t0:.0f} s, block {block}")

    import model_paths

    tiled = a.tiled or model_paths.default_target_cache()
    v = VerifyB8(w, a.root, a.cap, reads, a.layers, tiled=tiled)
    v.seed(kv, n)
    h0 = w.embed(block).astype(np.float32)
    t0 = time.time()
    got = v.forward(h0, n)
    print(f"verify: {time.time() - t0:.2f} s")
    coss = []
    for i in range(TOK):
        g, r_ = got[i].astype(np.float64), want[i].astype(np.float64)
        coss.append(g @ r_ / np.linalg.norm(g) / np.linalg.norm(r_))
        print(f"row {i}: hidden cos {coss[-1]:.5f}")
    # the first layers' hidden states; all 48 are checked by their argmax
    ok = bool(all(c > 0.995 for c in coss))
    if full:
        lg, lw = head(got), head(want)
        am, aw = lg.argmax(-1), lw.argmax(-1)
        nx = block[1:] + [nxt]
        print(f"device {am.tolist()}\nref    {aw.tolist()}\nnext   {nx}")
        ok = bool((am == np.asarray(nx)).all())
    if a.bench:
        times = dict.fromkeys(("host", "layers"), 0.0)
        t0 = time.time()
        diffs = []
        for _ in range(a.bench):
            g2 = v.forward(h0, n, times=times)
            diffs.append(float(np.abs(g2 - got).max()))
        tot = (time.time() - t0) / a.bench * 1e3
        parts = ", ".join(f"{k} {v_ / a.bench * 1e3:.1f}" for k, v_ in times.items())
        print(f"[bench] verify pass {tot:.1f} ms at P={n}: {parts} ms")
        print(f"repeats differ from the first pass by up to {max(diffs):.3g}")
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

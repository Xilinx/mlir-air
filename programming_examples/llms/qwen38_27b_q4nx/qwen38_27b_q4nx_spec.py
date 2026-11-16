# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.8-27B greedy decode, self-speculative with the model's MTP head
# (k = 1): each step verifies the pending token a and the head's draft d in
# one two-token pass (qwen38_27b_q4nx_verify.py), the LM head on both rows
# (lm_head.py, rows=2). If the model's next token after a is d, both are kept
# and the token after d comes free; otherwise only a is, and the model's own
# token replaces d. The head (mtp_draft.py) then runs on each kept position
# (the main model's output there and the token after it), and the LM head on
# its last output with the head's norm drafts the next d. The draft's LM head
# may take the first rows of the vocabulary only (--draft-vocab): a draft
# changes how often a step keeps two tokens, never which tokens.
#
# The tokens are the plain decode's: --check runs it after and compares.
import argparse
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import filelock
import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kernel_tests"))
sys.path.insert(0, str(HERE.parents[1] / "fused_decode"))
# llms/ is imported as the air_examples package, not through sys.path
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

import gdn_layer as G  # noqa: E402
import lm_head as LM  # noqa: E402
import mtp_draft as MT  # noqa: E402
import qwen38_27b_q4nx_verify as V  # noqa: E402
import qwen38_27b_q4nx_weights as W  # noqa: E402

D, PW = W.D, V.PER_WAVE
WAVES = W.N_LAYERS // PW


def top2(lg):
    """(the two highest logits' tokens, their values)."""
    i = np.argsort(-lg)[:2]
    return [int(t) for t in i], [float(lg[t]) for t in i]


def compile_spec(maxl, dv=W.VOCAB):
    """The verify, MTP and two LM head ELFs, each in a directory of its own
    here (params.txt is per build)."""
    here, arts = os.getcwd(), {}
    for name, fn in (
        ("ver", lambda: V.compile_all(maxl)),
        ("mtp", lambda: MT.compile_mtp(maxl)),
        ("lm1", lambda: (LM.compile_lm_head(1, dv), None)),
        ("lm2", lambda: (LM.compile_lm_head(2), None)),
    ):
        os.makedirs(name, exist_ok=True)
        os.chdir(name)
        art, params = fn()
        art.output_binary = os.path.abspath(art.output_binary)
        arts[name] = (art, params)
        os.chdir(here)
    return arts


class Spec:
    """The four ELFs loaded, the LM head's weights shared by both of its."""

    def __init__(self, arts, w, maxl, dv=W.VOCAB):
        import pyxrt as xrt

        self.xrt, self.w, self.maxl, self.dv = xrt, w, maxl, dv
        self.v = V.Q38Verifier(*arts["ver"], w, list(range(W.N_LAYERS)), WAVES, maxl)
        self.m = MT.MtpDraft(*arts["mtp"], w, maxl)
        dev = xrt.device(0)

        def bo(t):
            b = np.ascontiguousarray(t).view(np.uint8).ravel()
            o = xrt.ext.bo(dev, b.nbytes)
            np.frombuffer(o.map(), np.uint8)[:] = b
            o.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            return o

        wl = bo(LM.pack_weights(w))
        wl1 = wl if dv == W.VOCAB else bo(LM.pack_weights(w, dv))
        fn = bo(np.asarray(w.final_norm()).astype(bfloat16))
        nf = bo(np.asarray(self.m.norms[2]).astype(bfloat16))
        v = self.v
        # the verify's last-layer rows [2, D]
        xs = xrt.bo(v.bos[v.X], 2 * D * 2, (WAVES * PW + PW - 1) * 2 * D * 2)
        self.y1 = bo(np.zeros((LM.NCOL, LM.rounds(dv), LM.NROW * LM.RB), bfloat16))
        self.y2 = bo(np.zeros((LM.NCOL, LM.ROUNDS, LM.NROW * 2 * LM.RB), bfloat16))
        self.lm = {}
        for name, args in (
            ("lm1", [self.m.bos[self.m.OUT], nf, wl1, self.y1]),
            ("lm2", [xs, fn, wl, self.y2]),
        ):
            art = arts[name][0]
            ctx = xrt.hw_context(dev, xrt.elf(art.output_binary))
            run = xrt.run(xrt.ext.kernel(ctx, art.kernel))
            for i, b in enumerate(args):
                run.set_arg(i, b)
            self.lm[name] = (ctx, run, args)
        self.t = {k: 0.0 for k in ("verify", "lm2", "mtp", "lm1")}

    def _lm(self, name, rows, vocab=W.VOCAB):
        xrt = self.xrt
        _, run, args = self.lm[name]
        run.start()
        st = run.wait(120000)
        if st != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"{name} ended in {st}")
        args[-1].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        y = np.frombuffer(args[-1].map(), bfloat16)
        return LM.unpack_logits(y, rows, vocab).astype(np.float32)

    def _timed(self, key, fn, *a):
        t0 = time.perf_counter()
        r = fn(*a)
        self.t[key] += time.perf_counter() - t0
        return r

    def seed(self, pf):
        """A prefill's state as the verify's inputs: each attention layer's K
        and V rows, each Gated DeltaNet layer's conv cache and state."""
        v, xrt = self.v, self.xrt
        n, nv = pf.n, len(v.conv_w)
        kv = v.view(v.KV, bfloat16, (WAVES, 2, -1, V.DK))
        cw = v.view(v.CWI, bfloat16, (nv, V.NCH, 7 * V.HD))
        s = v.view(v.SI, np.float32, (nv, V.NV, V.R, V.HD * 32))
        g = 0
        for j in range(len(pf.layers)):
            if j in pf.kv:
                for h in range(2):
                    kv[j // PW, h, :n] = np.concatenate(pf.kv[j][h])
            else:
                cw[g] = G.conv_cache(pf.conv[j], v.conv_w[g])
                s[g] = G.state_chunks(pf.state[j])
                g += 1
        for i in (v.KV, v.CWI, v.SI):
            v.bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def head(self, h):
        """The model's next token after its last-layer row h."""
        v, xrt = self.v, self.xrt
        v.view(v.X, bfloat16, (WAVES + 1, PW, 2, D))[WAVES, PW - 1] = h
        v.bos[v.X].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        return int(self._lm("lm2", 2)[0].argmax())

    def verify(self, a, d, L):
        """The two tokens at L - 2, L - 1: the last layer's rows [2, D] and
        the model's next token after each."""
        emb = np.asarray(self.w.embed([a, d])).astype(bfloat16)
        xs = self._timed("verify", self.v.step, emb, L)
        lg = self._timed("lm2", self._lm, "lm2", 2)
        self.rows = [top2(r) for r in lg]
        return xs[WAVES, PW - 1], [int(t) for t in lg.argmax(-1)]

    def mtp(self, h, tok, L, draft=True):
        """The head at position L - 1 (the main model's output h there, the
        token after it); with draft, the token it predicts after tok."""
        self._timed("mtp", self.m.step, h, tok, L)
        if draft:
            return int(self._timed("lm1", self._lm, "lm1", 1, self.dv).argmax())

    def prompt_mtp(self, ids, hs, first):
        """The head over the prompt, whose last-layer rows are hs [n, D] (row
        i with the token after it); the draft after the first generated
        token."""
        n = len(ids)
        for i in range(n - 1):
            self.mtp(hs[i], ids[i + 1], i + 1, draft=False)
        return self.mtp(hs[n - 1], first, n)

    def generate(self, ids, gen, first, d):
        """gen tokens after the prompt ids: the first generated token and its
        draft known (the prompt's state seeded, prompt_mtp run)."""
        n = len(ids)
        out, a, p = [first], first, n  # a is at position p
        # each generated token's top two logits, for --check
        self.tops = [None]
        steps = acc = 0
        while len(out) < gen:
            h, (b0, b1) = self.verify(a, d, p + 2)
            steps += 1
            if b0 == d:
                acc += 1
                self.v.commit(True)
                out += [d, b1]
                self.tops += self.rows
                self.mtp(h[0], d, p + 1, draft=False)
                d, a, p = self.mtp(h[1], b1, p + 2), b1, p + 2
            else:
                self.v.commit(False)
                out.append(b0)
                self.tops.append(self.rows[0])
                d, a, p = self.mtp(h[0], b0, p + 1), b0, p + 1
        return out[:gen], steps, acc

    def prompt_by_pairs(self, ids):
        """No prefill: the prompt through the verify two tokens at a time (an
        odd one last with a placeholder kept out); its last-layer rows and the
        first generated token."""
        hs, n = [], len(ids)
        for p in range(0, n, 2):
            pair = ids[p : p + 2]
            both = len(pair) == 2
            h, nxt = self.verify(pair[0], pair[1] if both else 0, p + 2)
            self.v.commit(both)
            hs += [h[0], h[1]] if both else [h[0]]
        return np.stack(hs), nxt[1] if n % 2 == 0 else nxt[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France")
    ap.add_argument("--prompt-len", type=int, default=0)
    ap.add_argument(
        "--fused-dir", type=Path, default=os.environ.get("FUSED_PREFILL_DIR")
    )
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--maxl", type=int, default=256)
    ap.add_argument("--check", action="store_true", help="compare to the plain decode")
    ap.add_argument(
        "--queue", action="store_true", help="submit a step's wave runs at once"
    )
    ap.add_argument(
        "--draft-vocab", type=int, default=W.VOCAB, help="the draft's LM head rows"
    )
    a = ap.parse_args()
    w = W.Qwen38Q4nx()
    tok = w.tokenizer()
    ids = tok.encode(a.prompt).ids
    if a.prompt_len:
        ids = (ids * (a.prompt_len // len(ids) + 1))[: a.prompt_len]
    assert len(ids) + a.gen + 2 <= a.maxl
    fused = a.fused_dir is not None and (a.fused_dir / "manifest.json").is_file()
    if fused:
        fused = (
            len(ids)
            <= json.loads((a.fused_dir / "manifest.json").read_text())["max_len"]
        )
    arts = compile_spec(a.maxl, a.draft_vocab)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        t0 = time.time()
        sp = Spec(arts, w, a.maxl, a.draft_vocab)
        sp.v.queue = a.queue
        print(f"weights loaded in {time.time() - t0:.0f}s")
        t0 = time.perf_counter()  # the prompt by pairs
        if fused:
            import prefill as P

            pf = P.Qwen38Prefill(a.fused_dir, w, list(range(W.N_LAYERS)))
            t0 = time.perf_counter()
            hs = pf.prefill(ids).astype(bfloat16)
            sp.seed(pf)
            first = sp.head(hs[-1])
        else:
            hs, first = sp.prompt_by_pairs(ids)
        # the head over the prompt drafts the second token
        d = sp.prompt_mtp(ids, hs, first)
        ttft = time.perf_counter() - t0
        sp.t = {k: 0.0 for k in sp.t}
        t0 = time.perf_counter()
        out, steps, acc = sp.generate(ids, a.gen, first, d)
        t_gen = time.perf_counter() - t0
        print(repr(a.prompt), "->", repr(tok.decode(out)))
        print(f"tokens {out}")
        print(
            f"TTFT {ttft:.3f}s; {len(out)} tokens in {t_gen:.3f}s:"
            f" {(len(out) - 1) / t_gen:.2f} tok/s; {steps} verify steps,"
            f" {acc} drafts accepted ({acc / max(steps, 1):.1%})"
        )
        print(
            "time per step: "
            + ", ".join(
                f"{k} {v / max(steps, 1) * 1e3:.1f} ms" for k, v in sp.t.items()
            )
        )
        sp_tops = sp.tops
        del sp
        if a.check:
            import qwen38_27b_q4nx_decode as DEC

            os.makedirs("dec", exist_ok=True)
            os.chdir("dec")
            darts = DEC.compile_all(WAVES, a.maxl)
            os.chdir("..")
            dec = DEC.load(w, a.maxl, arts=darts)
            emb = lambda t: np.asarray(w.embed([t])[0]).astype(bfloat16)  # noqa
            # the decode from the same start as the speculative one: the
            # prefill's state, or the prompt a token at a time
            n, toks, tops = len(ids), list(ids), []
            if fused:
                pf.prefill(ids)
                dec.seed(pf)
                toks.append(int(np.argmax(dec.head(hs[-1], n))))
                del pf
            for i in range(n if fused else 0, n + len(out) - 1):
                lg = dec.step(emb(toks[i]), i + 1)
                if i + 1 >= n:
                    toks.append(int(np.argmax(lg)))
                    tops.append(top2(lg))
            ref = toks[n:]
            print(f"plain decode {ref}")
            if ref == out:
                print("PASS!")
                return 0
            k = next(i for i in range(len(out)) if ref[i] != out[i])
            # a near-tie both ways: the decode's and the verify's top two the
            # same tokens, a gap of a few bf16 steps
            (dt, dv), (st, sv) = tops[k - 1], sp_tops[k]
            print(
                f"first difference at token {k}: decode {ref[k]}, speculative"
                f" {out[k]}; decode's top two {dt} gap {dv[0] - dv[1]:.3f},"
                f" verify's {st} gap {sv[0] - sv[1]:.3f}"
            )
            tie = set(dt) == set(st) and max(dv[0] - dv[1], sv[0] - sv[1]) < 0.25
            print(
                "PASS (a near-tie)"
                if tie
                else "failed: the tokens differ from the decode's."
            )
            return 0 if tie else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

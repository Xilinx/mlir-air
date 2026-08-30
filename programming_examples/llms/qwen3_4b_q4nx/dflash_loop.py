#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""DFlash speculative decoding end to end on NPU2: BOTH models on the array.

One round of the loop, and where each piece already gates:

    1. TARGET VERIFY, batch B, WITH HIDDEN TAPS      taps_b8_L<N> template
       One dispatch of the block returns B next-token distributions AND every
       layer boundary of every one of the B positions. The distributions decide
       what is accepted (dflash_verify_gate.py: argmax 8/8 against eight
       batch-1 steps); five of the boundaries -- TAP_SLOTS, i.e.
       `hidden_states[layer_id + 1]` for the drafter's target_layer_ids -- are
       the next round's drafter input.
    2. PRE-PASS, as WAVES of step 1's own program       dflash_prepass_waves.py
       taps -> fc -> hidden_norm -> per-layer k/v_proj -> k_norm -> RoPE, for
       the positions the target just committed. Layer-invariant, so it runs
       once per block and not once per layer (dflash_draft_decomp.py).
       It is SPLIT ACROSS THE ROUND, because the chain forces it: fc's 25
       sub-waves ride step 1's tail, over tap slots that pass has just written;
       the host sums them, norms, and gets target_hidden; the 20 context-K/V
       waves are a projection OF target_hidden, so they take their own 2.5 ms
       dispatch here, before step 3. Both against ONE xclbin -- UNI_WAVE_LO/HI
       restrict only the launch loop, so a second instruction stream costs no
       PDI switch.
    3. DRAFTER, 5 layers, bidirectional, batch B     draft_b8_L<N> template
       Its context K/V is what the pre-pass produced; its own block is
       [committed token, mask, mask, ...] and every query sees the whole block.
       Slot j's logits ARE slot j's prediction -- the drafter is not
       autoregressive within a block (_dflash_upstream/model.py:243-262).
    4. ACCEPT                                        plain host arithmetic
       Greedy: slot j+1 survives if it equals the target's argmax at slot j,
       up to the first mismatch; then one bonus token from the target. Neither
       cache needs an explicit rollback -- both engines append in place, and
       the next dispatch's own writes cover every rejected slot before anything
       reads it.

WHAT THIS MEASURES, AND WHAT IT DOES NOT. The number it exists to produce is
the ACCEPTANCE LENGTH with the DEVICE drafter in the loop. Section 3.1 of
docs/DFlashFeasibility.md priced block 8 at 1.24x using a bf16 drafter's
acceptance distribution; the device drafter is int4 pre-pass + q4k layers and
agrees with the bf16 one on only 4 of 7 slots on the one block measured so far
(section 3.7), so 1.24x has been an upper bound with nothing under it.

It is now also a tok/s harness, which it was not: there used to be a third
device program here and three do not fit on the array at once, so the pre-pass
ELF was loaded and unloaded around every block -- 36.5 ms a block that no
shipping design would pay. The pre-pass is waves of the target's own xclbin now,
so there are two programs and no reload.

    python3 dflash_loop.py --n-tokens 32
    python3 dflash_loop.py --n-tokens 32 --no-spec   # the same tokens, block 1
"""

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# `dflash_phase2_device` (imported below for its tap-capturing prefill)
# setdefault()s the decode-artifact directory to its own hidden_taps_test/ and
# then imports the driver, which resolves _DECODE_DIR once. Pin it FIRST, or
# the template scan looks in a directory that holds none of these templates.
os.environ.setdefault("Q4NX_QWEN3_4B_DECODE_DIR", str(_HERE))

# The tap slots the drafter's `fc` consumes, in `extract_context_feature`'s own
# concatenation order. TARGET_LAYER_IDS are HF layer indices and the feature is
# `hidden_states[layer_id + 1]`, which in this engine's X buffer is the slot
# holding the output after that many layers.
from dflash_phase2_replay import TARGET_LAYER_IDS  # noqa: E402

TAP_SLOTS = [lid + 1 for lid in TARGET_LAYER_IDS]  # [2, 10, 18, 26, 34]
MASK_TOKEN_ID = 151669


def _taps_decoder(model, max_L, batch, stack, prefix="taps_b8_L", waves=None):
    """The target, built with DECODE_HIDDEN_TAPS so a verify dispatch also
    returns every layer boundary.

    `X_SLOTS` is a builder constant (`(UNI_DEC + 1) if HIDDEN_TAPS else 1`) and
    `FusedDecoder` already sizes the X BO as `X_SLOTS * K * batch`, so nothing
    is reallocated here -- only read back. Slot s of token t lives at
    `(s*B + t)*K`: a slot is one whole BLOCK of hidden states, because a block
    is what a layer consumes and produces.

    DECODE_MASK_BIDIR is passed explicitly OFF. `FusedDecoder` updates the
    environment rather than replacing it, so a drafter built earlier in the
    same process would otherwise leave the bidirectional bit set and the verify
    pass would let every draft token see the ones after it.
    """
    import numpy as np

    import qwen3_4b_q4nx_inference as INF

    class TapsFusedDecoder(INF.FusedDecoder):
        def dispatch(self, tok, p):
            lg = super().dispatch(tok, p)
            FROM = self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            self.x_bo.sync(FROM, self.nx * 2, 0)
            x = np.frombuffer(self.x_bo.map(), dtype=self.bf16, count=self.nx)
            x = x.reshape(self.X_SLOTS, self.batch, self.K).astype(np.float32)
            # [B, 5*K] -- one row per block position, ready for `fc`.
            self.last_taps = x[TAP_SLOTS].transpose(1, 0, 2).reshape(self.batch, -1)
            # Slot 0 is the input embedding of each of the B tokens, and the
            # host just wrote it. Checking it back is free and it is the one
            # thing that pins BOTH indices of `(slot, token)`: a transposed
            # readback returns the right size and the wrong five layers, and
            # nothing downstream of `fc` would say so.
            e = np.asarray(self.embed[np.asarray(tok).reshape(-1)], np.float32)
            self.taps_slot0_err = float(np.abs(x[0] - e).max())
            return lg

    # THE PRE-PASS'S WAVES RIDE THIS TEMPLATE. `waves` is the wave table from
    # dflash_prepass_waves; passing it makes the verify dispatch carry fc's 25
    # sub-waves in its tail, over the tap slots its own decode waves have just
    # written, and gives the loop a device program the context-K/V stream can be
    # dispatched against without a PDI switch. The template must have been BUILT
    # with the same table (UNI_WAVE_HI at UNI_DEC + UNI_LM + 25), which the
    # env_extra below is what pins.
    _env = {
        "DECODE_STACK": stack,
        "DECODE_HIDDEN_TAPS": "1",
        "DECODE_MASK_BIDIR": "0",
    }
    _extra_w = None
    if waves is not None:
        import dflash_prepass_waves as P

        _we, _extra_w = P.taps_decoder_args(waves=waves)
        _env.update(_we)
    dec = TapsFusedDecoder(
        model=model,
        max_L=max_L,
        batch=batch,
        template_prefix=prefix,
        env_extra=_env,
        extra_weights=_extra_w,
    )
    # FusedDecoder UPDATES the process environment rather than replacing it, and
    # the DRAFTER is built from that same environment a few lines later. Leaving
    # the wave table set would build a drafter that declares an extra weight BO
    # nobody binds -- so drop the two keys that are the target's alone, for the
    # same reason DECODE_MASK_BIDIR is passed explicitly off above.
    for _k in ("DECODE_EXTRA_WAVES", "UNI_WAVE_HI", "UNI_WAVE_LO"):
        os.environ.pop(_k, None)
    return dec


class DFlashLoop:
    def __init__(
        self,
        model,
        prompt,
        block=8,
        max_L=128,
        stack="6080",
        target_prefix="taps_b8_L",
        draft_prefix="draft_b8_L",
        speculate=True,
        verbose=False,
        prepass="waves",
    ):
        import gc

        import numpy as np

        import dflash_phase2_device as PD
        import dflash_prepass_waves as P
        import qwen3_4b_q4nx_weights as gw

        self.np = np
        self.B = int(block)
        self._pre = None

        # ---- prefill (numpy, the same dequantized reference the shipping
        # driver seeds its KV from) -- extended to keep the prompt's taps,
        # which are block 0's context feature.
        #
        # `prompt` may be a LIST OF PROMPTS. Every one is prefilled here,
        # before the decoders exist, because the prefill holds the whole
        # dequantized model and each decoder then allocates a multi-GiB
        # host-only BO -- holding the model across that is enough to die during
        # allocation with no traceback (measured in dflash_verify_gate.py: the
        # process simply stops after the banner). So: prefill everything, drop
        # the model, then open the device once and reuse it per prompt.
        from ml_dtypes import bfloat16

        self.bf16 = bfloat16
        many = bool(prompt) and isinstance(prompt[0], (list, tuple))
        prompts = [list(p) for p in prompt] if many else [list(prompt)]
        print(
            f"[loop] numpy prefill with taps, {len(prompts)} prompt(s)...", flush=True
        )
        qm = gw.Q4nxModel(model)
        self._pf = []
        for pr in prompts:
            pr = [int(t) for t in pr]
            Kc, Vc, logits, ptaps = PD.forward_prompt_with_taps(qm, pr, TAP_SLOTS)
            taps = np.stack([ptaps[s] for s in TAP_SLOTS], axis=1).reshape(len(pr), -1)
            # bf16, because 20 prompts of Kc+Vc in float32 is ~1 GiB of host
            # memory sitting next to two multi-GiB weight BOs. The device KV
            # and the pre-pass input are bf16 anyway, so nothing is lost.
            self._pf.append(
                dict(
                    prompt=pr,
                    Kc=Kc.astype(bfloat16),
                    Vc=Vc.astype(bfloat16),
                    first=int(logits[-1].argmax()),
                    taps=taps.astype(bfloat16),
                )
            )
            del Kc, Vc, logits, ptaps, taps
        del qm
        gc.collect()

        # ---- target, with taps AND the pre-pass's waves in its tail.
        #
        # THE PRE-PASS IS NOT A PROGRAM ANY MORE. It was a third PDI: 82.0 ms of
        # dispatch at 0.46 GB/s plus 36.5 ms of ELF load/unload, per block,
        # because three device programs do not co-reside. It is now 45 extra
        # launch iterations of THIS xclbin's own projection engine -- 6.65 ms
        # total, at 6.5 GB/s -- driven by two instruction streams against one
        # device program (dflash_prepass_waves.WavePrepass).
        self.speculate = bool(speculate)
        self.prepass = None
        _waves = None
        if self.speculate:
            _waves, _ = P.wave_specs(P._load_draft_fd())
        self.target = _taps_decoder(
            model, max_L, self.B, stack, target_prefix, waves=_waves
        )
        if self.speculate:
            self.prepass = (
                P.CpuPrepass(self.target)
                if prepass == "cpu"
                else P.WavePrepass(self.target, verbose=verbose)
            )

        # ---- drafter, bidirectional. DECODE_HIDDEN_TAPS explicitly off: the
        # target set it in this same process's environment.
        import dflash_draft_decoder as DD

        self.DD = DD
        self.drafter = None
        self.maxl = self.target.ATTN_MAXL
        if self.speculate:
            self.drafter = DD.build_draft_decoder(
                model,
                max_L=max_L,
                batch=self.B,
                stack=stack,
                template_prefix=draft_prefix,
                extra_env={"DECODE_HIDDEN_TAPS": "0"},
            )
            self.maxl = min(self.maxl, self.drafter.ATTN_MAXL)
        print(
            f"[loop] target ATTN_MAXL {self.target.ATTN_MAXL}, drafter "
            f"{self.drafter.ATTN_MAXL if self.drafter else '-'}, block {self.B}",
            flush=True,
        )
        self.select(0)

    @property
    def n_prompts(self):
        return len(self._pf)

    def select(self, i):
        """Point the loop at prefilled prompt `i` and re-seed the target's KV.

        The drafter needs no reset here: `select` clears the seeded flag, and
        round 0's `seed_context_kv` zeroes its cache before writing. Only the
        target's KV is prompt-state that must be reloaded.
        """
        pf = self._pf[i]
        self.prompt = pf["prompt"]
        self.P = len(self.prompt)
        self.first = pf["first"]
        self.prompt_taps = pf["taps"]
        assert self.P + self.B <= self.target.ATTN_MAXL, (
            f"prompt {self.P} + block {self.B} exceeds ATTN_MAXL "
            f"{self.target.ATTN_MAXL}; build a longer template pair"
        )
        self.target.seed_kv(
            pf["Kc"].astype(self.np.float32), pf["Vc"].astype(self.np.float32), self.P
        )
        self._sel = i

    def run(self, n_tokens, speculate=None, eos_ids=(151643, 151645), verbose=True):
        """Greedy DFlash decode. Returns (tokens, acceptance_lengths, timings)."""
        np = self.np
        speculate = self.speculate if speculate is None else bool(speculate)
        assert not (speculate and self.drafter is None), "built with speculate=False"
        B, P = self.B, self.P
        DD = self.DD
        max_length = min(P + n_tokens, self.maxl - B)

        out = list(self.prompt) + [self.first]
        # Block 0's context feature is the whole prompt (model.py:219); every
        # later round's is just what the last verify committed.
        ctx_taps = self.prompt_taps
        ctx_pos = np.arange(P)
        start = P
        seeded = False
        # Block 0's target_hidden is the only one no verify pass produced -- the
        # prompt's taps came out of the numpy prefill -- so it takes fc's own
        # instruction stream, chunked over B rows. Every later block's is
        # already on the device when its verify pass ends, because fc's 25
        # sub-waves ran in that pass's tail over the tap slots its own decode
        # waves had just written. That is the whole reason fc rides the verify
        # pass and the context K/V does not: fc's input is what the pass makes.
        ctx_th = self.prepass.th_from_taps(ctx_taps) if speculate else None
        acc_lens, t_target, t_draft, t_pre, stopped = [], 0.0, 0.0, 0.0, False

        while start + 1 < max_length and not stopped:
            blk = [int(out[start])] + [MASK_TOKEN_ID] * (B - 1)

            if speculate:
                t0 = time.time()
                k_ctx, v_ctx = self.prepass.ctxkv(ctx_th, ctx_pos)
                t_pre += time.time() - t0
                if not seeded:
                    DD.seed_context_kv(self.drafter, k_ctx, v_ctx, start)
                    seeded = True
                else:
                    DD.append_context_kv(self.drafter, k_ctx, v_ctx, int(ctx_pos[0]))
                t0 = time.time()
                dl = np.asarray(DD.draft_block(self.drafter, blk, start), np.float32)
                t_draft += time.time() - t0
                # Slot j's own logits are slot j's prediction: the drafter runs
                # the whole block at once and every query sees all of it.
                for j in range(1, B):
                    blk[j] = int(dl[j].argmax())

            t0 = time.time()
            y = np.asarray(self.target.dispatch(blk, start), np.float32)
            t_target += time.time() - t0
            taps = self.target.last_taps  # [B, 5*K]
            if not acc_lens:
                e = self.target.taps_slot0_err
                print(f"[loop] tap slot 0 vs the embeddings written: {e:.3e}")
                assert e == 0.0, "X slot/token indexing disagrees with the write"
            # fc already ran, in the tail of the dispatch above. No dispatch
            # here -- this reads the 25 partials out of the X buffer, subtracts
            # the tap each was added into, and norms the sum.
            if speculate:
                t0 = time.time()
                next_th = self.prepass.th_from_verify(taps)
                t_pre += time.time() - t0
            post = [int(r.argmax()) for r in y]

            # Greedy acceptance: the longest prefix of the drafted slots that
            # the target would itself have produced.
            acc = 0
            if speculate:
                while acc < B - 1 and blk[acc + 1] == post[acc]:
                    acc += 1
            bonus = post[acc]

            produced = acc + 1
            for j in range(produced):
                if start + j < len(out):
                    out[start + j] = blk[j]
                else:
                    out.append(blk[j])
            if start + produced < len(out):
                out[start + produced] = bonus
            else:
                out.append(bonus)
            acc_lens.append(produced)

            if verbose:
                print(
                    f"  block @{start:4d}: drafted {blk[1:]} -> accepted {acc}"
                    f" (+bonus {bonus}) = {produced}",
                    flush=True,
                )
            for j in range(1, produced + 1):
                if out[start + j] in eos_ids:
                    stopped = True
                    out = out[: start + j + 1]
                    break
            # The next round's context feature is the taps of exactly the
            # positions this verify committed -- slots 0..acc, which are
            # positions start..start+acc.
            ctx_taps = taps[:produced]
            ctx_pos = np.arange(start, start + produced)
            if speculate:
                ctx_th = next_th[:produced]
            start += produced

        return (
            out,
            acc_lens,
            {"target": t_target, "draft": t_draft, "prepass": t_pre},
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-tokens", type=int, default=32)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--max-L", type=int, default=128)
    ap.add_argument("--stack", default="6080")
    ap.add_argument("--target-prefix", default="taps_b8_L")
    ap.add_argument("--draft-prefix", default="draft_b8_L")
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default=None, help="text; default is PARIS_PROMPT ids")
    ap.add_argument("--prompt-ids", default=None, help="comma-separated token ids")
    ap.add_argument(
        "--no-spec",
        action="store_true",
        help="run the SAME loop with the drafter switched off, so every block "
        "commits exactly one token. The token stream is then the plain greedy "
        "one and any difference from --spec is speculation's own.",
    )
    ap.add_argument("--prepass", choices=("waves", "cpu"), default="waves")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import numpy as np

    import qwen3_4b_q4nx_inference as INF

    model = args.model or INF.MODEL_DEFAULT
    if args.prompt_ids:
        prompt = [int(x) for x in args.prompt_ids.split(",") if x.strip()]
    elif args.prompt:
        # OUT OF PROCESS. transformers and XRT segfault in one process, and
        # this one is about to open the device.
        import json
        import subprocess

        prompt = json.loads(
            subprocess.run(
                [
                    sys.executable,
                    str(_HERE / "dflash_tokenize.py"),
                    "--encode",
                    args.prompt,
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
    else:
        prompt = list(INF.PARIS_PROMPT)

    loop = DFlashLoop(
        model,
        prompt,
        block=args.block,
        max_L=args.max_L,
        stack=args.stack,
        target_prefix=args.target_prefix,
        draft_prefix=args.draft_prefix,
        speculate=not args.no_spec,
        verbose=args.verbose,
        prepass=args.prepass,
    )
    t0 = time.time()
    toks, acc, t = loop.run(args.n_tokens)
    wall = time.time() - t0

    gen = toks[loop.P :]
    print(f"\n[loop] {len(gen)} tokens: {gen}")
    ref = INF.PARIS_GREEDY
    if not args.prompt:
        n = min(len(ref), len(gen))
        same = sum(a == b for a, b in zip(gen[:n], ref[:n]))
        print(f"[loop] vs PARIS_GREEDY, first {n}: {same}/{n} match")
    if acc:
        print(
            f"[loop] {len(acc)} blocks, acceptance lengths {acc}\n"
            f"[loop] mean tokens per verify dispatch: {np.mean(acc):.3f}"
            f"   (block {args.block}, ceiling {args.block})"
        )
    pp = loop.prepass
    msg = f"[loop] wall {wall:.1f}s | target {t['target']:.2f}s"
    if pp is not None:
        # No ELF (re)load line any more, and that is the point: the pre-pass is
        # waves of the target's own program, so there is no third PDI to swap
        # in. `n_run` counts DISPATCHES, not blocks -- block 0's prompt takes
        # ceil(P/B) of them and every later block takes one.
        msg += (
            f" | draft {t['draft']:.2f}s | pre-pass {t['prepass']:.2f}s over "
            f"{pp.n_run} dispatches ({pp.t_run:.2f}s on the array)"
        )
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

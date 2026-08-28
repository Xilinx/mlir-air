#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Dump one DFlash draft block from the REAL drafter, as the device's oracle.

Runs z-lab/Qwen3-4B-DFlash-b16 itself -- the model code, not a
reimplementation -- over the recorded NPU target state, and writes everything
the on-device drafter has to reproduce for one block:

    taps          [ctx, 12800]   the target hidden-state taps, fc's input
    target_hidden [ctx, 2560]    hidden_norm(fc(taps)), the K/V pre-pass input
    k_ctx, v_ctx  [5, ctx, 1024] per layer, AS THE KV CACHE HOLDS THEM --
                                 K after k_norm AND RoPE, V raw
    k_blk, v_blk  [5, B, 1024]   the block's own rows, which the decode engine
                                 appends itself; kept so a device run can be
                                 checked against them instead of only against
                                 its own logits
    block_ids     [B]            [known token, mask, mask, ...]
    positions     [ctx + B]      ABSOLUTE, and the block's are the last B
    hidden        [B, 2560]      the drafter's output, post final norm
    draft_logits  [B-1, VOCAB]   what the head sees; row i predicts slot i+1
    draft_guess   [B-1]

TORCH AND XRT MUST NOT SHARE A PROCESS (they segfault together), which is the
whole reason this is a dumper and not a function the device gate calls. Run it
once; `dflash_draft_gate.py` consumes the npz with no torch import.

Everything about the block shape is read off the checkpoint rather than
assumed: the mask token id, the target layer ids, and the fact that slot 0 of
the block is a KNOWN token and slots 1.. are the mask token
(_dflash_upstream/model.py:197 fills output_ids with mask_token_id, and the
drafter is trained to predict from target_hidden plus those noise slots).

    python3 dflash_draft_oracle.py                       # block 0, B=8
    python3 dflash_draft_oracle.py --block 2 --block-size 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dflash_phase2_replay import SimpleCropCache, TARGET_LAYER_IDS


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--device-npz", default="dflash_phase2_device.npz")
    ap.add_argument("--out", default="dflash_draft_oracle.npz")
    ap.add_argument("--block", type=int, default=0, help="which block to dump")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument(
        "--embed-source",
        choices=("q4nx", "bf16"),
        default="q4nx",
        help="where the tied embedding/head come from. q4nx matches what the "
        "DEVICE will use, so it is the default: a clean-bf16 embedding in and "
        "a quantized one out would confound the comparison.",
    )
    args = ap.parse_args()

    d = np.load(args.device_npz)
    prompt, gen_ids = list(d["prompt"]), list(d["gen_ids"])
    P = len(prompt)
    prompt_taps, gen_taps = d["prompt_taps"], d["gen_taps"]
    assert list(d["tap_slots"]) == [i + 1 for i in TARGET_LAYER_IDS]

    def token_at(pos):
        return int(prompt[pos]) if pos < P else int(gen_ids[pos - P])

    def taps_at(pos):
        row = prompt_taps[pos] if pos < P else gen_taps[pos - P]
        return row.reshape(-1).astype(np.float32)

    max_tap_pos = P + gen_taps.shape[0] - 1
    max_tok_pos = P + len(gen_ids) - 1
    B = args.block_size

    import torch
    from transformers import AutoModel

    print("[oracle] loading z-lab/Qwen3-4B-DFlash-b16 ...", flush=True)
    model = AutoModel.from_pretrained(
        "z-lab/Qwen3-4B-DFlash-b16", trust_remote_code=True, dtype=torch.bfloat16
    )
    model.eval()
    mask_token_id = model.config.dflash_config["mask_token_id"]
    assert list(model.target_layer_ids) == TARGET_LAYER_IDS, model.target_layer_ids
    n_layers = len(model.layers)

    if args.embed_source == "q4nx":
        import qwen3_4b_q4nx_weights as gw

        qm = gw.Q4nxModel("FastFlowLM/Qwen3-4B-NPU2")
        embed, _fn, lm_head = qm.embed_norm_lmhead()
        embed_table = torch.from_numpy(np.asarray(embed, np.float32)).to(torch.bfloat16)
        lm_head_t = torch.from_numpy(np.asarray(lm_head, np.float32))
    else:
        from transformers import AutoModelForCausalLM

        bf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16)
        embed_table = bf.get_input_embeddings().weight.detach()
        lm_head_t = bf.get_output_embeddings().weight.detach().to(torch.float32)
        del bf

    cache = SimpleCropCache(n_layers)
    start, block_i, dump = P, 0, None
    with torch.no_grad():
        while start <= max_tok_pos and start - 1 <= max_tap_pos:
            if block_i == 0:
                # Block 0's context is the WHOLE PROMPT, not <= B rows
                # (model.py:219). Every later block's is `produced`.
                ctx_positions = list(range(P))
            block_len = min(B, max_tok_pos - start + 1)
            if block_len < 2:
                break

            ctx_rows = np.stack([taps_at(p) for p in ctx_positions], axis=0)
            th_raw = torch.as_tensor(ctx_rows, dtype=torch.bfloat16).unsqueeze(0)
            block_ids = [token_at(start)] + [mask_token_id] * (block_len - 1)
            noise = embed_table[block_ids].unsqueeze(0)
            positions = ctx_positions + list(range(start, start + block_len))
            pos_t = torch.tensor(positions, dtype=torch.long).unsqueeze(0)

            hidden = model(
                position_ids=pos_t,
                noise_embedding=noise,
                target_hidden=th_raw,
                past_key_values=cache,
                use_cache=True,
            )
            logits = hidden[0, 1:, :].to(torch.float32) @ lm_head_t.T
            guess = logits.argmax(dim=-1).tolist()

            if block_i == args.block:
                nc = len(ctx_positions)
                # The cache's LAST nc+block_len rows are what this round added,
                # in position order: the context rows then the block's own. Read
                # them rather than recomputing -- these are the model's own
                # post-k_norm, post-RoPE K and raw V, which is exactly what the
                # device KV cache has to hold.
                k = [cache.k[L][0].float().numpy() for L in range(n_layers)]
                v = [cache.v[L][0].float().numpy() for L in range(n_layers)]

                def flat(a, lo, hi):  # [n_kv, seq, dh] -> [hi-lo, n_kv*dh]
                    return np.ascontiguousarray(
                        a[:, lo:hi, :].transpose(1, 0, 2).reshape(hi - lo, -1)
                    )

                seq = k[0].shape[1]
                assert seq >= nc + block_len, (seq, nc, block_len)
                c0, c1 = seq - nc - block_len, seq - block_len
                dump = dict(
                    block=block_i,
                    start=start,
                    ctx_positions=np.asarray(ctx_positions, np.int64),
                    positions=np.asarray(positions, np.int64),
                    block_ids=np.asarray(block_ids, np.int64),
                    mask_token_id=mask_token_id,
                    taps=ctx_rows.astype(np.float32),
                    target_hidden=model.hidden_norm(model.fc(th_raw))[0]
                    .float()
                    .numpy(),
                    k_ctx=np.stack([flat(k[L], c0, c1) for L in range(n_layers)]),
                    v_ctx=np.stack([flat(v[L], c0, c1) for L in range(n_layers)]),
                    k_blk=np.stack([flat(k[L], c1, seq) for L in range(n_layers)]),
                    v_blk=np.stack([flat(v[L], c1, seq) for L in range(n_layers)]),
                    hidden=hidden[0].float().numpy(),
                    draft_logits=logits.numpy(),
                    draft_guess=np.asarray(guess, np.int64),
                    true_next=np.asarray(
                        [token_at(start + 1 + i) for i in range(block_len - 1)],
                        np.int64,
                    ),
                )

            cache.crop_keep_first(start)
            acc = 0
            for g, t in zip(
                guess, [token_at(start + 1 + i) for i in range(block_len - 1)]
            ):
                if g != t:
                    break
                acc += 1
            print(
                f"[oracle] block {block_i}: start={start} ctx={len(ctx_positions)} "
                f"accepted={acc + 1}/{block_len}",
                flush=True,
            )
            if dump is not None:
                break
            ctx_positions = list(range(start, start + acc + 1))
            start += acc + 1
            block_i += 1

    if dump is None:
        print(f"[oracle] block {args.block} never reached", file=sys.stderr)
        return 1
    np.savez(args.out, **dump)
    print(
        f"[oracle] wrote {args.out}: ctx={dump['taps'].shape[0]}, "
        f"B={len(dump['block_ids'])}, start={dump['start']}, "
        f"guess={dump['draft_guess'].tolist()}, true={dump['true_next'].tolist()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

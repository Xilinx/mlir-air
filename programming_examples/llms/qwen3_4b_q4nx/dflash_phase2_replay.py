#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Phase 2 of the DFlash drafter bring-up (docs/DFlashFeasibility.md, "## 8"):
offline replay half. Runs the REAL z-lab/Qwen3-4B-DFlash-b16 drafter (torch,
CPU) against the real NPU-sourced target state recorded by
dflash_phase2_device.py, and measures the real acceptance rate -- the number
that decides whether on-device batch=16 (and therefore item 11's L1-ceiling
unlock) is worth building at all.

WHY OFFLINE REPLAY IS VALID (not an approximation): the target's greedy
decode is a deterministic function of the token prefix alone, and DFlash's
own lossless guarantee is that the accepted output IS that greedy stream. So
recording one plain greedy continuation up front (dflash_phase2_device.py)
and replaying the drafter against sliding windows of it is exactly equivalent
to interleaved draft/verify: once a block's guesses diverge from that
recording, everything after the first mismatch is discarded by the standard
greedy-prefix acceptance rule anyway, so it never needs ground truth beyond
the true continuation up to that point.

WHY THIS USES A CUSTOM `SimpleCropCache` INSTEAD OF HF's `Cache`: `dflash.py`'s
own `spec_generate` was written against transformers 4.57.3, where
`Cache.crop(N)` means "keep only the first N cached positions". The installed
transformers (5.15) renamed the same method to `crop(tokens_to_remove)` --
"remove N from the end" -- an incompatible, silently-different meaning with
no error raised. Hand-simulating the ORIGINAL "keep first N" semantic with
concrete numbers (see docs/DFlashFeasibility.md section 2) shows the
draft-side cache actually PERSISTS AND GROWS, one entry per accepted
position, forever -- the opposite of an earlier, quicker reading of
`spec_generate`. `SimpleCropCache` below reproduces exactly that old
behavior (plain concatenate-on-update, explicit keep-first-N crop) so the
REAL model code (`Qwen3DFlashDecoderLayer`/`Qwen3DFlashAttention`, called
directly, not reimplemented) can be driven correctly regardless of which
transformers version is installed. Verified bit-exact against the model's
own `past_key_values=None` no-cache path for block 0 (where there is no
cache-history difference to hide a bug in) before trusting it for later
blocks -- an earlier hand-rolled reimplementation of the attention loop
(replaced by this) was NOT bit-exact against that same reference (max abs
diff 1.25) and was discarded rather than debugged further, since reusing the
real model code is strictly safer than re-deriving RoPE/GQA/norm-order by
hand.
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

TARGET_LAYER_IDS = [1, 9, 17, 25, 33]
BLOCK_SIZE = 16


class SimpleCropCache:
    """A `transformers.Cache`-compatible-enough stand-in implementing exactly
    the "keep first N" semantic `dflash.py`'s `spec_generate` was written
    against (transformers 4.57.3's `DynamicCache.crop`), independent of
    whatever the installed transformers version's own `Cache.crop` now means.
    `update()` unconditionally concatenates (plain growth, no sliding-window
    trimming) -- correct for this model, which has no sliding-window layers
    (`config.json`'s `layer_types` are all `full_attention`)."""

    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.k[layer_idx] is None:
            k_full, v_full = key_states, value_states
        else:
            import torch

            k_full = torch.cat([self.k[layer_idx], key_states], dim=2)
            v_full = torch.cat([self.v[layer_idx], value_states], dim=2)
        self.k[layer_idx] = k_full
        self.v[layer_idx] = v_full
        return k_full, v_full

    def crop_keep_first(self, n):
        for i in range(len(self.k)):
            if self.k[i] is not None:
                self.k[i] = self.k[i][:, :, :n, :]
                self.v[i] = self.v[i][:, :, :n, :]

    def get_seq_length(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]


def main():
    device_npz = sys.argv[1] if len(sys.argv) > 1 else "dflash_phase2_device.npz"
    embed_source = "q4nx"
    for a in sys.argv[2:]:
        if a.startswith("--embed-source="):
            embed_source = a.split("=", 1)[1]
    assert embed_source in ("q4nx", "bf16"), embed_source

    d = np.load(device_npz)
    prompt = list(d["prompt"])
    P = len(prompt)
    gen_ids = list(d["gen_ids"])
    gen_taps = d["gen_taps"]  # [n_dispatches, 5, K]
    prompt_taps = d["prompt_taps"]  # [P, 5, K]
    tap_slots = list(d["tap_slots"])
    assert tap_slots == [lid + 1 for lid in TARGET_LAYER_IDS]

    def token_at(pos):
        if pos < P:
            return int(prompt[pos])
        return int(gen_ids[pos - P])

    max_tap_pos = P + gen_taps.shape[0] - 1
    max_tok_pos = P + len(gen_ids) - 1

    def taps_at(pos):
        row = prompt_taps[pos] if pos < P else gen_taps[pos - P]  # [5, K]
        return row.reshape(-1).astype(np.float32)  # [5*K], TAP_SLOTS order

    print("[dflash_phase2_replay] loading real drafter: z-lab/Qwen3-4B-DFlash-b16 ...", flush=True)
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "z-lab/Qwen3-4B-DFlash-b16", trust_remote_code=True, dtype=torch.bfloat16
    )
    model.eval()
    mask_token_id = model.config.dflash_config["mask_token_id"]
    assert list(model.target_layer_ids) == TARGET_LAYER_IDS, model.target_layer_ids

    if embed_source == "q4nx":
        print("[dflash_phase2_replay] loading target's tied embedding (Q4NX-dequantized)...", flush=True)
        import qwen3_4b_q4nx_weights as gw

        qm = gw.Q4nxModel("FastFlowLM/Qwen3-4B-NPU2")
        embed, _final_norm, lm_head = qm.embed_norm_lmhead()  # tied: lm_head is embed
        embed_table = torch.from_numpy(np.asarray(embed, dtype=np.float32)).to(torch.bfloat16)
        lm_head_t = torch.from_numpy(np.asarray(lm_head, dtype=np.float32))  # f32 for the final matmul
    else:
        print("[dflash_phase2_replay] loading target's tied embedding (clean bf16, Qwen/Qwen3-4B)...", flush=True)
        from transformers import AutoModelForCausalLM

        bf16_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16)
        embed_table = bf16_model.get_input_embeddings().weight.detach()  # [VOCAB, D] bf16
        lm_head_t = bf16_model.get_output_embeddings().weight.detach().to(torch.float32)  # [VOCAB, D]
        del bf16_model

    cache = SimpleCropCache(len(model.layers))
    start = P
    accepted_lens = []
    block_i = 0
    with torch.no_grad():
        while start <= max_tok_pos and start - 1 <= max_tap_pos:
            if block_i == 0:
                ctx_positions = list(range(0, P))

            block_len = min(BLOCK_SIZE, max_tok_pos - start + 1)
            if block_len < 2:
                break

            ctx_rows = np.stack([taps_at(p) for p in ctx_positions], axis=0)
            target_hidden_raw = torch.as_tensor(ctx_rows, dtype=torch.bfloat16).unsqueeze(0)
            block_ids = [token_at(start)] + [mask_token_id] * (block_len - 1)
            noise_embedding = embed_table[block_ids].unsqueeze(0)
            position_ids = torch.tensor(
                ctx_positions + list(range(start, start + block_len)), dtype=torch.long
            ).unsqueeze(0)

            hidden = model(
                position_ids=position_ids,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden_raw,
                past_key_values=cache,
                use_cache=True,
            )
            cache.crop_keep_first(start)  # dflash.py's own crop(start), pre-increment

            draft_logits = hidden[0, 1:, :].to(torch.float32) @ lm_head_t.T  # [block_len-1, vocab]
            draft_guess = draft_logits.argmax(dim=-1).tolist()

            true_next = [token_at(start + 1 + i) for i in range(block_len - 1)]
            acc_len = 0
            for g, t in zip(draft_guess, true_next):
                if g == t:
                    acc_len += 1
                else:
                    break
            accepted_lens.append(acc_len + 1)
            print(
                f"[dflash_phase2_replay] block {block_i}: start={start} block_len={block_len} "
                f"ctx_len={len(ctx_positions)} accepted={acc_len + 1}/{block_len} "
                f"draft={draft_guess[:acc_len+2]} true={true_next[:acc_len+2]}",
                flush=True,
            )
            ctx_positions = list(range(start, start + acc_len + 1))
            start += acc_len + 1
            block_i += 1

    accepted_lens = np.array(accepted_lens)
    print(f"\n[dflash_phase2_replay] {len(accepted_lens)} blocks, block_size={BLOCK_SIZE}")
    print(f"[dflash_phase2_replay] accepted lengths: {accepted_lens.tolist()}")
    print(
        f"[dflash_phase2_replay] mean accepted = {accepted_lens.mean():.2f} of {BLOCK_SIZE} "
        f"({100*accepted_lens.mean()/BLOCK_SIZE:.1f}% of lossless ceiling)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

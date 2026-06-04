# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HuggingFace transformers runner — bf16, runs on CPU.

The single bf16 reference for both `make verify` and `make diagnosis`.
Two modes:
  - lite_mode=True  (used by `make verify`): pass output_hidden_states=
    False so HF skips the per-layer hidden-state list internally; only
    logits + top1 are read back.
  - lite_mode=False (used by `make diagnosis`): collect per-layer
    hidden_states. Per HF transformers v5.3 convention, hidden_states is
    a tuple of length n_layers + 1: index 0 is the embedding output;
    indices 1..n_layers-1 are the *raw* outputs of layers 0..n_layers-2;
    index n_layers is the *post-final-norm* version of layer n_layers-1
    (the last layer's raw output is NOT exposed). We therefore expose
    ffn_out for layers 0..n_layers-2 and ALSO surface hidden_states[-1]
    as final_hidden_normed so the orchestrator can pair the L15 cell
    with the NPU's own post-final-norm hidden state.

All intermediates are cast to float32 NumPy before returning since NumPy
has no native bfloat16 and the comparators all operate in F32 space.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from runners._records import PrefillRecord, DecodeStepRecord


class HfRunner:
    name = "hf_bf16"

    def __init__(
        self,
        model_name: str,
        config,
        max_seq: int,
        lite_mode: bool = False,
        model=None,
    ):
        """Build an HF reference runner.

        If `model` is provided (any `torch.nn.Module` returning
        `(logits, hidden_states, past_key_values)`-shaped output), use it
        as-is. Otherwise load via `AutoModelForCausalLM.from_pretrained`.
        Adapters wanting a non-trivial weight setup (e.g. patching AWQ-
        dequant weights into the meta-llama architecture so the verify
        gate isolates NPU drift from quantization error) build the model
        themselves and inject it here.
        """
        self.config = config
        self.max_seq = max_seq
        self.lite_mode = lite_mode
        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            )
        else:
            self.model = model
        self.model.eval()
        self.past_key_values = None
        self._n_layers = config.n_layers
        self._emb_dim = config.emb_dim
        self._n_kv = config.n_kv_heads
        self._head_dim = config.head_dim

    @torch.no_grad()
    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        # Reset KV cache so verify-loop reuse across prompts does not
        # cross-pollinate prompt N's state into prompt N+1's decode.
        self.past_key_values = None
        input_ids = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0)
        out = self.model(
            input_ids,
            output_hidden_states=not self.lite_mode,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[0, -1].cpu().float().numpy()  # (vocab,)
        top1 = int(np.argmax(logits))
        self.past_key_values = out.past_key_values
        if self.lite_mode:
            empty = np.empty((0,), dtype=np.float32)
            return PrefillRecord(
                layer_intermediates=[],
                final_hidden_normed=empty,
                logits_at_pred=logits,
                top1_token=top1,
            )
        hidden_states = out.hidden_states
        layer_intermediates: list[dict[str, np.ndarray]] = []
        for li in range(self._n_layers - 1):
            # .float() upcasts bf16 to f32 — NumPy has no native bf16.
            ffn_out = hidden_states[li + 1][0].cpu().float().numpy()
            layer_intermediates.append({"ffn_out": ffn_out})
        # Last-layer entry intentionally has no ffn_out — the orchestrator
        # uses final_hidden_normed for the L15 probe instead.
        layer_intermediates.append({})
        # hidden_states[-1] is the post-final-norm version of the last
        # layer's output (HF v5.3 convention). Same value the model fed
        # into lm_head. Empirically: for raw last-layer hidden of magnitude
        # ~130, max|raw + final_norm - hs[-1]| ~ 1e-2.
        final_hidden_normed = hidden_states[-1][0].cpu().float().numpy()
        return PrefillRecord(
            layer_intermediates=layer_intermediates,
            final_hidden_normed=final_hidden_normed,
            logits_at_pred=logits,
            top1_token=top1,
        )

    @torch.no_grad()
    def decode_step(self, input_token: int, current_pos: int) -> DecodeStepRecord:
        if self.past_key_values is None:
            raise RuntimeError("decode_step called before prefill")
        input_ids = torch.tensor([[input_token]], dtype=torch.long)
        out = self.model(
            input_ids,
            past_key_values=self.past_key_values,
            output_hidden_states=False,  # decode probes are not collected
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[0, -1].cpu().float().numpy()
        top1 = int(np.argmax(logits))
        self.past_key_values = out.past_key_values
        return DecodeStepRecord(
            lm_head_logits=logits,
            top1_token=top1,
        )

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the `--ref dequant` reference for accuracy.py.

The reference is the HF bf16 checkpoint with every weight the NPU runs in
quantized form replaced by those same values, dequantized. Scored against it,
an NPU model's KL divergence measures the implementation alone; scored against
the plain checkpoint, it also includes what the weight format costs.

Which tensors to replace, and with what, is the adapter's knowledge: a bundle
can carry a quantized tensor the NPU path never reads, or the decode can
requantize a tensor the bundle stores in bf16. Adapters therefore list the
weights explicitly rather than having every quantized tensor in a file swapped
in.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

LM_HEAD = "lm_head.weight"


def dequantized_reference(
    hf_ref_model: str,
    weights: Iterable[tuple[str, Callable[[tuple], np.ndarray]]],
):
    """`hf_ref_model` in bf16 with each named parameter set to `fn(shape)`.

    `weights` yields (HF parameter name, fn taking the parameter's shape and
    returning float values of that shape). Replacing `lm_head.weight` unties it
    from the input embedding first, because the NPU head can be quantized while
    the embedding lookup stays bf16.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        hf_ref_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    params = dict(model.named_parameters())
    n = 0
    with torch.no_grad():
        for name, fn in weights:
            if name == LM_HEAD:
                target = model.get_output_embeddings().weight
            elif name in params:
                target = params[name]
            else:
                # A renamed HF parameter would otherwise stay bf16 silently.
                raise KeyError(f"{hf_ref_model} has no parameter {name}")
            shape = tuple(target.shape)
            values = np.ascontiguousarray(fn(shape), dtype=np.float32)
            if values.shape != shape:
                raise ValueError(f"{name}: got {values.shape}, model has {shape}")
            if name == LM_HEAD:
                replace_head(model, values)
            else:
                # copy_ converts in place, without a bf16 temporary the size
                # of the weight.
                target.copy_(torch.from_numpy(values))
            del values
            n += 1
    print(f"[dequant_reference] replaced {n} weights in {hf_ref_model}")
    return model


def replace_head(model, values: np.ndarray):
    """Set `model`'s LM head to `values`, untied from the input embedding."""
    import torch

    head = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    model.get_output_embeddings().weight = torch.nn.Parameter(head.to(torch.bfloat16))
    model.config.tie_word_embeddings = False

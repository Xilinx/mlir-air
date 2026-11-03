# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the shared fixed-cohort saved-prefill decode study for this model."""

from functools import partial

from dflash_model import MODEL
from common.benchmark_runner import main
from common.runtime import load_pair
from benchmark_data import verify_inputs
from model_paths import load_tokenizer
from prepare_inputs import load_prefill
from provenance import runtime_identity
from baseline import load_baseline

if __name__ == "__main__":
    main(
        verify_inputs=verify_inputs,
        load_tokenizer=load_tokenizer,
        load_prefill=load_prefill,
        runtime_identity=runtime_identity,
        load_pair=partial(load_pair, MODEL),
        load_baseline=load_baseline,
        eos_ids=MODEL.eos_ids,
    )

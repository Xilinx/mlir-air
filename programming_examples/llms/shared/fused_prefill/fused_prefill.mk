# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The fused prefill targets of a dense q4nx model (dense_main.py). A model's
# Makefile sets FUSED_MODEL (its name in models.py) and MODEL_SOURCE, then
# includes this.

FUSED_MAIN := $(dir $(lastword $(MAKEFILE_LIST)))dense_main.py
# The build (kernels + every op's insts), under the invoking directory. The
# prefill is chunked; FUSED_MAX_LEN, when set, is the longest prompt the build
# takes (models.py's max_len otherwise).
FUSED_DIR ?= $(CURDIR)/_fused_prefill
override FUSED_DIR := $(abspath $(FUSED_DIR))
FUSED_JOBS ?= 6
# The drivers and verify adapters prefill on the build in FUSED_PREFILL_DIR when
# it has one, else on the per-op prefill; FUSED_PREFILL_DIR= selects the latter.
FUSED_PREFILL_DIR ?= $(FUSED_DIR)
export FUSED_PREFILL_DIR

.PHONY: compile-fused-prefill fused-prefill fused-prefill-long fused-prefill-8k \
  profile-fused-prefill

## The fused prefill: every op of a 128-token chunk on one configured device,
## int4 weights dequantized on the cores.
compile-fused-prefill:
	python3 $(FUSED_MAIN) build $(FUSED_MODEL) $(FUSED_DIR) -j $(FUSED_JOBS) \
	  $(if $(FUSED_MAX_LEN),--max-len $(FUSED_MAX_LEN))

## One chunk, gated against the fp32 reference by logit cosine and first token.
fused-prefill:
	python3 $(FUSED_MAIN) verify $(FUSED_MODEL) $(FUSED_DIR) --model $(MODEL_SOURCE)

## Three chunks of README text, same gate.
fused-prefill-long:
	python3 $(FUSED_MAIN) verify $(FUSED_MODEL) $(FUSED_DIR) --model $(MODEL_SOURCE) \
	  --text-file $(FUSED_TEXT) --text-tokens 280 --repeat 1

## FUSED_LONG_TOKENS tokens of FUSED_LONG_TEXT, repeated to length, same gate.
## The build must take the prompt (FUSED_MAX_LEN).
FUSED_LONG_TOKENS ?= 8000
fused-prefill-8k:
	python3 $(FUSED_MAIN) verify $(FUSED_MODEL) $(FUSED_DIR) --model $(MODEL_SOURCE) \
	  --text-file $(FUSED_LONG_TEXT) --text-tokens $(FUSED_LONG_TOKENS) --repeat 1

## Prefill time at each of FUSED_LENGTHS on synthetic prompts, no reference.
## The build must take the longest (FUSED_MAX_LEN).
FUSED_LENGTHS ?= 512,2048
profile-fused-prefill:
	python3 $(FUSED_MAIN) bench $(FUSED_MODEL) $(FUSED_DIR) --model $(MODEL_SOURCE) \
	  --lengths $(FUSED_LENGTHS)

# Next to FUSED_MAIN: MAKEFILE_LIST's last entry is whatever the model's
# Makefile included last by the time a recipe expands these.
FUSED_TEXT ?= $(dir $(FUSED_MAIN))../../README.md
# A frozen copy of README.md: editing the README would change the 8k prompt.
FUSED_LONG_TEXT ?= $(dir $(FUSED_MAIN))gate_8k.txt

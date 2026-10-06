# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The fused prefill targets of a dense q4nx model (dense_main.py). A model's
# Makefile sets FUSED_MODEL (its name in models.py) and MODEL_SOURCE, then
# includes this.

FUSED_MAIN := $(dir $(lastword $(MAKEFILE_LIST)))dense_main.py
# The build (kernels + every op's insts), under the invoking directory. The
# prefill is chunked, so it has no CTX.
FUSED_DIR ?= $(CURDIR)/_fused_prefill
override FUSED_DIR := $(abspath $(FUSED_DIR))
FUSED_JOBS ?= 6

.PHONY: compile-fused-prefill fused-prefill fused-prefill-long

## The fused prefill: every op of a 128-token chunk on one configured device,
## int4 weights dequantized on the cores.
compile-fused-prefill:
	python3 $(FUSED_MAIN) build $(FUSED_MODEL) $(FUSED_DIR) -j $(FUSED_JOBS)

## One chunk, gated against the fp32 reference by logit cosine and first token.
fused-prefill:
	python3 $(FUSED_MAIN) verify $(FUSED_MODEL) $(FUSED_DIR) --model $(MODEL_SOURCE)

## Three chunks of README text, same gate.
fused-prefill-long:
	python3 $(FUSED_MAIN) verify $(FUSED_MODEL) $(FUSED_DIR) --model $(MODEL_SOURCE) \
	  --text-file $(FUSED_TEXT) --text-tokens 280 --repeat 1

FUSED_TEXT ?= $(dir $(lastword $(MAKEFILE_LIST)))../../README.md

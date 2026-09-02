# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Gemma4-E2B (text) Q4NX -- CPU reference stage.
#
# This example is IN PROGRESS. Only the CPU reference exists so far; there is no
# NPU decode yet (that is the fused_decode_ple builder). Everything below runs on
# the host with numpy: no NPU, no Peano, no aie-opt, no mlir-air toolchain. All
# you need is the FLM Q4NX bundle and this repo's Python deps.
#
# The default target is `check`, the per-layer gate against FastFlowLM's own
# golden activations. That is deliberate: the end-to-end token test is too blunt
# to trust on its own. Every serious bug found here passed a "looks like English"
# check while being numerically wrong --
#   - a doubled embedding scale made all 35 layers a near-passthrough and still
#     produced fluent text;
#   - the per-layer-embedding input read from the wrong tensor left layer 8 at
#     cos 0.79 while the model still answered some prompts correctly.
# So: gate on `check`, use `gen` as a smoke test, never the other way round.
#
# Quick start:
#   make            # == make check   (per-layer cosine vs FLM's reference)
#   make gen        # greedy generation with the chat template
#   make paris      # fast single-forward smoke test
#
# Expected `check` result on a correct tree (2026-09-02):
#   input_embedding cos 0.9969  (int8 embedding table; this is the floor)
#   all 35 layers   cos 0.986 - 0.9998 teacher-forced, 33 of them >= 0.99
#   free-running    0.9635 at layer_34
#   rms_ratio       ~1.000 on every layer
# A layer dropping to ~0.8 with rms_ratio still ~1.0 means a CONTENT bug (wrong
# tensor feeding a block), not a scaling one -- cosine cannot see scale errors,
# which is why the gate prints both.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

# FLM's Q4NX bundle (weights the port reproduces) and FLM's golden activations.
# Both are inputs to the reference; neither is produced by this repo.
BUNDLE   ?= /home/strixminipc/rocm_fastflowlm/FastFlowLM/models/Gemma4-E2B-IT-NPU2
FLM_REF  ?= /home/strixminipc/rocm_fastflowlm/FastFlowLM/FLM_Xclbin/Gemma4/decoding/references/gemma4_e2b_ref.safetensors
# Full-precision checkpoint, ONLY needed to regenerate sub-layer ground truth.
# Pinned by FLM's own references/gen_gemma4_ref.py as MODEL_DIR.
HF_FP    ?= /home/strixminipc/fastflowlm/models/gemma-4-E2B-it-fp

LAYERS   ?= 35
# Per-layer cosine floor. NOT 0.99: layer 0 inherits the int8 embedding table's
# error (input_embedding cos 0.9969) and lands at 0.986, so 0.99 makes the gate
# fail on a correct tree. 0.98 still catches anything real -- the per-layer
# embedding bug put layer 8 at 0.79, and a doubled embedding scale put layer 2
# at 0.32.
TOL      ?= 0.98
PROMPT   ?= What is the capital of France?
N_GEN    ?= 12

.DEFAULT_GOAL := check
.PHONY: check gen paris sublayer-ref help

## Per-layer gate vs FLM's golden activations. THE correctness gate. ~4 min.
check:
	cd $(srcdir) && $(PYTHON) check_vs_flm_reference.py \
		--bundle $(BUNDLE) --ref $(FLM_REF) --layers $(LAYERS) --tol $(TOL)

## Greedy generation through the chat template. Smoke test, NOT a gate.
## Slow by construction: no KV reuse across tokens, so each token re-runs the
## whole prompt (~35 s/token at 16 tokens of context).
gen:
	cd $(srcdir) && $(PYTHON) run_reference.py --bundle $(BUNDLE) gen \
		--prompt "$(PROMPT)" --n-tokens $(N_GEN)

## Fast smoke test: one forward, expect ' Paris' as top-1.
paris:
	cd $(srcdir) && $(PYTHON) run_reference.py --bundle $(BUNDLE) paris

## Regenerate sub-layer ground truth from the full-precision checkpoint.
## Only needed when bisecting INSIDE a layer -- FLM's shipped reference has one
## tensor per layer, which cannot tell attention from FFN from the PLE tail.
## Loads a 10.2 GB checkpoint in float32; needs ~25 GB RAM and a few minutes.
sublayer-ref:
	cd $(srcdir) && $(PYTHON) dump_sublayer_reference.py \
		--layers 7,8,9 --out sublayer_ref.npz

help:
	@echo "Gemma4-E2B Q4NX (CPU reference stage; no NPU path yet)"
	@echo ""
	@echo "  make / make check   Per-layer cosine vs FLM's golden activations (THE gate)"
	@echo "  make gen            Greedy generation via the chat template (smoke test)"
	@echo "  make paris          One forward; PASS iff top-1 is ' Paris'"
	@echo "  make sublayer-ref   Rebuild sub-layer ground truth (needs $(HF_FP))"
	@echo ""
	@echo "Options: BUNDLE= FLM_REF= HF_FP= LAYERS=$(LAYERS) TOL=$(TOL) N_GEN=$(N_GEN) PROMPT=..."
	@echo ""
	@echo "Runs entirely on the host: no NPU, no Peano, no aie-opt."

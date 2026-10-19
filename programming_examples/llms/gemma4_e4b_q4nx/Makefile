# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Gemma4-E4B (text) Q4NX on NPU2 (MLIR-AIR).
#
# The same targets as ../gemma4_e2b_q4nx/Makefile. The Python entry points here
# run that directory's code at GEMMA4_VARIANT=e4b; artifacts land here.
#
# Quick start:
#   make compile          Build the prefill ELFs + both decode templates
#   make run              Generate on device; gated on the Paris continuation
#   make prefill          Prefill on device, gated vs the CPU reference
#   make layer-gate       Score each layer class on device against numpy

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)
DEC    := $(srcdir)/../../fused_decode_ple
PREFILL_DRIVER := $(srcdir)/gemma4_e4b_q4nx_prefill.py
FUSED          := $(srcdir)/fused_prefill
INFER_DRIVER   := $(srcdir)/gemma4_e4b_q4nx_inference.py
BUILD_DIR := build_peano

# The decode engine's build of this model, passed on every engine make line.
DEC_MODEL := DECODE_MODEL=gemma4-e4b MODEL_TYPE=GEMMA4_E4B VOCAB_CHUNK_I2=5

# Padded prefill length of the per-op prefill ELFs; changing it is a rebuild.
CTX ?= 2048
export Q4NX_SEQ_LEN := $(CTX)

# Kernel cache under the invoking directory, so lit tests do not share one.
export Q4NX_CACHE_DIR := $(CURDIR)/_q4nx_cache_seq$(CTX)

# The fused prefill's build. The drivers use it when FUSED_PREFILL_DIR holds
# one; FUSED_PREFILL_DIR= selects the per-op prefill.
FUSED_DIR ?= $(CURDIR)/_fused_prefill
override FUSED_DIR := $(abspath $(FUSED_DIR))
FUSED_JOBS ?= 6
FUSED_PREFILL_DIR ?= $(FUSED_DIR)
export FUSED_PREFILL_DIR

# Weight source: HF repo id (model.q4nx bundle) or a local dir/file.
MODEL_SOURCE ?= FastFlowLM/Gemma4-E4B-IT-NPU2
export Q4NX_MODEL_SOURCE := $(MODEL_SOURCE)

# FLM's golden activations (`make check`) and the full-precision checkpoint
# (`make sublayer-ref`); both local development artifacts.
FLM_REFERENCE ?=
export FLM_REFERENCE
GEMMA4_FP_CHECKPOINT ?=
export GEMMA4_FP_CHECKPOINT

# ATTN_MAXL of the layer gate's decode template, and LREF = the second build
# of the calibrated pair the decode sweep needs.
LBUILD ?= 16
LREF ?= $(shell expr $(LBUILD) - 1)

# Decoder layers in the token driver's and the sweep's template.
SWEEP_UNI_DEC ?= 42

# Which slab each decode wave attends; derived, since a wrong map builds fine.
SLIDING_WINDOW = $(shell cd $(srcdir) && $(PYTHON) -c "import gemma4_e4b_q4nx_weights as gw; print(gw.SLIDING_WINDOW)")
SWEEP_KV_SRC = $(shell cd $(srcdir) && $(PYTHON) -c "import gemma4_e4b_q4nx_weights as gw; print(','.join(str(gw.kv_source_layer(i)) for i in range($(SWEEP_UNI_DEC))))")

# ATTN_MAXL of the token driver's template: the context cap of run/ask/profile.
RUN_LBUILD ?= 128

LAYERS ?= 42
# Per-layer cosine floor for `make check`, and the prefill logit-cosine floor.
TOL    ?= 0.98
PREFILL_TOL ?= 0.99
# Empty selects the canonical Paris prompt for `make run`.
PROMPT ?=
REF_PROMPT ?= What is the capital of France?
PROMPT_ARG := $(if $(PROMPT),--prompt "$(PROMPT)",)
N_GEN  ?= 12
N_TOKENS ?= 64

# One layer per class: sliding / full with their own KV, and the KV-shared
# ones against the layer whose cache they read.
GATE_LAYERS ?= 0 5 23
GATE_PAIRS  ?= 22:24 23:29 23:41

VERIFY_RUNNER  := $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER := gemma4_e4b_q4nx.verify_adapter
MODEL_CHOICE   ?= instruct
MAX_PROMPTS    ?= 2

.DEFAULT_GOAL := help
.PHONY: help compile compile-decode compile-decode-full compile-decode-elf \
        _compile_decode_build compile-prefill prefill run ask profile \
        profile-prefill verify verify-paris verify-topk verify-full \
        diagnosis layer-gate layer-gate-shared check gen paris sublayer-ref clean

help:
	@echo "============================================================"
	@echo " GEMMA4-E4B Q4NX on NPU2 (MLIR-AIR) -- prefill + fused decode"
	@echo "============================================================"
	@echo ""
	@echo "On device:"
	@echo "  make compile            Build everything the device path needs (prefill + decode)"
	@echo "  make compile-decode     Build the LAYER GATE's decode template (1 layer, weight-free)"
	@echo "  make compile-decode-full Build the TOKEN DRIVER's template (42 layers, weight-free)"
	@echo "  make compile-decode-elf  The token driver's decode as one full ELF (DECODE_ELF=1)"
	@echo "  make compile-prefill    Build the prefill ELFs (weight-free)"
	@echo "  make run                Generate on NPU; gate the Paris continuation + stop"
	@echo "  make ask PROMPT=...     One Q&A turn ($(N_TOKENS) tokens)"
	@echo "  make prefill            Prefill on NPU; gate the logits vs the CPU reference"
	@echo "  make verify-paris       Alias for 'prefill' (fast weight-integrity smoke)"
	@echo "  make profile            TTFT + decode tok/s (the perf.json row)"
	@echo "  make profile-prefill    TTFT only, under the name the prefill sweep drives"
	@echo "  make compile-fused-prefill  Build the one-device chunked prefill (fused_prefill/)"
	@echo "  make fused-prefill      Fused prefill, one chunk; gate the logits vs the CPU reference"
	@echo "  make fused-prefill-long Fused prefill over three chunks of README text, same gate"
	@echo "  make run-fused          'run' with the fused prefill seeding the decoder"
	@echo "  make verify             The full on-device LAYER gate (both halves below)"
	@echo "  make layer-gate         Score own-KV layers ($(GATE_LAYERS)) vs the numpy reference"
	@echo "  make layer-gate-shared  Score KV-SHARED layers ($(GATE_PAIRS)), two-layer builds"
	@echo "  make verify-topk        Shared top-k gate vs the bf16 HF reference (capped)"
	@echo "  make verify-full        Same, over the whole prompt file"
	@echo "  make diagnosis          Single-prompt lens through the shared runner"
	@echo ""
	@echo "On the host (no NPU, no Peano, no aie-opt):"
	@echo "  make check              Per-layer cosine vs FLM's goldens (needs FLM_REFERENCE)"
	@echo "  make gen                Greedy generation via the chat template (smoke test)"
	@echo "  make paris              One forward; PASS iff top-1 is ' Paris'"
	@echo "  make sublayer-ref       Rebuild sub-layer ground truth (needs GEMMA4_FP_CHECKPOINT)"
	@echo ""
	@echo "  make clean              Remove build artifacts and caches"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  MODEL_SOURCE=<repo|dir> model.q4nx source (default $(MODEL_SOURCE))"
	@echo "  LBUILD=$(LBUILD)                ATTN_MAXL of the LAYER GATE's decode template"
	@echo "  RUN_LBUILD=$(RUN_LBUILD)         Context cap of the TOKEN DRIVER's template"
	@echo "  N_TOKENS=$(N_TOKENS)            Tokens generated by 'make ask' / 'make profile'"
	@echo "  TOL=$(TOL) LAYERS=$(LAYERS)      CPU gate floor / layer count"
	@echo "  CTX=$(CTX)               Padded prefill length (rebuilds the prefill ELFs)"

## The layer gate's decode template, left in the engine directory.
compile-decode:
	@$(MAKE) --no-print-directory -C $(DEC) compile-decode $(DEC_MODEL) \
	  LBUILD=$(LBUILD) PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)

## The per-op prefill ELFs, built for CTX rows. Weight-free.
compile-prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(PREFILL_DRIVER) --compile-only

## The token driver's full-depth template pair, at RUN_LBUILD.
compile-decode-full:
	@$(MAKE) --no-print-directory -f $(srcdir)/Makefile _compile_decode_build \
	  LBUILD=$(RUN_LBUILD) PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)

## The token driver's decode as one full ELF, selected with DECODE_ELF=1.
compile-decode-elf:
	@test -n "$(SWEEP_KV_SRC)" || { echo "cannot derive the KV sharing map from gemma4_e4b_q4nx_weights.kv_source_layer -- refusing to build an identity-mapped decode" >&2; exit 1; }
	@$(MAKE) --no-print-directory -C $(DEC) compile-decode-elf $(DEC_MODEL) LBUILD=$(RUN_LBUILD) \
	  UNI_DEC=$(SWEEP_UNI_DEC) KV_SRC=$(SWEEP_KV_SRC) PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)
	@mv -f $(DEC)/decode_scratchpad.elf $(DEC)/decode_scratchpad.params.txt \
	  $(DEC)/decode_scratchpad.maxl $(srcdir)/

compile: compile-decode compile-decode-full compile-prefill compile-fused-prefill

## Per-op prefill, gated on the whole logit vector vs the CPU reference.
prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(PREFILL_DRIVER) --gate --tol $(PREFILL_TOL)

## The fused prefill (../shared/fused_prefill).
compile-fused-prefill:
	$(PYTHON) $(FUSED)/build.py $(FUSED_DIR) -j $(FUSED_JOBS)

fused-prefill:
	$(PYTHON) $(FUSED)/verify.py $(FUSED_DIR) --tol $(PREFILL_TOL)

fused-prefill-long:
	$(PYTHON) $(FUSED)/verify.py $(FUSED_DIR) --tol $(PREFILL_TOL) \
	  --text-file $(srcdir)/README.md --text-tokens 280 --repeat 1

## NPU prefill + fused decode. With no PROMPT, the Paris gate.
run:
	@cd $(srcdir) && $(PYTHON) $(INFER_DRIVER) --gate --n-tokens 9 $(PROMPT_ARG)

run-fused:
	@cd $(srcdir) && $(PYTHON) $(INFER_DRIVER) --gate --n-tokens 9 --fused-prefill $(FUSED_DIR) $(PROMPT_ARG)

ask:
	@cd $(srcdir) && $(PYTHON) $(INFER_DRIVER) \
		--n-tokens $(N_TOKENS) $(PROMPT_ARG)

verify-paris: prefill

## TTFT + decode tok/s for the nightly's perf.json. --ignore-eos: the Paris
## prompt is answered in two tokens.
profile:
	@cd $(srcdir) && $(PYTHON) $(INFER_DRIVER) --ignore-eos \
		--n-tokens $(N_TOKENS) $(PROMPT_ARG)

## Warm per-op prefill TTFT at CTX, as bench/sweep_prefill.py reads it.
profile-prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && Q4NX_BENCH_L=$(CTX) $(PYTHON) $(PREFILL_DRIVER)

## The full-depth template pair bench/sweep_decode.py builds and dispatches,
## left here as decode_L<N>.{xclbin,insts.bin}. Not idempotent on purpose: the
## sweep rebuilds per context.
RB_STEP = cd $(DEC) && $(DEC_MODEL) DECODE_GOLDEN=1 DECODE_GOLDEN_L=$(LBUILD) FUSED_DECODE_PRINT_CONST=RB_STEP $(PYTHON) fused_decode_ple.py | tail -1

_compile_decode_build:
	@test -n "$(SWEEP_KV_SRC)" || { echo "cannot derive the KV sharing map from gemma4_e4b_q4nx_weights.kv_source_layer -- refusing to build an identity-mapped template" >&2; exit 1; }
	@set -e; for L in $(LBUILD) $(LREF); do \
	  DECODE_RT_ROUNDS=1 DECODE_RT_WINDOW=1 $(MAKE) --no-print-directory -C $(DEC) compile-decode $(DEC_MODEL) LBUILD=$$L \
	    UNI_DEC=$(SWEEP_UNI_DEC) KV_SRC=$(SWEEP_KV_SRC) \
	    PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR); \
	  mv -f $(DEC)/decode.xclbin    $(srcdir)/decode_L$$L.xclbin; \
	  mv -f $(DEC)/decode.insts.bin $(srcdir)/decode_L$$L.insts.bin; \
	  rm -f $(srcdir)/decode_L$$L.rt_pair; \
	  if [ -f $(DEC)/decode.rt_pair ]; then mv -f $(DEC)/decode.rt_pair $(srcdir)/decode_L$$L.rt_pair; fi; \
	done
	@# The .rb build locates the readback length words the host patches; the
	@# .rt_rounds sidecar holds the blocks a split readback drops.
	@set -e; S=$$($(RB_STEP)); R=$$(( ($(LBUILD) + 15) / 16 - S )); rm -f $(srcdir)/decode_L$(LBUILD).rb.insts.bin; \
	echo $$S > $(srcdir)/decode_L$(LBUILD).rt_rounds; \
	if [ $$R -ge 1 ]; then \
	  DECODE_RT_ROUNDS=1 DECODE_RT_WINDOW=1 DECODE_RB_ROUNDS=$$R $(MAKE) --no-print-directory -C $(DEC) compile-decode $(DEC_MODEL) \
	    LBUILD=$(LBUILD) UNI_DEC=$(SWEEP_UNI_DEC) KV_SRC=$(SWEEP_KV_SRC) \
	    PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR); \
	  mv -f $(DEC)/decode.insts.bin $(srcdir)/decode_L$(LBUILD).rb.insts.bin; \
	  rm -f $(DEC)/decode.xclbin; \
	fi
	@# The .sw build locates the sliding-window readback start words.
	@set -e; S=$$($(RB_STEP)); W=$(SLIDING_WINDOW); rm -f $(srcdir)/decode_L$(LBUILD).sw.insts.bin; \
	echo $$W > $(srcdir)/decode_L$(LBUILD).rt_window; \
	if [ $(LBUILD) -gt $$W ]; then \
	  DECODE_RT_ROUNDS=1 DECODE_RT_WINDOW=1 DECODE_RB_SWA_SKIP=$$S $(MAKE) --no-print-directory -C $(DEC) compile-decode $(DEC_MODEL) \
	    LBUILD=$(LBUILD) UNI_DEC=$(SWEEP_UNI_DEC) KV_SRC=$(SWEEP_KV_SRC) \
	    PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR); \
	  mv -f $(DEC)/decode.insts.bin $(srcdir)/decode_L$(LBUILD).sw.insts.bin; \
	  rm -f $(DEC)/decode.xclbin; \
	fi
	@echo "Gemma4 decode templates built: decode_L$(LBUILD) + decode_L$(LREF) (UNI_DEC=$(SWEEP_UNI_DEC))."

## validate_layer_npu.py exits 0 pass, 1 fail, 2 inconclusive (a dispatch
## after an NPU timeout cannot be scored). Only 2 is retried, in a new process.
SCORE_ATTEMPTS ?= 3
define score_layer
  n=0; while :; do \
    n=$$((n+1)); \
    $(PYTHON) validate_layer_npu.py $$SCORE_ARGS && break; rc=$$?; \
    if [ $$rc -ne 2 ]; then exit $$rc; fi; \
    if [ $$n -ge $(SCORE_ATTEMPTS) ]; then \
      echo "FAIL: still INCONCLUSIVE after $(SCORE_ATTEMPTS) attempts; the NPU2 TDR did not clear, so this run scored nothing"; \
      exit 1; \
    fi; \
    echo "  (inconclusive: an NPU2 TDR, not a numeric failure -- re-scoring in a fresh process, attempt $$((n+1)) of $(SCORE_ATTEMPTS))"; \
  done
endef

## Own-KV layers, one per class. Rebuilds the template each time: the gate
## dispatches whatever decode.xclbin is on disk.
layer-gate:
	@set -e; for L in $(GATE_LAYERS); do \
	  $(MAKE) --no-print-directory -C $(DEC) compile-decode $(DEC_MODEL) LBUILD=$(LBUILD) \
	    UNI_DEC=1 LAYERS=$$L PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) >/dev/null; \
	  cd $(srcdir); \
	  SCORE_ARGS="--layer $$L --attn-maxl $(LBUILD) --rebuild-cache --cache /tmp/g4e4b_L$$L.npz"; \
	  $(score_layer); \
	done

## KV-shared layers: a two-layer build, the source in slab 0, scored from the
## device's own slab-0 output (--prefix-dump).
layer-gate-shared:
	@set -e; for P in $(GATE_PAIRS); do \
	  SRC=$${P%:*}; L=$${P#*:}; \
	  $(MAKE) --no-print-directory -C $(DEC) compile-decode $(DEC_MODEL) LBUILD=$(LBUILD) \
	    UNI_DEC=1 LAYERS=$$SRC PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) >/dev/null; \
	  cd $(srcdir); \
	  SCORE_ARGS="--layer $$SRC --attn-maxl $(LBUILD) --rebuild-cache --cache /tmp/g4e4b_L$$SRC.npz --dump /tmp/g4e4b_d$$SRC.npz"; \
	  $(score_layer); \
	  $(MAKE) --no-print-directory -C $(DEC) compile-decode $(DEC_MODEL) LBUILD=$(LBUILD) \
	    UNI_DEC=2 KV_SRC=0,0 LAYERS=$$SRC,$$L PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) >/dev/null; \
	  cd $(srcdir); \
	  SCORE_ARGS="--layers $$SRC,$$L --attn-maxl $(LBUILD) --rebuild-cache --cache /tmp/g4e4b_L$${SRC}_$$L.npz --prefix-dump /tmp/g4e4b_d$$SRC.npz"; \
	  $(score_layer); \
	done

## The on-device correctness gate: the decoder layer per class.
verify: layer-gate layer-gate-shared

## The shared top-k token-set gate vs the bf16 HF reference.
verify-topk:
	cd $(srcdir)/.. && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL_CHOICE) --max-prompts $(MAX_PROMPTS)

verify-full:
	cd $(srcdir)/.. && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL_CHOICE)

diagnosis:
	cd $(srcdir)/.. && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(if $(PROMPT),$(PROMPT),$(REF_PROMPT))" \
		--model $(MODEL_CHOICE)

## CPU-only per-layer gate against FLM's golden activations (FLM_REFERENCE).
check:
	cd $(srcdir) && $(PYTHON) check_vs_flm_reference.py \
		--bundle $(MODEL_SOURCE) --layers $(LAYERS) --tol $(TOL)

## Greedy generation through the chat template. Smoke test, not a gate.
gen:
	cd $(srcdir) && $(PYTHON) run_reference.py --bundle $(MODEL_SOURCE) gen \
		--prompt "$(if $(PROMPT),$(PROMPT),$(REF_PROMPT))" --n-tokens $(N_GEN)

paris:
	cd $(srcdir) && $(PYTHON) run_reference.py --bundle $(MODEL_SOURCE) paris

sublayer-ref:
	cd $(srcdir) && $(PYTHON) dump_sublayer_reference.py \
		--layers 7,8,9 --out sublayer_ref.npz

clean:
	@$(MAKE) --no-print-directory -C $(DEC) clean
	rm -rf $(srcdir)/__pycache__ $(srcdir)/build_peano $(srcdir)/build_chess 2>/dev/null || true
	rm -f $(srcdir)/*.npz $(srcdir)/decode_L*.xclbin $(srcdir)/decode_L*.insts.bin $(srcdir)/decode_L*.rt_rounds $(srcdir)/decode_L*.rt_window $(srcdir)/decode_L*.rt_pair $(srcdir)/decode_scratchpad.* 2>/dev/null || true
	rm -rf $(srcdir)/_q4nx_cache_seq* $(CURDIR)/_q4nx_cache_seq* $(srcdir)/build_peano $(FUSED_DIR) 2>/dev/null || true
	@echo "Gemma4 caches and the PLE decode build removed."

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
# More host threads slow the NPU down: they compete with its weight stream
# for memory bandwidth.
export OMP_NUM_THREADS ?= 4
export OPENBLAS_NUM_THREADS ?= 4
BUILD_DIR ?= $(srcdir)/build
override BUILD_DIR := $(abspath $(BUILD_DIR))
N_TOKENS ?= 128
PROMPT ?= Explain how speculative decoding speeds up language models.
# Longest prompt run and chat accept, 2048 or 4096; one prefill build serves both.
PREFILL_CTX ?= 2048
PREFILL_DIR = $(BUILD_DIR)/prefill_C2048
BENCHMARK_BUNDLE ?= $(HOME)/.cache/mlir-air/dflash/gemma4_bolt_e4b/benchmark-npu-v1
BENCHMARK_OUTPUT ?= $(BUILD_DIR)/benchmark.jsonl
STUDY_TOKENS ?= 2048
PER_DATASET ?= 16
ifndef VERIFY_DIR
VERIFY_DIR := $(BUILD_DIR)/verification/$(shell date +%Y%m%d-%H%M%S)
endif
PROFILE_OUTPUT ?= $(BUILD_DIR)/gemma4_bolt_e4b_dflash_b8.perf.json
FULL20_OUTPUT ?= $(BUILD_DIR)/gemma4_bolt_e4b_dflash_b8.full20.json
FULL80_OUTPUT ?= $(BUILD_DIR)/gemma4_bolt_e4b_dflash_b8.full80.json
OFFLINE ?= 0
OFFLINE_ARG = $(if $(filter 1,$(OFFLINE)),--offline,)
export DFLASH_PROMPT := $(PROMPT)
export AIR_TARGET_DEVICE := npu2
# Keep the session lock distinct from KernelCache's inner /tmp/npu.lock.
NPU_RUN ?= flock --exclusive /tmp/mlir-air-npu.lock $(PYTHON)
PAIR = --target-artifacts $(BUILD_DIR)/target --draft-artifacts $(BUILD_DIR)/draft
PREFILL = --prefill npu --prefill-context $(PREFILL_CTX) --prefill-artifacts $(PREFILL_DIR)
MODEL_PAIR = --target $(BUILD_DIR)/target --draft $(BUILD_DIR)/draft --prepass $(BUILD_DIR)/prepass

.NOTPARALLEL:
.PHONY: help download weights compile compile-target compile-draft compile-prepass compile-prefill compile-baseline compile-sequences verify-sequences run chat verify verify-state verify-model profile full20 full80 benchmark-data benchmark clean
help:
	@echo 'Nemotron-3.5-Bolt-E4B DFlash B8 on NPU2 (use the normal AIR environment)'
	@echo 'make download         Seed the pinned checkpoints for offline runs'
	@echo 'make weights          Download the pinned checkpoints and pack the drafter'
	@echo 'make compile          Compile target, drafter and NPU prefill'
	@echo 'make run              Stream one response; PROMPT="..." N_TOKENS=128'
	@echo 'make chat             Load once, then read and stream independent prompts'
	@echo 'make verify           Device state, loop and top-5 model checks'
	@echo 'make benchmark        Fixed 80-prompt saved-prefill decode study'
	@echo 'make profile          Fixed-prompt NPU prefill + DFlash performance JSON'
	@echo 'make full20           Nightly B1 + DFlash study, 4 prompts per dataset'
	@echo 'make full80           Optional B1 + DFlash study, 16 prompts per dataset'
	@echo 'make compile-sequences  Target for two independent sequences'
	@echo 'make verify-sequences   Its checks against the float32 reference'
	@echo 'make clean            Remove this example build directory'

download:
	$(PYTHON) $(srcdir)/resources.py $(OFFLINE_ARG)

weights:
	$(PYTHON) $(srcdir)/resources.py $(OFFLINE_ARG)
	$(PYTHON) $(srcdir)/prepare_weights.py --manifest $(BUILD_DIR)/weights.json $(OFFLINE_ARG)

compile: compile-target compile-draft compile-prepass compile-prefill
compile-target:
	$(PYTHON) $(srcdir)/build_target.py --output $(BUILD_DIR)/target --check-existing

compile-draft:
	$(PYTHON) $(srcdir)/build_draft.py --output $(BUILD_DIR)/draft --check-existing

compile-prepass:
	$(PYTHON) $(srcdir)/context_prepass.py --output $(BUILD_DIR)/prepass

compile-prefill:
	$(PYTHON) $(srcdir)/npu_prefill.py --cache-dir $(PREFILL_DIR)

# The target for two independent sequences, one token each per dispatch.
SEQUENCES_DIR = $(BUILD_DIR)/sequences
compile-sequences:
	$(PYTHON) $(srcdir)/build_target.py --output $(SEQUENCES_DIR) --batch 2 --sequences 2 --capacity $(PREFILL_CTX) --check-existing

verify-sequences:
	$(NPU_RUN) $(srcdir)/tests/check_sequences.py --artifact $(SEQUENCES_DIR) --output $(VERIFY_DIR)/sequences.json
	$(NPU_RUN) $(srcdir)/tests/check_attention.py --batch 2 --output $(VERIFY_DIR)/attention_b2

compile-baseline:
	$(PYTHON) $(srcdir)/build_baseline.py --output $(BUILD_DIR)/baseline

run:
	$(NPU_RUN) $(srcdir)/run.py $(PAIR) $(PREFILL) --tokens $(N_TOKENS)

chat:
	$(NPU_RUN) $(srcdir)/run.py $(PAIR) $(PREFILL) --tokens $(N_TOKENS) --interactive --output $(BUILD_DIR)/chat.jsonl

verify: verify-state verify-model
verify-state:
	$(NPU_RUN) $(srcdir)/tests/check_target.py --artifact $(BUILD_DIR)/target --output $(VERIFY_DIR)/target.json
	$(NPU_RUN) $(srcdir)/tests/check_long_context.py --target $(BUILD_DIR)/target --prefill-artifacts $(PREFILL_DIR) --output $(VERIFY_DIR)/long_context.json
	$(NPU_RUN) $(srcdir)/tests/check_attention.py --output $(VERIFY_DIR)/attention_full
	$(NPU_RUN) $(srcdir)/tests/check_attention.py --window 512 --output $(VERIFY_DIR)/attention_sliding
	$(NPU_RUN) $(srcdir)/verify_loop.py $(PAIR) $(PREFILL) --output $(VERIFY_DIR)/loop.json

verify-model:
	$(NPU_RUN) $(srcdir)/tests/check_model.py --phase npu $(MODEL_PAIR) --prefill-artifacts $(PREFILL_DIR) --output $(VERIFY_DIR)/model-npu.json
	$(PYTHON) $(srcdir)/tests/check_model.py --phase reference --npu-capture $(VERIFY_DIR)/model-npu.json --output $(VERIFY_DIR)/model-reference.json
	$(PYTHON) $(srcdir)/tests/check_model.py --phase compare --npu-capture $(VERIFY_DIR)/model-npu.json --hf-capture $(VERIFY_DIR)/model-reference.json --output $(VERIFY_DIR)/model.json

benchmark-data:
	$(NPU_RUN) $(srcdir)/benchmark_data.py --destination $(BENCHMARK_BUNDLE) --prefill-artifacts $(PREFILL_DIR) --per-dataset $(PER_DATASET)

benchmark: benchmark-data
	$(NPU_RUN) $(srcdir)/benchmark.py --inputs $(BENCHMARK_BUNDLE)/inputs $(MODEL_PAIR) --output $(BENCHMARK_OUTPUT) --tokens $(STUDY_TOKENS) --per-dataset $(PER_DATASET)

profile:
	$(NPU_RUN) $(srcdir)/profile_dflash.py --build $(BUILD_DIR) --output $(PROFILE_OUTPUT) --model gemma4_bolt_e4b_dflash_b8

full20:
	$(MAKE) -f $(srcdir)/Makefile benchmark-data PER_DATASET=4
	$(NPU_RUN) $(srcdir)/paired_benchmark.py --build $(BUILD_DIR) --inputs $(BENCHMARK_BUNDLE)/inputs --per-dataset 4 --output $(FULL20_OUTPUT)

full80:
	$(MAKE) -f $(srcdir)/Makefile benchmark-data PER_DATASET=16
	$(NPU_RUN) $(srcdir)/paired_benchmark.py --build $(BUILD_DIR) --inputs $(BENCHMARK_BUNDLE)/inputs --per-dataset 16 --output $(FULL80_OUTPUT)

clean:
	@case "$(abspath $(BUILD_DIR))" in "$(srcdir)"/build*) ;; *) echo 'Refusing to remove a directory outside this example build*'; exit 1;; esac
	@if [ -e "$(BUILD_DIR)" ] && [ ! -d "$(BUILD_DIR)" ]; then echo "BUILD_DIR must be a directory"; exit 1; fi
	rm -rf -- "$(BUILD_DIR)"

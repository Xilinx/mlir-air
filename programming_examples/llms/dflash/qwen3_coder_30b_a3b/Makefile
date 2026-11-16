# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
BUILD_DIR ?= $(srcdir)/build
override BUILD_DIR := $(abspath $(BUILD_DIR))
N_TOKENS ?= 128
PROMPT ?= Write a Python function that checks if a number is prime.
# Longest prompt run and chat accept, up to the target's 4096 positions.
PREFILL_CTX ?= 2048
BENCHMARK_BUNDLE ?= $(HOME)/.cache/mlir-air/dflash/qwen3_coder_30b_a3b/benchmark-npu-v1
BENCHMARK_OUTPUT ?= $(BUILD_DIR)/benchmark.jsonl
STUDY_TOKENS ?= 2048
PER_DATASET ?= 16
ifndef VERIFY_DIR
VERIFY_DIR := $(BUILD_DIR)/verification/$(shell date +%Y%m%d-%H%M%S)
endif
# The drafter is trained on code: profile a fixed coding prompt.
PROFILE_PROMPT := Write a Python function that returns the n-th Fibonacci number.
PROFILE_OUTPUT ?= $(BUILD_DIR)/qwen3_coder_30b_a3b_dflash_b8.perf.json
FULL20_OUTPUT ?= $(BUILD_DIR)/qwen3_coder_30b_a3b_dflash_b8.full20.json
FULL80_OUTPUT ?= $(BUILD_DIR)/qwen3_coder_30b_a3b_dflash_b8.full80.json
TARGET_DIR := $(abspath $(srcdir)/../../qwen3_coder_30b_a3b_q4k)
export DFLASH_PROMPT := $(PROMPT)
# Also lets the builders compile on hosts without device access.
export AIR_TARGET_DEVICE := npu2
# Keep the session lock distinct from the backend's inner /tmp/npu.lock.
NPU_RUN ?= flock --exclusive /tmp/mlir-air-npu.lock $(PYTHON)
PAIR = --target-artifacts $(BUILD_DIR)/target --draft-artifacts $(BUILD_DIR)/draft
PREFILL = --prefill npu --prefill-context $(PREFILL_CTX)
MODEL_PAIR = --target $(BUILD_DIR)/target --draft $(BUILD_DIR)/draft --prepass $(BUILD_DIR)/prepass

.NOTPARALLEL:
.PHONY: help weights compile compile-target compile-draft compile-prepass compile-baseline run chat verify verify-state verify-model benchmark-data benchmark profile full20 full80 clean
help:
	@echo 'Qwen3-Coder-30B-A3B DFlash B8 on NPU2 (use the normal AIR environment)'
	@echo 'make weights          Quantize the target into its Q4_K bundle; pack the drafter'
	@echo 'make compile          Compile the target (every cache read) and the drafter'
	@echo 'make run              Stream one response; PROMPT="..." N_TOKENS=128'
	@echo 'make chat             Load once, then read and stream independent prompts'
	@echo 'make verify           Loop against its sequential oracle; target against the CPU reference'
	@echo 'make benchmark        Fixed 80-prompt saved-prefill decode study'
	@echo 'make profile          Fixed-prompt NPU prefill + DFlash performance JSON'
	@echo 'make full20           Nightly B1 + DFlash study, 4 prompts per dataset'
	@echo 'make full80           Optional B1 + DFlash study, 16 prompts per dataset'
	@echo 'make clean            Remove this example build directory'

weights:
	$(MAKE) -f $(TARGET_DIR)/Makefile prepare
	$(PYTHON) $(srcdir)/prepare_coder_draft.py

compile: compile-target compile-draft compile-prepass
compile-target:
	$(PYTHON) $(srcdir)/build_target.py --output $(BUILD_DIR)/target

compile-draft:
	$(PYTHON) $(srcdir)/build_draft.py --output $(BUILD_DIR)/draft --role drafter --block-size 8 --check-existing

compile-prepass:
	$(PYTHON) $(srcdir)/context_prepass.py --output $(BUILD_DIR)/prepass

compile-baseline:
	$(PYTHON) $(srcdir)/build_baseline.py --output $(BUILD_DIR)/baseline

run:
	$(NPU_RUN) $(srcdir)/run.py $(PAIR) $(PREFILL) --tokens $(N_TOKENS)

chat:
	$(NPU_RUN) $(srcdir)/run.py $(PAIR) $(PREFILL) --tokens $(N_TOKENS) --interactive --output $(BUILD_DIR)/chat.jsonl

verify: verify-state verify-model
verify-state:
	$(NPU_RUN) $(srcdir)/verify_loop.py $(PAIR) $(PREFILL) --output $(VERIFY_DIR)/loop.json

# The 48-layer verify pass's argmax against the CPU reference's greedy.
verify-model:
	$(NPU_RUN) $(srcdir)/verify_b8.py --root $(BUILD_DIR)/target

benchmark-data:
	$(NPU_RUN) $(srcdir)/benchmark_data.py --destination $(BENCHMARK_BUNDLE) --target $(BUILD_DIR)/target --per-dataset $(PER_DATASET)

benchmark: benchmark-data
	$(NPU_RUN) $(srcdir)/benchmark.py --inputs $(BENCHMARK_BUNDLE)/inputs $(MODEL_PAIR) --output $(BENCHMARK_OUTPUT) --tokens $(STUDY_TOKENS) --per-dataset $(PER_DATASET)

profile:
	$(NPU_RUN) $(srcdir)/profile_dflash.py --build $(BUILD_DIR) --output $(PROFILE_OUTPUT) --model qwen3_coder_30b_a3b_dflash_b8 --prompt "$(PROFILE_PROMPT)"

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

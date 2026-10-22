# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
export OMP_NUM_THREADS ?= 8
export OPENBLAS_NUM_THREADS ?= 8
BUILD_DIR ?= $(srcdir)/build
override BUILD_DIR := $(abspath $(BUILD_DIR))
N_TOKENS ?= 128
PROMPT ?= Explain how speculative decoding speeds up language models.
PREFILL_CTX ?= 2048
BENCHMARK_BUNDLE ?= $(HOME)/.cache/mlir-air/dflash/llama31_8b/benchmark-npu-v1
BENCHMARK_OUTPUT ?= $(BUILD_DIR)/benchmark.jsonl
STUDY_TOKENS ?= 2048
PER_DATASET ?= 16
ifndef VERIFY_DIR
VERIFY_DIR := $(BUILD_DIR)/verification/$(shell date +%Y%m%d-%H%M%S)
endif
PROFILE_OUTPUT ?= $(BUILD_DIR)/llama31_8b_dflash_b8.perf.json
FULL20_OUTPUT ?= $(BUILD_DIR)/llama31_8b_dflash_b8.full20.json
FULL80_OUTPUT ?= $(BUILD_DIR)/llama31_8b_dflash_b8.full80.json
OFFLINE ?= 0
OFFLINE_ARG = $(if $(filter 1,$(OFFLINE)),--offline,)
export DFLASH_PROMPT := $(PROMPT)
# Also lets shared prefill builders compile on hosts without device access.
export AIR_TARGET_DEVICE := npu2
# Keep the session lock distinct from KernelCache's inner /tmp/npu.lock.
NPU_RUN ?= flock --exclusive /tmp/mlir-air-npu.lock $(PYTHON)
PAIR = --target-artifacts $(BUILD_DIR)/target --draft-artifacts $(BUILD_DIR)/draft
PREFILL = --prefill npu --prefill-context $(PREFILL_CTX) --prefill-artifacts $(BUILD_DIR)/prefill_C$(PREFILL_CTX)
MODEL_PAIR = --target $(BUILD_DIR)/target --draft $(BUILD_DIR)/draft --prepass $(BUILD_DIR)/prepass

.NOTPARALLEL:
.PHONY: help download weights compile compile-target compile-draft compile-prepass compile-prefill compile-baseline run chat verify verify-host verify-state verify-model profile full20 full80 benchmark-data benchmark clean
help:
	@echo 'Llama-3.1-8B DFlash B8 on NPU2 (use the normal AIR environment)'
	@echo 'make weights          Download pinned checkpoints and pack weights'
	@echo 'make download         Seed pinned HF checkpoints; prompt texts are included for offline runs'
	@echo 'make compile          Compile target, drafter, NPU prepass and prefill'
	@echo 'make run              Stream one response; PROMPT="..." N_TOKENS=128'
	@echo 'make chat             Load once, then read and stream independent prompts'
	@echo 'make verify-host      Host acceptance and streaming checks'
	@echo 'make verify           Device state, loop and HF BF16 top-5 model checks'
	@echo 'make benchmark        Fixed 80-prompt saved-prefill decode study'
	@echo 'make profile          Fixed-prompt NPU prefill + DFlash performance JSON'
	@echo 'make full20           Nightly B1 + DFlash study, 4 prompts per dataset'
	@echo 'make full80           Optional B1 + DFlash study, 16 prompts per dataset'
	@echo 'make clean            Remove this example build directory'

download:
	$(PYTHON) $(srcdir)/resources.py --reference $(OFFLINE_ARG)

weights:
	$(PYTHON) $(srcdir)/resources.py $(OFFLINE_ARG)
	$(PYTHON) $(srcdir)/prepare_weights.py --manifest $(BUILD_DIR)/weights.json $(OFFLINE_ARG)

compile: compile-target compile-draft compile-prepass compile-prefill
compile-target:
	$(PYTHON) $(srcdir)/build_target.py --output $(BUILD_DIR)/target --role target --check-existing

compile-draft: compile-target
	$(PYTHON) $(srcdir)/build_target.py --output $(BUILD_DIR)/draft --role drafter --reuse-objects $(BUILD_DIR)/target --check-existing

compile-prepass:
	$(PYTHON) $(srcdir)/build_bfp_prepass.py --output $(BUILD_DIR)/prepass --check-existing

compile-prefill:
	$(PYTHON) $(srcdir)/npu_prefill.py --seq-len $(PREFILL_CTX) --cache-dir $(BUILD_DIR)/prefill_C$(PREFILL_CTX)

compile-baseline:
	$(PYTHON) $(srcdir)/build_baseline.py --output $(BUILD_DIR)/baseline

run:
	$(NPU_RUN) $(srcdir)/run.py $(PAIR) $(PREFILL) --tokens $(N_TOKENS)

chat:
	$(NPU_RUN) $(srcdir)/run.py $(PAIR) $(PREFILL) --tokens $(N_TOKENS) --interactive --output $(BUILD_DIR)/chat.jsonl

verify: verify-state verify-model
verify-host:
	$(MAKE) -f $(srcdir)/../qwen3_4b/Makefile verify-host

verify-state:
	$(NPU_RUN) $(srcdir)/tests/check_decoder.py --artifact $(BUILD_DIR)/target --role target --prefixes 0 7 16 64 1024 4088 --output $(VERIFY_DIR)/target.json
	$(NPU_RUN) $(srcdir)/tests/check_decoder.py --artifact $(BUILD_DIR)/draft --role drafter --prefixes 0 7 16 64 1024 4088 --output $(VERIFY_DIR)/draft.json
	$(NPU_RUN) $(srcdir)/verify_loop.py $(PAIR) $(PREFILL) --output $(VERIFY_DIR)/loop.json

verify-model:
	$(PYTHON) $(srcdir)/resources.py --reference $(OFFLINE_ARG)
	$(NPU_RUN) $(srcdir)/tests/check_model.py --phase npu $(MODEL_PAIR) --prefill-artifacts $(BUILD_DIR)/prefill_C2048 --output $(VERIFY_DIR)/model-npu.json
	$(PYTHON) $(srcdir)/tests/check_model.py --phase hf --output $(VERIFY_DIR)/model-hf.json
	$(PYTHON) $(srcdir)/tests/check_model.py --phase compare --npu-capture $(VERIFY_DIR)/model-npu.json --hf-capture $(VERIFY_DIR)/model-hf.json --output $(VERIFY_DIR)/model.json

# Benchmark states are prepared with the C2048 prefill, whatever PREFILL_CTX is.
benchmark-data:
	$(NPU_RUN) $(srcdir)/benchmark_data.py --destination $(BENCHMARK_BUNDLE) --prefill-artifacts $(BUILD_DIR)/prefill_C2048 --per-dataset $(PER_DATASET)

benchmark: benchmark-data
	$(NPU_RUN) $(srcdir)/benchmark.py --inputs $(BENCHMARK_BUNDLE)/inputs $(MODEL_PAIR) --output $(BENCHMARK_OUTPUT) --tokens $(STUDY_TOKENS) --per-dataset $(PER_DATASET)

profile:
	$(NPU_RUN) $(srcdir)/profile_dflash.py --build $(BUILD_DIR) --output $(PROFILE_OUTPUT) --model llama31_8b_dflash_b8

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

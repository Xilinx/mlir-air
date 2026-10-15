# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
BUILD_DIR ?= $(srcdir)/build
N_TOKENS ?= 128
PROMPT ?= Explain how speculative decoding speeds up language models.
PREFILL_CTX ?= 2048
BENCHMARK_BUNDLE ?= $(BUILD_DIR)/benchmark_bundle
BENCHMARK_OUTPUT ?= $(BUILD_DIR)/benchmark.jsonl
STUDY_TOKENS ?= 2048
PER_DATASET ?= 16
VERIFY_DIR := $(BUILD_DIR)/verification/$(shell date +%Y%m%d-%H%M%S)
export DFLASH_PROMPT := $(PROMPT)
# Also lets shared prefill builders compile on hosts without device access.
export AIR_TARGET_DEVICE := npu2
NPU_RUN = flock --exclusive --nonblock /tmp/mlir-air-npu.lock $(PYTHON)
PAIR = --target-artifacts $(BUILD_DIR)/target --draft-artifacts $(BUILD_DIR)/draft
PREFILL = --prefill npu --prefill-context $(PREFILL_CTX) --prefill-artifacts $(BUILD_DIR)/prefill_C$(PREFILL_CTX)
MODEL_PAIR = --target $(BUILD_DIR)/target --draft $(BUILD_DIR)/draft --prepass $(BUILD_DIR)/prepass

.NOTPARALLEL:
.PHONY: help weights compile compile-target compile-draft compile-prepass compile-prefill run chat verify verify-host verify-state verify-model benchmark-data benchmark clean
help:
	@echo 'Qwen3-4B DFlash B8 on NPU2 (use the normal AIR environment)'
	@echo 'make weights          Download pinned checkpoints and pack weights'
	@echo 'make compile          Compile target, drafter, NPU prepass and prefill'
	@echo 'make run              Stream one response; PROMPT="..." N_TOKENS=128'
	@echo 'make chat             Load once, then read and stream independent prompts'
	@echo 'make verify-host      Host acceptance and streaming checks'
	@echo 'make verify           Host, device state and HF BF16 top-5 model checks'
	@echo 'make benchmark        Fixed 80-prompt saved-prefill decode study'
	@echo 'make clean            Remove this example build directory'

weights:
	$(PYTHON) $(srcdir)/prepare_weights.py --manifest $(BUILD_DIR)/weights.json

compile: compile-target compile-draft compile-prepass compile-prefill
compile-target:
	$(PYTHON) $(srcdir)/build_target.py --output $(BUILD_DIR)/target --role target --check-existing

compile-draft: compile-target
	$(PYTHON) $(srcdir)/build_target.py --output $(BUILD_DIR)/draft --role drafter --reuse-objects $(BUILD_DIR)/target --check-existing

compile-prepass:
	$(PYTHON) $(srcdir)/build_bfp_prepass.py --output $(BUILD_DIR)/prepass --check-existing

compile-prefill:
	$(PYTHON) $(srcdir)/npu_prefill.py --seq-len $(PREFILL_CTX) --cache-dir $(BUILD_DIR)/prefill_C$(PREFILL_CTX)

run:
	$(NPU_RUN) $(srcdir)/run.py $(PAIR) $(PREFILL) --tokens $(N_TOKENS)

chat:
	$(NPU_RUN) $(srcdir)/run.py $(PAIR) $(PREFILL) --tokens $(N_TOKENS) --interactive --output $(BUILD_DIR)/chat.jsonl

verify: verify-host verify-state verify-model
verify-host:
	$(PYTHON) $(srcdir)/verify.py --host-only
	$(PYTHON) $(srcdir)/test_streaming.py

verify-state:
	$(NPU_RUN) $(srcdir)/tests/check_decoder.py --artifact $(BUILD_DIR)/target --role target --prefixes 0 7 16 64 1024 4088 --output $(VERIFY_DIR)/target.json
	$(NPU_RUN) $(srcdir)/tests/check_decoder.py --artifact $(BUILD_DIR)/draft --role drafter --prefixes 0 7 16 64 1024 4088 --output $(VERIFY_DIR)/draft.json
	$(NPU_RUN) $(srcdir)/verify.py $(PAIR) $(PREFILL) --output $(VERIFY_DIR)/loop.json

verify-model:
	$(NPU_RUN) $(srcdir)/tests/check_model.py --phase npu $(MODEL_PAIR) --prefill-artifacts $(BUILD_DIR)/prefill_C2048 --output $(VERIFY_DIR)/model-npu.json
	$(PYTHON) $(srcdir)/tests/check_model.py --phase hf --output $(VERIFY_DIR)/model-hf.json
	$(PYTHON) $(srcdir)/tests/check_model.py --phase compare --npu-capture $(VERIFY_DIR)/model-npu.json --hf-capture $(VERIFY_DIR)/model-hf.json --output $(VERIFY_DIR)/model.json

benchmark-data:
	$(PYTHON) $(srcdir)/benchmark_data.py --destination $(BENCHMARK_BUNDLE)

benchmark: benchmark-data
	$(NPU_RUN) $(srcdir)/benchmark.py --inputs $(BENCHMARK_BUNDLE)/inputs $(MODEL_PAIR) --output $(BENCHMARK_OUTPUT) --tokens $(STUDY_TOKENS) --per-dataset $(PER_DATASET)

clean:
	@case "$(abspath $(BUILD_DIR))" in "$(srcdir)"/build*) ;; *) echo 'Refusing to remove a directory outside this example build*'; exit 1;; esac
	@if [ -e "$(BUILD_DIR)" ] && [ ! -d "$(BUILD_DIR)" ]; then echo "BUILD_DIR must be a directory"; exit 1; fi
	rm -rf -- "$(BUILD_DIR)"

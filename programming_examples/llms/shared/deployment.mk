# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
PYTHON ?= python3
BUILD_DIR ?= $(srcdir)/build
P ?= 512
N_TOKENS ?= 32
REFERENCE_DIR ?= $(abspath $(BUILD_DIR))/reference
DRIVER = cd $(srcdir)/.. && $(PYTHON) -m $(MODEL_MODULE)
OPTIONS = --build-dir "$(abspath $(BUILD_DIR))" --prefill $(P) \
  $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)") \
  $(if $(Q4_DIR),--q4-dir "$(Q4_DIR)") \
  $(if $(INPUT_IDS),--input-ids "$(INPUT_IDS)")

.PHONY: help info weights compile compile-kernels compile-decode compile-prefill \
        compile-head reference verify run profile profile-decode compile-all verify-all \
        profile-all
.DEFAULT_GOAL := help
help:
	@echo "$(MODEL_MODULE): affine Q4, BFP16 prefill GEMM, NPU2"
	@echo "make weights [CHECKPOINT=/path/to/HF] [Q4_DIR=/path/to/canonical-q4]"
	@echo "make compile P=512  (native P values: $(NATIVE_LENGTHS))"
	@echo "make reference P=512 CHECKPOINT=/path/to/HF"
	@echo "make $(VERIFY_NATIVE) P=512 REFERENCE_DIR=/path/to/reference"
	@echo "make run P=512 N_TOKENS=32; make profile P=512"
	@echo "compile-all / verify-all / profile-all visit every native configuration"
ifdef RUNNER_ADAPTER
	@echo "make verify / accuracy / tasks: verify/ subsystem (after make weights compile P=1024)"
endif
ifneq ($(RUNNER_ADAPTER)$(FUSED_PREFILL),)
	@echo "fused prefill: any input up to its built length (one build for every P)"
	@echo "make compile-fused-prefill / profile-fused-prefill FUSED_LENGTHS=512,2048"
endif
info:
	$(DRIVER) info
weights:
	$(DRIVER) weights $(OPTIONS)
compile:
	$(DRIVER) compile $(OPTIONS)
compile-kernels:
	$(DRIVER) compile $(OPTIONS) --part kernels
compile-decode:
	$(DRIVER) compile $(OPTIONS) --part decode
compile-prefill:
	$(DRIVER) compile $(OPTIONS) --part prefill
compile-head:
	$(DRIVER) compile $(OPTIONS) --part head
reference:
	$(DRIVER) reference $(OPTIONS) --out "$(REFERENCE_DIR)"
# A model on the shared fused prefill sets RUNNER_ADAPTER: `verify` is then the
# verify/ gate, and this fixed-input gate is `verify-native`.
VERIFY_NATIVE := $(if $(RUNNER_ADAPTER),verify-native,verify)
$(VERIFY_NATIVE):
	$(DRIVER) verify $(OPTIONS) --reference-dir "$(REFERENCE_DIR)"
run:
	$(DRIVER) run $(OPTIONS) --outputs $(N_TOKENS)
profile:
	$(DRIVER) profile $(OPTIONS)
# Decode tok/s at each of CONTEXTS (bench/sweep_decode_runtime.py). Each
# context is the capacity of a native configuration; its builds are made first.
CONTEXTS ?= 2048
profile-decode: weights compile-all
	$(DRIVER) profile-decode $(OPTIONS) --contexts $(CONTEXTS)
compile-all verify-all profile-all:
	@set -e; for p in $(NATIVE_LENGTHS); do $(MAKE) -f $(srcdir)/Makefile \
	  $(patsubst verify,$(VERIFY_NATIVE),$(@:-all=)) P=$$p; done

# CI uses a source-bound shared build and unique evidence directories.
# The profile target performs its own correctness gate before measuring.
CI_DIR ?= $(abspath $(BUILD_DIR))/ci
.PHONY: ci-verify ci-profile
ci-verify ci-profile:
	cd $(srcdir)/.. && $(PYTHON) -m shared.deployment_ci $(@:ci-%=%) \
	  --model $(MODEL_MODULE) --prefill $(P) --work-dir "$(abspath $(CI_DIR))" \
	  $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)") \
	  $(if $(Q4_DIR),--q4-dir "$(Q4_DIR)")

# Decode templates at any KV capacity, for bench/sweep_decode.py: decode_L$(LBUILD)
# and decode_L$(LBUILD-1) in DECODE_SWEEP_DIR (its --template-dir).
DECODE_SWEEP_DIR ?= $(abspath $(BUILD_DIR))/sweep-decode
.PHONY: _compile_decode_build
_compile_decode_build:
	$(DRIVER) compile --part decode --capacity $(LBUILD) --out "$(DECODE_SWEEP_DIR)"

ifdef RUNNER_ADAPTER
# The verify/ subsystem through verify_adapter.py: `verify` (top-5 gate on its
# prompts), `accuracy` (verify/accuracy.mk) and `tasks`. The adapter runs the
# session of the native configuration whose KV capacity covers the tool's
# sequence length, so build that configuration first (P=1024 for 2048).
.PHONY: verify-native
VERIFY_RUNNER := $(srcdir)/../verify/verify_runner.py
MODEL_CHOICE ?= instruct
MAX_PROMPTS ?= 2
VERIFY_ENV = AIR_MODEL_BUILD_DIR="$(abspath $(BUILD_DIR))" \
  $(if $(Q4_DIR),AIR_MODEL_Q4_DIR="$(abspath $(Q4_DIR))") \
  $(if $(CHECKPOINT),AIR_MODEL_CHECKPOINT="$(abspath $(CHECKPOINT))")
verify:
	cd $(srcdir)/.. && $(VERIFY_ENV) $(PYTHON) $(VERIFY_RUNNER) \
	  --runner=$(RUNNER_ADAPTER) --prompts topk_token --model $(MODEL_CHOICE) \
	  --max-prompts $(MAX_PROMPTS)
ACC_MODEL := $(MODEL_CHOICE)
ACC_DIR := $(srcdir)/..
ACC_ENV = $(VERIFY_ENV)
include $(srcdir)/../verify/accuracy.mk
FUSED_PREFILL := 1
endif

# A model on the shared fused prefill without RUNNER_ADAPTER sets FUSED_PREFILL.
ifdef FUSED_PREFILL
# The fused prefill's own targets (fused_prefill.mk), on the logical Q4.
.PHONY: weights-prefill
weights-prefill:
	$(DRIVER) weights $(OPTIONS) --part prefill
FUSED_MODEL := $(MODEL_MODULE)
MODEL_SOURCE = $(if $(Q4_DIR),$(abspath $(Q4_DIR)),$(abspath $(BUILD_DIR))/logical-q4)
FUSED_DIR ?= $(abspath $(BUILD_DIR))/fused_prefill
include $(srcdir)/../shared/fused_prefill/fused_prefill.mk
profile-fused-prefill: weights-prefill
endif

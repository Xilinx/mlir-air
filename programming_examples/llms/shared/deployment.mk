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
        compile-head reference verify run profile compile-all verify-all profile-all
.DEFAULT_GOAL := help
help:
	@echo "$(MODEL_MODULE): affine Q4, BFP16 prefill GEMM, NPU2"
	@echo "make weights [CHECKPOINT=/path/to/HF] [Q4_DIR=/path/to/canonical-q4]"
	@echo "make compile P=512  (native P values: $(NATIVE_LENGTHS))"
	@echo "make reference P=512 CHECKPOINT=/path/to/HF"
	@echo "make verify P=512 REFERENCE_DIR=/path/to/reference"
	@echo "make run P=512 N_TOKENS=32; make profile P=512"
	@echo "compile-all / verify-all / profile-all visit every native configuration"
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
verify:
	$(DRIVER) verify $(OPTIONS) --reference-dir "$(REFERENCE_DIR)"
run:
	$(DRIVER) run $(OPTIONS) --outputs $(N_TOKENS)
profile:
	$(DRIVER) profile $(OPTIONS)
compile-all verify-all profile-all:
	@set -e; for p in $(NATIVE_LENGTHS); do $(MAKE) -f $(srcdir)/Makefile $(@:-all=) P=$$p; done

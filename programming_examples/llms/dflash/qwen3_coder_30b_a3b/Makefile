# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
BUILD_DIR ?= $(srcdir)/build
override BUILD_DIR := $(abspath $(BUILD_DIR))
# cache reads the verify pass is built for; a block uses the shortest that covers it
READS ?= 256,512,1024,2048,4096
N_TOKENS ?= 256
PROMPT ?= Write a Python function that checks if a number is prime.
# Also lets the builders compile on hosts without device access.
export AIR_TARGET_DEVICE := npu2

.NOTPARALLEL:
.PHONY: help weights compile compile-verify compile-drafter verify check run clean
help:
	@echo 'Qwen3-Coder-30B-A3B DFlash B8 on NPU2'
	@echo 'make weights   Pack the z-lab drafter for the shared B8 decoder'
	@echo 'make compile   Build the verify pass (every read), its LM head and the drafter'
	@echo 'make verify    48-layer verify pass against the CPU reference greedy'
	@echo 'make check     64 DFlash tokens against the CPU reference greedy'
	@echo 'make run       Generate; PROMPT="..." N_TOKENS=256'
	@echo 'make clean     Remove the build directory'

weights:
	cd $(srcdir) && $(PYTHON) prepare_coder_draft.py

compile: compile-verify compile-drafter

compile-verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/verify_b8.py --compile-only --root . --reads $(READS)

compile-drafter:
	@mkdir -p $(BUILD_DIR)
	if [ ! -f $(BUILD_DIR)/drafter/manifest.json ] || ! grep -q '"compiled"' $(BUILD_DIR)/drafter/manifest.json; then \
		rm -rf $(BUILD_DIR)/drafter; \
		cd $(srcdir) && $(PYTHON) build_target.py --output $(BUILD_DIR)/drafter --role drafter --block-size 8; \
	fi

verify:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/verify_b8.py --root . --reads 256

check:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/dflash_b8.py --root . --reads $(READS) --gen 64 --check

run:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/dflash_b8.py --root . --reads $(READS) --gen $(N_TOKENS) --prompt "$(PROMPT)"

clean:
	rm -rf $(BUILD_DIR)

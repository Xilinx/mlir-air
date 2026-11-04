# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GPT-OSS-20B Q4NX decode on NPU2 (MLIR-AIR), FastFlowLM's layout: all 24
# layers and the LM head in one dispatch, the router choosing and fetching the
# experts on device. Weights come from FastFlowLM's model.q4nx bundle
# (FastFlowLM/GPT-OSS-20B-NPU2; GPT_OSS_FLM_DIR names a local copy).
#
#   make run      Greedy decode of PROMPT, checked against the CPU reference
#   make bench    Time BENCH dispatches of the full decode
#   make block    The MoE block alone, against its CPU pipeline
#   make kernels  Every kernel test
#
# The 24-layer decode keeps every weight resident: it needs a memlock limit
# above the bundle's size (ulimit -l).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3

MAXL   ?= 256
STEPS  ?= 8
BENCH  ?= 30
PROMPT ?= The capital of France is
BUILD_DIR ?= build_decode

.PHONY: run bench block kernels clean

run:
	$(PYTHON) $(srcdir)/decode.py --maxl $(MAXL) --steps $(STEPS) \
		--prompt "$(PROMPT)" --build-dir $(BUILD_DIR)

bench:
	$(PYTHON) $(srcdir)/decode.py --maxl $(MAXL) --bench $(BENCH) \
		--build-dir $(BUILD_DIR)

block:
	$(PYTHON) $(srcdir)/moe_block.py
	$(PYTHON) $(srcdir)/moe_block.py --output-format elf

kernels:
	for t in router mvm_mxfp4 swiglu attn proj_q4; do \
		$(PYTHON) $(srcdir)/kernel_tests/$$t.py || exit 1; done
	$(PYTHON) $(srcdir)/kernel_tests/rope_rms.py rope
	$(PYTHON) $(srcdir)/kernel_tests/rope_rms.py rms
	$(PYTHON) $(srcdir)/attn_block.py

clean:
	rm -rf $(BUILD_DIR) air_project *.o *.elf *.xclbin *.bin

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GPT-OSS-20B Q4NX decode on NPU2 (MLIR-AIR), FastFlowLM's layout: all 24
# layers and the LM head in one dispatch, the router choosing and fetching the
# experts on device. Weights come from FastFlowLM's model.q4nx bundle
# (FastFlowLM/GPT-OSS-20B-NPU2; GPT_OSS_FLM_DIR names a local copy).
#
#   make compile         Build the decode ELF
#   make run             Greedy decode of PROMPT, checked against the CPU reference
#   make verify          MoE block, decode vs the CPU reference, reference vs HF
#   make profile         TTFT and decode tok/s, no reference check
#   make profile-decode  Decode tok/s at each of CONTEXTS (one build each)
#   make profile-prefill TTFT of a CTX-token prompt
#   make kernels         Every kernel test
#
# There is no prefill yet: the prompt goes through the decode loop, one token
# per dispatch, and TTFT is the time of those dispatches.
#
# The 24-layer decode keeps every weight resident: it needs a memlock limit
# above the bundle's size (ulimit -l).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
DECODE := $(PYTHON) $(srcdir)/decode.py
comma := ,

MAXL     ?= 256
STEPS    ?= 8
N_TOKENS ?= 32
BENCH    ?= 20
PROMPT   ?= The capital of France is
CONTEXTS ?= 1024,4096,8192
CTX      ?= 512
BUILD_DIR ?= build_decode

.PHONY: compile run verify profile profile-decode profile-prefill bench block \
	kernels clean

compile:
	$(DECODE) --compile-only --maxl $(MAXL) --build-dir $(BUILD_DIR)

run:
	$(DECODE) --maxl $(MAXL) --steps $(STEPS) --prompt "$(PROMPT)" \
		--build-dir $(BUILD_DIR)

verify: block run
	$(PYTHON) $(srcdir)/check_vs_hf.py forward --prompt "$(PROMPT)"

profile:
	$(DECODE) --maxl $(MAXL) --steps $(N_TOKENS) --prompt "$(PROMPT)" --no-ref \
		--build-dir $(BUILD_DIR)

# The context is compiled in: one build per point, each at its own maxl.
profile-decode:
	for c in $(subst $(comma), ,$(CONTEXTS)); do \
		$(DECODE) --maxl $$c --bench $(BENCH) --bench-ctx $$c \
			--build-dir build_ctx$$c || exit 1; done

profile-prefill:
	$(DECODE) --maxl $$(( ($(CTX) + 31) / 16 * 16 )) --prompt-len $(CTX) \
		--steps 2 --no-ref --build-dir build_ctx$(CTX)

bench:
	$(DECODE) --maxl $(MAXL) --bench $(BENCH) --build-dir $(BUILD_DIR)

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
	rm -rf $(BUILD_DIR) build_ctx* air_project *.o *.elf *.xclbin *.bin

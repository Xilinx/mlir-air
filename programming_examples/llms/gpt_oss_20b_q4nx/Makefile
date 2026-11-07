# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GPT-OSS-20B Q4NX decode on NPU2 (MLIR-AIR), FastFlowLM's layout: all 24
# layers and the LM head in one dispatch, the router choosing and fetching the
# experts on device. Weights come from FastFlowLM's model.q4nx bundle
# (FastFlowLM/GPT-OSS-20B-NPU2; GPT_OSS_FLM_DIR names a local copy).
#
#   make compile         Build the decode ELF
#   make compile-fused-prefill  Build the fused prefill device into FUSED_DIR
#   make run             Greedy decode of PROMPT, checked against the CPU reference
#   make prefill-fused   The fused prefill of PROMPT repeated to LONG_LEN tokens
#                        vs the CPU reference
#   make decode-prefilled  That prompt prefilled into the decode, its first token
#                        vs the one prefill-fused's reference chose
#   make run-long        run after a LONG_LEN-token prompt is prefilled (holds the
#                        decode's weights and the CPU reference at once)
#   make verify          MoE block, run, prefill-fused, decode-prefilled,
#                        reference vs HF
#   make profile         TTFT and decode tok/s, no reference check
#   make profile-decode  Decode tok/s at each of CONTEXTS (one build each)
#   make profile-prefill TTFT of a CTX-token prompt
#   make prefill         The per-op prefill's NPU ops, then it vs the CPU reference
#   make kernels         Every kernel test
#
# A prompt of MIN_PREFILL tokens or more is prefilled on the shared fused
# prefill device (fused_spec.py, prefill_fused.py): one PDI for every op, so a
# dispatch costs its insts rather than a full ELF's. Its KV goes to the decode.
# FUSED_PREFILL_DIR= prefills with FastFlowLM's per-op ELFs instead
# (prefill.py). A shorter prompt goes through the decode loop.
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
LONG_LEN ?= 128

# The fused prefill build, under the invoking directory; decode.py prefills on
# the build in FUSED_PREFILL_DIR, and with FUSED_PREFILL_DIR= on per-op ELFs.
FUSED_DIR ?= $(CURDIR)/_fused_prefill
override FUSED_DIR := $(abspath $(FUSED_DIR))
FUSED_JOBS ?= 4
FUSED_PREFILL_DIR ?= $(FUSED_DIR)
export FUSED_PREFILL_DIR
# the build the decode uses, made once for the targets that may prefill;
# compile-fused-prefill rebuilds FUSED_DIR
FUSED_DEP := $(if $(FUSED_PREFILL_DIR),$(abspath $(FUSED_PREFILL_DIR))/manifest.json)

.PHONY: compile compile-fused-prefill run run-long verify profile profile-decode \
	profile-prefill prefill prefill-fused decode-prefilled bench block kernels \
	clean

compile:
	$(DECODE) --compile-only --maxl $(MAXL) --build-dir $(BUILD_DIR)

compile-fused-prefill:
	$(PYTHON) $(srcdir)/fused_build.py $(FUSED_DIR) -j $(FUSED_JOBS)

%/manifest.json:
	$(PYTHON) $(srcdir)/fused_build.py $* -j $(FUSED_JOBS)

run: $(FUSED_DEP)
	$(DECODE) --maxl $(MAXL) --steps $(STEPS) --prompt "$(PROMPT)" \
		--build-dir $(BUILD_DIR)

# PROMPT repeated to LONG_LEN tokens: prefilled, then decoded, each token
# checked against the CPU reference
run-long: $(FUSED_DEP)
	$(DECODE) --maxl $$(( ($(LONG_LEN) + $(STEPS) + 15) / 16 * 16 )) \
		--steps $(STEPS) --prompt "$(PROMPT)" --prompt-len $(LONG_LEN) \
		--build-dir build_long

verify: block run decode-prefilled
	$(PYTHON) $(srcdir)/check_vs_hf.py forward --prompt "$(PROMPT)"

prefill-fused: $(FUSED_DIR)/manifest.json
	$(PYTHON) $(srcdir)/prefill_fused.py $(FUSED_DIR) --prompt "$(PROMPT)" \
		--prompt-len $(LONG_LEN) --token-out $(CURDIR)/first_token.txt

# the decode's resident weights, the fused prefill's KV handed to it; the
# reference token comes from prefill-fused, so the two never share memory
decode-prefilled: prefill-fused
	FUSED_PREFILL_DIR=$(FUSED_DIR) $(DECODE) \
		--maxl $$(( ($(LONG_LEN) + $(STEPS) + 15) / 16 * 16 )) --steps $(STEPS) \
		--prompt "$(PROMPT)" --prompt-len $(LONG_LEN) --no-ref \
		--first-token $(CURDIR)/first_token.txt --build-dir build_long

profile: $(FUSED_DEP)
	$(DECODE) --maxl $(MAXL) --steps $(N_TOKENS) --prompt "$(PROMPT)" --no-ref \
		--build-dir $(BUILD_DIR)

# The context is compiled in: one build per point, each at its own maxl.
profile-decode:
	for c in $(subst $(comma), ,$(CONTEXTS)); do \
		$(DECODE) --maxl $$c --bench $(BENCH) --bench-ctx $$c \
			--build-dir build_ctx$$c || exit 1; done

profile-prefill: $(FUSED_DEP)
	$(DECODE) --maxl $$(( ($(CTX) + 31) / 16 * 16 )) --prompt-len $(CTX) \
		--steps 2 --no-ref --warm --build-dir build_ctx$(CTX)

prefill:
	$(PYTHON) $(srcdir)/prefill_gemm.py --build-dir build_prefill_ops
	$(PYTHON) $(srcdir)/prefill_gemm.py --build-dir build_prefill_ops --tm 32 \
		--max-m 128 --n 3072 --m 128,40
	$(PYTHON) $(srcdir)/prefill_dequant.py
	$(PYTHON) $(srcdir)/prefill_attn.py
	$(PYTHON) $(srcdir)/prefill_attn.py --window 128 --rounds 0,1,3
	$(PYTHON) $(srcdir)/prefill.py --npu --build-dir build_prefill_ops

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
	rm -rf $(BUILD_DIR) build_ctx* build_prefill* build_long $(FUSED_DIR) \
		first_token.txt air_project *.o *.elf *.xclbin *.bin

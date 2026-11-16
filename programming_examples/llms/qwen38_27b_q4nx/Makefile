# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.8-27B Q4NX decode on NPU2 (MLIR-AIR), FastFlowLM's layout: the 64
# layers in one dispatch and the Q8 LM head in a second. Weights come from
# FastFlowLM's model.q4nx bundle (FastFlowLM/Qwen3.8-27B-NPU2; QWEN38_FLM_DIR
# names a local copy).
#
#   make compile         Build the decode and LM head ELFs
#   make compile-fused-prefill  Build the fused prefill device into FUSED_DIR
#   make run             Greedy decode of PROMPT, checked against the CPU reference
#   make prefill-fused   The fused prefill of PROMPT repeated to LONG_LEN tokens
#                        vs the CPU reference
#   make verify          Each stage alone, prefill-fused, the decode, the
#                        decode across a READS boundary, the reference against
#                        HF transformers' Qwen3.5 layers, and the shared top-5
#                        gate against Qwen/Qwen3.8-27B in bf16
#   make profile         TTFT and decode throughput of PROMPT, no reference
#   make profile-decode  Decode tok/s at each of CONTEXTS (one build each)
#   make profile-prefill TTFT of a CTX-token prompt
#   make kernels         Every kernel test
#   make spec            Self-speculative greedy decode of PROMPT with the MTP
#                        head (qwen38_27b_q4nx_spec.py), checked token for
#                        token against the plain decode
#   make profile-spec    Its tok/s, draft acceptance and step times
#   make verify-spec     Its stages alone (the two-token MLP, the verify pass,
#                        the MTP head, the two-row LM head), then make spec
#
# The prompt is prefilled on the shared fused prefill device (fused_spec.py,
# prefill.py) and handed to the decode, whose LM head gives the first token;
# with FUSED_PREFILL_DIR= it goes through the decode a token at a time.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
DRIVER := $(PYTHON) $(srcdir)/qwen38_27b_q4nx_inference.py
SPEC   := $(PYTHON) $(srcdir)/qwen38_27b_q4nx_spec.py
comma := ,

MAXL      ?= 64
# shorter cache reads below MAXL, each in a layers' ELF of its own
READS     ?=
GEN       ?= 8
PROMPT    ?= The capital of France
N_TOKENS  ?= 32
CTX       ?= 1024
CONTEXTS  ?= 1024,4096,8192
BENCH     ?= 20
BUILD_DIR ?= build_decode
LONG_LEN  ?= 300
MODEL_CHOICE ?= instruct
MAX_PROMPTS  ?= 2

FUSED_DIR ?= $(CURDIR)/_fused_prefill
override FUSED_DIR := $(abspath $(FUSED_DIR))
FUSED_PREFILL_DIR ?= $(FUSED_DIR)
# absolute, as the recipes run in their build directories; empty opts out
override FUSED_PREFILL_DIR := $(if $(FUSED_PREFILL_DIR),$(abspath $(FUSED_PREFILL_DIR)))
export FUSED_PREFILL_DIR
FUSED_DEP := $(if $(FUSED_PREFILL_DIR),$(abspath $(FUSED_PREFILL_DIR))/manifest.json)
STAGES    := mlp_block gdn_layer attn_layer lm_head
KERNELS   := q4k_mvm gdn_prep gdn_decode gate rope attn q4k_mvm2 gdn_prep2
SPEC_STAGES := mlp_block2 qwen38_27b_q4nx_verify mtp_draft

.PHONY: compile compile-fused-prefill run prefill-fused verify profile \
	profile-decode profile-prefill kernels spec profile-spec verify-spec clean

compile:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		$(DRIVER) --compile-only --maxl $(MAXL) --reads "$(READS)"

compile-fused-prefill:
	$(PYTHON) $(srcdir)/fused_build.py $(FUSED_DIR)

%/manifest.json:
	$(PYTHON) $(srcdir)/fused_build.py $*

run: $(FUSED_DEP)
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		$(DRIVER) --maxl $(MAXL) --reads "$(READS)" --gen $(GEN) --prompt "$(PROMPT)"

prefill-fused: $(FUSED_DIR)/manifest.json
	$(PYTHON) $(srcdir)/prefill.py $(FUSED_DIR) --prompt "$(PROMPT)" \
		--prompt-len $(LONG_LEN)

verify:
	for s in $(STAGES); do \
		mkdir -p build_$$s && (cd build_$$s && $(PYTHON) $(srcdir)/$$s.py) \
			|| exit 1; done
	$(MAKE) -f $(srcdir)/Makefile prefill-fused
	$(MAKE) -f $(srcdir)/Makefile run GEN=4
	$(MAKE) -f $(srcdir)/Makefile run MAXL=64 READS=16 GEN=14 FUSED_PREFILL_DIR= \
		BUILD_DIR=build_reads
	$(PYTHON) $(srcdir)/check_vs_hf.py
	cd $(srcdir)/.. && QWEN38_VERIFY_DIR=$(CURDIR)/build_verify \
		$(PYTHON) verify/verify_runner.py \
		--runner=qwen38_27b_q4nx.verify_adapter --prompts topk_token \
		--model $(MODEL_CHOICE) --max-prompts $(MAX_PROMPTS)

# The context the build holds covers the prompt and the generated tokens.
profile: $(FUSED_DEP)
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		$(DRIVER) --maxl $$(( ($(N_TOKENS) + 64 + 15) / 16 * 16 )) \
			--gen $(N_TOKENS) --no-ref --prompt "$(PROMPT)"

# The context is compiled in: one build per point, each at its own maxl.
profile-decode:
	for c in $(subst $(comma), ,$(CONTEXTS)); do \
		mkdir -p build_ctx$$c && (cd build_ctx$$c && \
			$(DRIVER) --maxl $$c --gen 2 --no-ref --bench $(BENCH) --bench-l $$c) \
			|| exit 1; done

profile-prefill: $(FUSED_DEP)
	mkdir -p build_ctx$(CTX) && cd build_ctx$(CTX) && \
		$(DRIVER) --maxl $$(( ($(CTX) + 2 + 15) / 16 * 16 )) --prompt-len $(CTX) \
			--gen 2 --no-ref --warm

kernels:
	mkdir -p build_kernels && cd build_kernels && \
		for k in $(KERNELS); do \
			$(PYTHON) $(srcdir)/kernel_tests/$$k.py || exit 1; done

# the build holds the prompt, the generated tokens and a step's two
spec: $(FUSED_DEP)
	mkdir -p build_spec && cd build_spec && \
		$(SPEC) --maxl $$(( ($(N_TOKENS) + 64 + 15) / 16 * 16 )) \
			--gen $(N_TOKENS) --prompt "$(PROMPT)" --check

profile-spec: $(FUSED_DEP)
	mkdir -p build_spec && cd build_spec && \
		$(SPEC) --maxl $$(( ($(N_TOKENS) + 64 + 15) / 16 * 16 )) \
			--gen $(N_TOKENS) --prompt "$(PROMPT)"

verify-spec:
	for s in $(SPEC_STAGES); do \
		mkdir -p build_$$s && (cd build_$$s && $(PYTHON) $(srcdir)/$$s.py) \
			|| exit 1; done
	mkdir -p build_lm_head2 && cd build_lm_head2 && \
		$(PYTHON) $(srcdir)/lm_head.py --tokens 9338,760
	$(MAKE) -f $(srcdir)/Makefile spec

clean:
	rm -rf $(BUILD_DIR) build_* $(FUSED_DIR) *.o

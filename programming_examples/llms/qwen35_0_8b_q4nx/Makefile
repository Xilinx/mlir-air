# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.5-0.8B (text) with FastFlowLM's Q4NX weights on NPU2: prefill on the
# shared fused_prefill device (shared/fused_prefill/qwen35.py), decode as one
# dispatch per token on programming_examples/fused_decode_gdn (q35_fdg.py).
#
#   make compile         the prefill device and the decode ELF of every capacity
#   make run             greedy generation of PROMPT at capacity C
#   make verify          top-5 token gate against HF BF16 (the CI gate)
#   make accuracy        PPL + KL divergence against HF BF16 on wikitext-2 (C=2048)
#   make tasks           GSM8K / MMLU on the NPU and on HF BF16 (C=2048)
#   make profile         TTFT + decode throughput (bench/extract_perf.py lines)
#   make reference       CPU BF16 reference of the gate prompts
#   make gate            the model gate for one configuration (P, C); gate-all
#   make reference-timing, fullfit   the gate on the bucket-filling prompts
#   make bench           complete-request timing for one configuration; bench-all
#   make profile-decode  decode tok/s at each of CONTEXTS (bench/sweep_decode_runtime.py)
#
# Builds go to BUILD (default ./build). Weights: the pinned Hub revision of
# FastFlowLM/Qwen3.5-0.8B-NPU2 (cached under HF_HOME), or MODEL_SOURCE (a
# model.q4nx file or a directory holding it); either must match the pinned
# sha256. Serialize NPU runs (e.g. flock /tmp/mlir-air-npu.lock make ...).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
BUILD ?= $(CURDIR)/build
override BUILD := $(abspath $(BUILD))
OUT ?= $(CURDIR)/out
# Decode capacities (KV cache length, one ELF each) and the gate / timing
# configurations P/C: 512/1024, 1024/2048, 2048/4096, 4096/8192.
CAPACITIES ?= 1024 2048 4096 8192
# Decode contexts `make profile-decode` measures, comma-separated: one ELF of
# that capacity each (built if missing).
CONTEXTS ?= 2048,8192,32768
P ?= 512
C ?= 1024
PROMPT ?= Introduce me what is GPU
N_TOKENS ?= 32
REF ?= $(OUT)/ref
REF_TIMING ?= $(OUT)/ref_timing

MODEL_SOURCE ?=
ifneq ($(MODEL_SOURCE),)
export Q4NX_MODEL_SOURCE := $(MODEL_SOURCE)
endif
# read by verify_adapter.py
export Q35_BUILD := $(BUILD)
export Q35_CAPACITY := $(C)

VERIFY_RUNNER  := $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER := qwen35_0_8b_q4nx.verify_adapter
MODEL_CHOICE   ?= instruct
MAX_PROMPTS    ?= 2

FDG := $(srcdir)/../../fused_decode_gdn

.PHONY: help compile compile-fused-prefill compile-decode run verify verify-full profile \
        profile-fused-prefill profile-decode reference reference-timing gate gate-all fullfit fullfit-all bench \
        bench-all clean

help:
	@sed -n '4,23p' $(srcdir)/Makefile

compile: compile-fused-prefill compile-decode

## The prefill device (q35_spec.py): every GEMM, the attention and the
## lm_head as ops of one xclbin; the host libraries are built into it on
## first use. Takes prompts up to 4096 tokens (FUSED_MAX_LEN is accepted for
## bench/sweep_prefill.py and not needed).
FUSED_DIR ?= $(BUILD)/device
compile-fused-prefill:
	$(PYTHON) $(srcdir)/device_build.py $(FUSED_DIR) -j 4

## One fused_decode_gdn full ELF per capacity in $(BUILD)/fdg_<C>; the context
## length reaches the device as scratchpad parameters. Needs an llvm-link
## older than LLVM 23 on PATH (see ../../fused_decode/README.md).
compile-decode:
	for c in $(CAPACITIES); do \
	  $(MAKE) --no-print-directory -C $(FDG) compile-decode-elf BUILD=$(BUILD)/fdg_$$c LBUILD=$$c \
	    PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) || exit 1; done

run:
	$(PYTHON) $(srcdir)/q35_npu.py --build $(BUILD) --capacity $(C) \
	  --tokens $(N_TOKENS) --prompt "$(PROMPT)"

## Top-k token-set inclusion gate (NPU vs HF BF16 Qwen/Qwen3.5-0.8B at the
## pinned revision, 2 prompts x 32 greedy tokens, k=5) through the shared
## verify/ subsystem. Needs the prefill device and the decode of capacity C.
verify:
	cd $(srcdir)/.. && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL_CHOICE) --max-prompts $(MAX_PROMPTS)

## Full-sweep variant of `make verify` (every prompt in the prompt file).
verify-full:
	cd $(srcdir)/.. && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL_CHOICE)

## TTFT + decode throughput over N_TOKENS tokens after one warm-up request
## (prints the lines the nightly LLM benchmark extracts).
profile:
	$(PYTHON) $(srcdir)/q35_npu.py --build $(BUILD) --capacity $(C) \
	  --tokens $(N_TOKENS) --warmup 1 --prompt "$(PROMPT)"

## Prefill time at each of FUSED_LENGTHS tokens (bench/sweep_prefill.py
## --fused) on the prefill device in FUSED_DIR.
FUSED_LENGTHS ?= 512,1024,2048,4096
profile-fused-prefill:
	$(PYTHON) $(srcdir)/bench.py --prefill-only $(FUSED_LENGTHS) --device $(FUSED_DIR)

## Decode tok/s at each of CONTEXTS: the decode of capacity C stepping at
## position C - 1 over synthetic cache rows (bench.py --decode-only).
comma := ,
profile-decode:
	$(MAKE) --no-print-directory -f $(srcdir)/Makefile compile-decode \
	  CAPACITIES="$(subst $(comma), ,$(CONTEXTS))"
	$(PYTHON) $(srcdir)/bench.py --build $(BUILD) --decode-only $(CONTEXTS)

reference:
	$(PYTHON) $(srcdir)/reference.py --out $(REF)

reference-timing:
	$(PYTHON) $(srcdir)/reference.py --out $(REF_TIMING) --timing-prompts 512,1024,2048,4096

## The model gate: 8 instruct prompts, 32 greedy tokens, mutual top-5 at the
## first divergence, complete finite logits, at prefill bucket P / capacity C.
gate:
	$(PYTHON) $(srcdir)/gate.py --build $(BUILD) --reference $(REF) --prefill $(P) --capacity $(C) \
	  --out $(OUT)/gate/p$(P)_c$(C)

fullfit:
	$(PYTHON) $(srcdir)/gate.py --build $(BUILD) --reference $(REF_TIMING) --prefill $(P) --capacity $(C) \
	  --fit --out $(OUT)/fullfit/p$(P)_c$(C)

bench:
	$(PYTHON) $(srcdir)/bench.py --build $(BUILD) --prefill $(P) --capacity $(C) \
	  --out $(OUT)/timing/p$(P)_c$(C).json

CONFIGS := 512/1024 1024/2048 2048/4096 4096/8192
gate-all bench-all fullfit-all:
	for pc in $(CONFIGS); do \
	  $(MAKE) --no-print-directory -f $(srcdir)/Makefile $(@:-all=) P=$${pc%/*} C=$${pc#*/} || exit 1; done

clean:
	rm -rf $(BUILD) $(srcdir)/__pycache__

## `make accuracy` (teacher-forced PPL + KL divergence against HF BF16 on
## wikitext-2) and `make tasks` (GSM8K / MMLU scores on the NPU and on HF
## BF16), verify/accuracy.mk. Their 2048-token sequences need the decode of
## capacity C=2048 (`make compile CAPACITIES=2048`).
ACC_MODEL := $(MODEL_CHOICE)
ACC_DIR := $(srcdir)/..

include $(srcdir)/../verify/accuracy.mk

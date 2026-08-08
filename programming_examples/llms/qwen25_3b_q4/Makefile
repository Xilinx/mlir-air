# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen2.5-3B Q4_0 prefill on NPU2 (MLIR-AIR).
#
#   make compile   — build/cache the prefill ELFs (no weights, no NPU dispatch)
#   make run       — prefill "The capital of France is" -> expect " Paris"
#   make bench     — warm TTFT at BENCH_L (default 2048)

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

SEQ_LEN  ?= 2048
N_LAYERS ?= 36
BENCH_L  ?= 2048
MODEL    ?= Qwen/Qwen2.5-3B-Instruct

PREFILL := $(srcdir)/qwen25_3b_q4_prefill.py

.PHONY: help compile run bench weights clean

help:
	@echo "Qwen2.5-3B Q4_0 prefill (MLIR-AIR, NPU2)"
	@echo "  make compile   build/cache the prefill ELFs (CI-runnable, no weights)"
	@echo "  make run       prefill gate: 'The capital of France is' -> ' Paris'"
	@echo "  make bench     warm TTFT at BENCH_L=$(BENCH_L)"
	@echo "  make weights   quant-error report for layer 0"

## Build/cache the prefill ELFs and exit. No weights, no NPU dispatch.
compile:
	cd $(srcdir) && python3 $(PREFILL) --compile-only \
		--seq-len $(SEQ_LEN) --n-layers $(N_LAYERS)

## First-token gate.
run:
	cd $(srcdir) && python3 $(PREFILL) \
		--seq-len $(SEQ_LEN) --n-layers $(N_LAYERS) --model $(MODEL)

## Warm TTFT benchmark (+ per-ELF profile).
bench:
	cd $(srcdir) && python3 $(PREFILL) \
		--seq-len $(SEQ_LEN) --n-layers $(N_LAYERS) --model $(MODEL) \
		--bench-l $(BENCH_L)

## Q4_0 quant-error report for layer 0 (no NPU).
weights:
	cd $(srcdir) && python3 $(srcdir)/qwen25_3b_q4_weights.py --model $(MODEL)

clean:
	rm -rf $(srcdir)/_q4_cache_seq* $(srcdir)/__pycache__

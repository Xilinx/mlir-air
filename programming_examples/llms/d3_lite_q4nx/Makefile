# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# vllm-sr/d3-lite on NPU2 with FastFlowLM's Q4K weights (FastFlowLM/D3-Lite-NPU2):
# one forward pass per System One question on the shared fused_prefill device
# (shared/fused_prefill/d3.py), the 255-way FP32 readout of the last row and
# the option softmax on the host. The runtime, gate, reference, timing and
# request set are d3_edge's, run with --model lite.
#
#   make compile         the prefill device
#   make run             a System One example
#   make verify          the d3 decision gate against the CPU reference (the CI gate)
#   make reference       CPU reference of the gate's request set (upstream runtime,
#                        the bundle's Q4K / Q8 values dequantized)
#   make gate            the decision gate against REF
#   make bench           single-question latency at P = 512 ... 4096
#
# Builds go to BUILD (default ./build). Weights: the pinned Hub revision of
# FastFlowLM/D3-Lite-NPU2 (cached under HF_HOME), or MODEL_SOURCE (a
# model.q4nx file or a directory holding it); either must match the pinned
# sha256. Serialize NPU runs (e.g. flock /tmp/mlir-air-npu.lock make ...).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
D3 := $(abspath $(srcdir)/../d3_edge)
PYTHON ?= python3
BUILD ?= $(CURDIR)/build
override BUILD := $(abspath $(BUILD))
OUT ?= $(CURDIR)/out
FUSED_DIR ?= $(BUILD)/device
REF ?= $(OUT)/reference_lite.json
LENGTHS ?= 512,1024,2048,4096
JOBS ?= 4

MODEL_SOURCE ?=
ifneq ($(MODEL_SOURCE),)
export D3_LITE_SOURCE := $(MODEL_SOURCE)
endif

MODEL := --build $(FUSED_DIR) --model lite

.PHONY: help compile compile-fused-prefill run verify reference gate bench clean

help:
	@sed -n '4,21p' $(srcdir)/Makefile

compile: compile-fused-prefill

## The prefill device (d3_spec.py): every projection GEMM and the noncausal
## attention as ops of one xclbin; the host libraries are built into it on
## first use. Takes prompts up to 4096 tokens.
compile-fused-prefill:
	$(PYTHON) $(srcdir)/device_build.py $(FUSED_DIR) -j $(JOBS)

run:
	$(PYTHON) $(D3)/d3_npu.py $(MODEL)

## The d3 decision gate (d3_edge/gate.py) on d3_edge/requests.json against
## REF, made by `make reference` when it does not exist.
verify:
	test -f $(REF) || $(MAKE) --no-print-directory -f $(srcdir)/Makefile reference
	$(MAKE) --no-print-directory -f $(srcdir)/Makefile gate

reference:
	mkdir -p $(OUT)
	$(PYTHON) $(D3)/reference.py --model lite --out $(REF)

gate:
	mkdir -p $(OUT)
	$(PYTHON) $(D3)/gate.py $(MODEL) --reference $(REF) --out $(OUT)/gate_lite.json

bench:
	mkdir -p $(OUT)
	$(PYTHON) $(D3)/bench.py $(MODEL) --lengths $(LENGTHS) --out $(OUT)/timing_lite.json

clean:
	rm -rf $(BUILD) $(OUT)

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# vllm-sr/d3-edge (BF16) on NPU2: one forward pass per System One question on
# the shared fused_prefill device (shared/fused_prefill/d3.py), the 255-way
# FP32 readout of the last row and the option softmax on the host.
#
#   make compile         the prefill device
#   make run             a System One example
#   make verify          the d3 decision gate against the CPU reference (the CI gate)
#   make reference       CPU reference of the gate's request set (upstream runtime, BF16)
#   make gate            the decision gate against REF
#   make bench           single-question latency at P = 512 ... 4096
#   make profile-fused-prefill   the same at FUSED_LENGTHS, for bench/sweep_prefill.py
#
# WFORM is the weight form, hilo:gu,down (default) or hilo. Builds go to BUILD
# (default ./build). Weights: the pinned Hub revision of vllm-sr/d3-edge
# (cached under HF_HOME), or MODEL_SOURCE (a directory holding the package);
# either must match the pinned sha256. Serialize NPU runs (e.g. flock
# /tmp/mlir-air-npu.lock make ...).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
BUILD ?= $(CURDIR)/build
override BUILD := $(abspath $(BUILD))
OUT ?= $(CURDIR)/out
FUSED_DIR ?= $(BUILD)/device
WFORM ?=
REF ?= $(OUT)/reference_edge.json
LENGTHS ?= 512,1024,2048,4096
FUSED_LENGTHS ?= $(LENGTHS)
JOBS ?= 4

MODEL_SOURCE ?=
ifneq ($(MODEL_SOURCE),)
export D3_EDGE_SOURCE := $(MODEL_SOURCE)
endif

MODEL := --build $(FUSED_DIR) --model edge $(if $(WFORM),--wform $(WFORM))
comma := ,
TAG := edge$(if $(WFORM),_$(subst :,_,$(subst $(comma),_,$(WFORM))))

.PHONY: help compile compile-fused-prefill profile-fused-prefill run verify reference gate \
        bench clean

help:
	@sed -n '4,20p' $(srcdir)/Makefile

compile: compile-fused-prefill

## The prefill device (d3_spec.py): every projection GEMM and the noncausal
## attention as ops of one xclbin; the host libraries are built into it on
## first use. Takes prompts up to 4096 tokens (FUSED_MAX_LEN is accepted for
## bench/sweep_prefill.py and not needed).
compile-fused-prefill:
	$(PYTHON) $(srcdir)/device_build.py $(FUSED_DIR) -j $(JOBS)

run:
	$(PYTHON) $(srcdir)/d3_npu.py $(MODEL)

## The d3 decision gate (gate.py) on requests.json against REF, made by
## `make reference` when it does not exist.
verify:
	test -f $(REF) || $(MAKE) --no-print-directory -f $(srcdir)/Makefile reference
	$(MAKE) --no-print-directory -f $(srcdir)/Makefile gate

reference:
	mkdir -p $(OUT)
	$(PYTHON) $(srcdir)/reference.py --model edge --out $(REF)

gate:
	mkdir -p $(OUT)
	$(PYTHON) $(srcdir)/gate.py $(MODEL) --reference $(REF) --out $(OUT)/gate_$(TAG).json

bench:
	mkdir -p $(OUT)
	$(PYTHON) $(srcdir)/bench.py $(MODEL) --lengths $(LENGTHS) --out $(OUT)/timing_$(TAG).json

## Single-question latency at each of FUSED_LENGTHS on the device in FUSED_DIR
## (bench/sweep_prefill.py --fused).
profile-fused-prefill:
	mkdir -p $(OUT)
	$(PYTHON) $(srcdir)/bench.py $(MODEL) --lengths $(FUSED_LENGTHS) \
	  --out $(OUT)/prefill_sweep_$(TAG).json

clean:
	rm -rf $(BUILD) $(OUT)

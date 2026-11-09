# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Whisper large-v3-turbo (speech to text) with FastFlowLM's Q4NX weights on
# NPU2: every GEMM on the shared fused_prefill device (whisper_spec.py), the
# encoder's attention on its own FlashAttention ELF (whisper_attn.py).
#
#   make compile      the device and the attention ELF
#   make run          transcribe AUDIO (16 kHz, at most 30 s)
#   make verify       top-5 token gate against HF BF16 on the first N_VERIFY clips
#   make verify-full  the gate on every clip
#   make profile      complete-request timing on the first N_VERIFY clips
#   make reference    CPU BF16 reference of the gate's clips
#
# Builds go to BUILD (default ./build). Weights: the pinned Hub revision of
# FastFlowLM/Whisper-V3-Turbo-NPU2 (cached under HF_HOME), or MODEL_SOURCE (a
# model.q4nx file or a directory holding it); either must match the pinned
# sha256. Serialize NPU runs (e.g. flock /tmp/mlir-air-npu.lock make ...).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
BUILD ?= $(CURDIR)/build
override BUILD := $(abspath $(BUILD))
OUT ?= $(CURDIR)/out
REF ?= $(OUT)/ref
AUDIO ?=
N_VERIFY ?= 8

MODEL_SOURCE ?=
ifneq ($(MODEL_SOURCE),)
export Q4NX_MODEL_SOURCE := $(MODEL_SOURCE)
endif

.PHONY: help compile run verify verify-full profile reference clean

help:
	@sed -n 's/^#   make/make/p' $(srcdir)/Makefile

compile:
	$(PYTHON) $(srcdir)/device_build.py $(BUILD)/device -j 4
	@echo "Compilation passed."

run:
	@test -n "$(AUDIO)" || (echo "set AUDIO=<file>"; exit 1)
	$(PYTHON) $(srcdir)/whisper.py --build $(BUILD)/device $(AUDIO)

## The reference of the clips the gate reads, written once into REF.
$(REF)/reference.json:
	$(PYTHON) $(srcdir)/reference.py --out $(REF)

reference:
	$(PYTHON) $(srcdir)/reference.py --out $(REF)

verify: $(REF)/reference.json
	$(PYTHON) $(srcdir)/gate.py --build $(BUILD)/device --reference $(REF) \
	  --clips $(N_VERIFY) --out $(OUT)/gate

verify-full: $(REF)/reference.json
	$(PYTHON) $(srcdir)/gate.py --build $(BUILD)/device --reference $(REF) \
	  --out $(OUT)/gate_full

profile:
	$(PYTHON) $(srcdir)/bench.py --build $(BUILD)/device --clips $(N_VERIFY) \
	  --out $(OUT)/timing.json

clean:
	rm -rf $(BUILD) $(OUT)

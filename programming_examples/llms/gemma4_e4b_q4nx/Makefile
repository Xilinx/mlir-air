# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Gemma4-E4B (text) Q4NX on NPU2 (MLIR-AIR).
#
# The E2B example's code and recipes, run with GEMMA4_VARIANT=e4b: every target
# is ../gemma4_e2b_q4nx/Makefile's, and the artifacts land in this directory.
# `make help` lists them. See README.md for what differs from E2B.

GEMMA4_VARIANT := e4b
DECODE_MODEL   := gemma4-e4b
MODEL_TYPE     := GEMMA4_E4B
VOCAB_CHUNK_I2 := 5
MODEL_SOURCE   ?= FastFlowLM/Gemma4-E4B-IT-NPU2

# 42 decoder layers. The gate scores one layer per class: sliding / full with
# their own KV, and the KV-shared ones (>= 24) against their sources, 22
# (sliding) and 23 (full).
SWEEP_UNI_DEC ?= 42
LAYERS        ?= 42
GATE_LAYERS   ?= 0 5 23
GATE_PAIRS    ?= 22:24 23:29 23:41

include $(dir $(realpath $(lastword $(MAKEFILE_LIST))))../gemma4_e2b_q4nx/Makefile

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
MODEL_MODULE := openelm_1b_q4nx
NATIVE_LENGTHS := 512 1024
# Prefill on the shared fused prefill. The checkpoint has no tokenizer and its
# HF model is remote code, which the verify/ tools do not load, so the
# fixed-input gate stays `verify`.
FUSED_PREFILL := 1
include $(srcdir)/../shared/deployment.mk

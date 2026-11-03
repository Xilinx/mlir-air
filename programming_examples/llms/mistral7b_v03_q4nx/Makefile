# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
MODEL_MODULE := mistral7b_v03_q4nx
NATIVE_LENGTHS := 512 1024 2048 4096
# Prefill on the shared fused prefill; verify/ through this adapter.
RUNNER_ADAPTER := $(MODEL_MODULE).verify_adapter
include $(srcdir)/../shared/deployment.mk

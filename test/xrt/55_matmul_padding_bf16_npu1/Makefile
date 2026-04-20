# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
BUILD_DIR := build

M ?= 500
N ?= 500
K ?= 784
K_L2_TILE ?= 16
HERD_M ?= 4
HERD_N ?= 4

COMMON_FLAGS = --k-l2-tile $(K_L2_TILE) --herd-m $(HERD_M) --herd-n $(HERD_N)

all: run

run:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 ${srcdir}/run.py $(COMMON_FLAGS) \
		--M $(M) --N $(N) --K $(K) -v

clean:
	rm -rf $(BUILD_DIR)

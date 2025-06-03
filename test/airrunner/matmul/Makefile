# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
BUILD_DIR := build

all: run

run:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/mmult_aie2.py --tile-l1-m 64 --tile-l1-n 64 --tile-l1-k 64 --tile-l2-m 128  --tile-l2-n 128 --m 512 --n 512 --k 512

clean:
	rm -rf $(BUILD_DIR) __pycache__

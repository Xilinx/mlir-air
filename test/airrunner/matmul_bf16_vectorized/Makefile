# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
programming_examples_dir := ${srcdir}/../../../programming_examples
matmul_example_dir := ${programming_examples_dir}/matrix_multiplication/bf16
BUILD_DIR := build

all: run

run:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && ${powershell} python3 ${matmul_example_dir}/run.py -p --herd-m 4 --herd-n 4 --m 512 --n 512 --k 512 > input.mlir
	cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/mmult_aie2.py 

clean:
	rm -rf $(BUILD_DIR) __pycache__

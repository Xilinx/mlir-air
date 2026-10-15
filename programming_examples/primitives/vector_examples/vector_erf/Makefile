# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

BUILD_DIR := build_peano

# Output format: xclbin (default) or elf
OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

# DTYPE: bf16 (default) or f32
DTYPE ?= bf16

# VECTOR_SIZE: 16 (default); 0 runs the scalar path
VECTOR_SIZE ?= 16

PY_FLAGS = $(OUTPUT_FORMAT_FLAG) --dtype $(DTYPE) --vector-size $(VECTOR_SIZE)

all: run

print:
	${powershell} $(PYTHON) ${srcdir}/vector_erf.py $(PY_FLAGS) -p

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} $(PYTHON) ${srcdir}/vector_erf.py $(PY_FLAGS)

clean:
	rm -rf $(BUILD_DIR) __pycache__

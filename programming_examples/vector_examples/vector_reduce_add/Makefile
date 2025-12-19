# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Vector size parameter (16 or 32)
VECTOR_SIZE ?= 16

all: run

print:
	${powershell} python3 ${srcdir}/vector_reduce_add.py -p --vector-size $(VECTOR_SIZE)

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/vector_reduce_add.py --vector-size $(VECTOR_SIZE)
clean:
	rm -rf $(BUILD_DIR) __pycache__

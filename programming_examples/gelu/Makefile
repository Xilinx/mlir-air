# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

# Defaults reproduce the original 1x2-herd smoke shape (lit test). Override
# on the command line for the registry best config, e.g.
#   make run N=3145728 TILE_N=4096 HERD_X=8 HERD_Y=2
N ?=
TILE_N ?=
HERD_X ?=
HERD_Y ?=
ITERATIONS ?= 30
ARGS = $(if $(N),--n $(N)) $(if $(TILE_N),--tile-n $(TILE_N)) \
       $(if $(HERD_X),--herd-x $(HERD_X)) $(if $(HERD_Y),--herd-y $(HERD_Y))

all: run

print:
	${powershell} python3 ${srcdir}/gelu.py $(OUTPUT_FORMAT_FLAG) $(ARGS) -p

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/gelu.py $(OUTPUT_FORMAT_FLAG) $(ARGS)

profile:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/gelu.py $(OUTPUT_FORMAT_FLAG) $(ARGS) --perf-iters $(ITERATIONS)

clean:
	rm -rf $(BUILD_DIR) __pycache__

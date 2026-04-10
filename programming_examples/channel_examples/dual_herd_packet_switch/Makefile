# (c) Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Output format: xclbin (default) or elf
OUTPUT_FORMAT ?= elf
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

# Profile parameters
TILE_SIZE ?= 4096
WARMUP ?= 5
ITERATIONS ?= 100

all: run

print:
	${powershell} python3 ${srcdir}/dual_herd_packet_switch.py $(OUTPUT_FORMAT_FLAG) -p

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/dual_herd_packet_switch.py $(OUTPUT_FORMAT_FLAG)

profile:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} python3 ${srcdir}/dual_herd_packet_switch.py $(OUTPUT_FORMAT_FLAG) \
		--tile-size $(TILE_SIZE) --warmup $(WARMUP) --iterations $(ITERATIONS)

clean:
	rm -rf $(BUILD_DIR) __pycache__

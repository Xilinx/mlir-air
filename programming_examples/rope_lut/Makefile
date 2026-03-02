# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)

all: run

print:
	${powershell} python3 ${srcdir}/rope_lut.py $(OUTPUT_FORMAT_FLAG) -p

$(BUILD_DIR)/rope.o: $(srcdir)/rope.cc
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ -O2 -std=c++20 --target=aie2p-none-unknown-elf \
		-DNDEBUG -I$(AIEOPT_DIR)/include -c $< -o $@

run: $(BUILD_DIR)/rope.o
	mkdir -p $(BUILD_DIR)/air_project
	cp $(BUILD_DIR)/rope.o $(BUILD_DIR)/air_project/rope.o
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && \
		${powershell} python3 ${srcdir}/rope_lut.py $(OUTPUT_FORMAT_FLAG)

clean:
	rm -rf $(BUILD_DIR) __pycache__

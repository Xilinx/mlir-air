# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

all: run

print:
	${powershell} python3 ${srcdir}/run.py -p

run:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) ${powershell} python3 ${srcdir}/run.py

clean:
	rm -rf $(BUILD_DIR) __pycache__

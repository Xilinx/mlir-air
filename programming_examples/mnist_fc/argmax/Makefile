# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

NE0 ?= 10
NE1 ?= 500

all: run

print:
	${powershell} $(PYTHON) ${srcdir}/run.py --ne0 $(NE0) --ne1 $(NE1) -p

run:
	mkdir -p $(BUILD_DIR)
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && ${powershell} $(PYTHON) ${srcdir}/run.py --ne0 $(NE0) --ne1 $(NE1) -v

clean:
	rm -rf $(BUILD_DIR) __pycache__

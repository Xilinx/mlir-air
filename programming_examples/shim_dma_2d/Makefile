# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
targetname := shim_dma_2d

# AIE_TARGET can be set to 'aie2' (default) or 'aie2p' to select the target architecture.
AIE_TARGET ?= aie2

# Select device based on AIE_TARGET
ifeq ($(AIE_TARGET),aie2p)
  DEVICE := npu2_1col
else
  DEVICE := npu1_1col
endif

MLIR_AIE_INSTALL_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano

# Output format: xclbin (default) or elf
OUTPUT_FORMAT ?= xclbin
OUTPUT_FORMAT_FLAG = --output-format $(OUTPUT_FORMAT)
  AIRCC_FLAGS := --no-xbridge --no-xchesscc
else
  BUILD_DIR := build_chess
  AIRCC_FLAGS := --xbridge --xchesscc
endif

all: $(BUILD_DIR)/final.xclbin

$(BUILD_DIR)/air.mlir: ${srcdir}/${targetname}.py
	mkdir -p ${@D}
	python3 $< > $@

$(BUILD_DIR)/final.xclbin: $(BUILD_DIR)/air.mlir
	mkdir -p ${@D}
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && aircc.py $(AIRCC_FLAGS) -o final.xclbin -i final.insts.bin --tmpdir tmp --device $(DEVICE) --host-target x86_64 air.mlir

$(BUILD_DIR)/final.py.xclbin: $(BUILD_DIR)/air.mlir
	mkdir -p ${@D}
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && python3 build.py $(OUTPUT_FORMAT_FLAG) air.mlir

${targetname}.exe: ${srcdir}/test.cpp
	@GPP=$$( \
		for bin in /usr/bin/g++-*; do \
			ver=$$(echo $$bin | grep -oE '[0-9]+$$'); \
			if [ "$$ver" -ge 13 ] 2>/dev/null; then \
				echo "$$ver $$bin"; \
			fi; \
		done | sort -nr | head -n1 | awk '{print $$2}' \
	); \
	if [ -z "$$GPP" ]; then \
		echo "Error: No g++ version >= 13 found in /usr/bin."; \
		exit 1; \
	fi; \
	if [ -z "$$XILINX_XRT" ]; then \
		echo "Error: XILINX_XRT environment variable not set. Please make sure to have sourced xrt/setup.sh."; \
		exit 1; \
	fi; \
	if [ -z "$(MLIR_AIE_INSTALL_DIR)" ]; then \
		echo "Error: MLIR_AIE_INSTALL_DIR could not be determined. Please make sure aie-opt is in your PATH."; \
		exit 1; \
	fi; \
	echo "Using compiler: $$GPP"; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && $$GPP ${srcdir}/test.cpp -o ${targetname}.exe -std=c++23 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-I$(MLIR_AIE_INSTALL_DIR)/runtime_lib/x86_64/test_lib/include \
		-L$(MLIR_AIE_INSTALL_DIR)/runtime_lib/x86_64/test_lib/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

run: ${targetname}.exe $(BUILD_DIR)/final.xclbin $(BUILD_DIR)/final.insts.bin 
	${powershell} $(BUILD_DIR)/${targetname}.exe -x $(BUILD_DIR)/final.xclbin -i $(BUILD_DIR)/final.insts.bin -k MLIR_AIE

run_py: $(BUILD_DIR)/final.xclbin $(BUILD_DIR)/final.insts.bin
	${powershell} python3 ${srcdir}/test.py $(BUILD_DIR)/final.xclbin $(BUILD_DIR)/final.insts.bin

pyworkflow:
	mkdir -p pybuild
	cd pybuild && ${powershell} python3 ${srcdir}/run.py

clean:
	rm -rf build_peano build_chess pybuild tmp __pycache__

# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)
targetname := shim_dma_2d

# AIE_TARGET can be set to 'aie2' (default) or 'aie2p' to select the target architecture.
AIE_TARGET ?= aie2

# Select device based on AIE_TARGET
ifeq ($(AIE_TARGET),aie2p)
  DEVICE := npu2_1col
else
  DEVICE := npu1_1col
endif

# Windows sets OS itself, but lit sanitizes the environment, so fall back to
# uname -- Git Bash, which make uses as its shell here, reports MINGW*.
ifeq ($(OS),Windows_NT)
  WINDOWS := 1
else ifneq (,$(filter MINGW% MSYS% CYGWIN%,$(shell uname -s 2>/dev/null)))
  WINDOWS := 1
endif

# Prefer the environment (utils/env_setup.sh on Linux, mlir-aie's iron_env.cmd
# on Windows) over probing PATH, and normalize separators: make's shell strips
# backslashes, and realpath under Git Bash emits an MSYS path that native
# Windows tools cannot open.
MLIR_AIE_INSTALL_DIR := $(if $(MLIR_AIE_INSTALL_DIR),$(subst \,/,$(MLIR_AIE_INSTALL_DIR)),$(shell realpath $(dir $(shell which aie-opt))/..))

# XRT's own setup.sh exports XILINX_XRT; the Windows SDK is pointed at by
# XRT_ROOT instead (iron_env.cmd deliberately clears XILINX_XRT).
XRT_DIR := $(subst \,/,$(if $(XILINX_XRT),$(XILINX_XRT),$(XRT_ROOT)))
TEST_LIB_DIR := $(MLIR_AIE_INSTALL_DIR)/runtime_lib/x86_64/test_lib

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
	$(PYTHON) $< > $@

$(BUILD_DIR)/final.xclbin: $(BUILD_DIR)/air.mlir
	mkdir -p ${@D}
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && aircc $(AIRCC_FLAGS) -o final.xclbin -i final.insts.bin --tmpdir tmp --device $(DEVICE) --host-target x86_64 air.mlir

$(BUILD_DIR)/final.py.xclbin: $(BUILD_DIR)/air.mlir
	mkdir -p ${@D}
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && $(PYTHON) build.py $(OUTPUT_FORMAT_FLAG) air.mlir

ifdef WINDOWS

# MSVC build of the same host program. Flags are spelled with a leading dash,
# which cl accepts as a synonym for a slash: make's shell here is Git Bash,
# and MSYS rewrites anything that looks like a POSIX path, turning /nologo
# into C:/.../Git/nologo. Two of them are load-bearing:
#   -Zc:__cplusplus  - without it cl reports __cplusplus as 199711L, so XRT's
#                      detail/any.h takes its pre-C++17 branch and demands
#                      boost/any.hpp, which the Windows SDK does not ship.
#   -MD              - test_utils.lib in the mlir_aie wheel is built against
#                      the dynamic CRT, while cl defaults to /MT (LNK2038).
# uuid/rt/stdc++ have no Windows equivalent and are not needed.
${targetname}.exe: ${srcdir}/test.cpp
	@if ! command -v cl >/dev/null 2>&1; then \
		echo "Error: cl.exe not found. Run from an x64 Native Tools Command Prompt for Visual Studio."; \
		exit 1; \
	fi; \
	if [ -z "$(XRT_DIR)" ]; then \
		echo "Error: neither XRT_ROOT nor XILINX_XRT is set. Point one at your XRT install."; \
		exit 1; \
	fi; \
	if [ -z "$(MLIR_AIE_INSTALL_DIR)" ]; then \
		echo "Error: MLIR_AIE_INSTALL_DIR could not be determined. Please make sure aie-opt is in your PATH."; \
		exit 1; \
	fi; \
	mkdir -p $(BUILD_DIR); \
	cd $(BUILD_DIR) && cl -nologo -EHsc -MD -std:c++20 -Zc:__cplusplus \
		${srcdir}/test.cpp -Fe:${targetname}.exe \
		-I"$(XRT_DIR)/include" -I"$(TEST_LIB_DIR)/include" \
		-link "$(XRT_DIR)/lib/xrt_coreutil.lib" "$(TEST_LIB_DIR)/lib/test_utils.lib"

else

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
	if [ -z "$(XRT_DIR)" ]; then \
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
		-I$(XRT_DIR)/include -L$(XRT_DIR)/lib \
		-I$(TEST_LIB_DIR)/include \
		-L$(TEST_LIB_DIR)/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++ -ltest_utils

endif

run: ${targetname}.exe $(BUILD_DIR)/final.xclbin $(BUILD_DIR)/final.insts.bin 
	${powershell} $(BUILD_DIR)/${targetname}.exe -x $(BUILD_DIR)/final.xclbin -i $(BUILD_DIR)/final.insts.bin -k MLIR_AIE

run_py: $(BUILD_DIR)/final.xclbin $(BUILD_DIR)/final.insts.bin
	${powershell} $(PYTHON) ${srcdir}/test.py $(BUILD_DIR)/final.xclbin $(BUILD_DIR)/final.insts.bin

pyworkflow:
	mkdir -p pybuild
	cd pybuild && ${powershell} $(PYTHON) ${srcdir}/run.py

clean:
	rm -rf build_peano build_chess pybuild tmp __pycache__

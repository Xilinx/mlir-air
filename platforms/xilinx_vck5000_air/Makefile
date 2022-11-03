# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

PLATFORM ?= xilinx_vck5000_air
CWD=$(shell pwd)
XSA_DIR ?= $(CWD)/vivado

.phony: all

$(PLATFORM): all
all: $(PLATFORM)

$(PLATFORM): xsa platform
	@echo 'Platform build complete'

xsa:
	$(MAKE) -C vivado PLATFORM=$(PLATFORM)

xsa_clean:
	$(MAKE) -C vivado clean PLATFORM=$(PLATFORM)
	$(MAKE) -C vivado ultraclean PLATFORM=$(PLATFORM)

platform: vivado/xilinx_vck5000_air.xsa
	$(MAKE) -C aie_platform all

platform_clean:
	$(MAKE) -C aie_platform clean
	$(MAKE) -C aie_platform ultraclean

clean: xsa_clean platform_clean
	${RM} -r platform_repo
	

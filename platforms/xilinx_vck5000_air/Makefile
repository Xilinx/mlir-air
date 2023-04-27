# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

TOP = $(shell git rev-parse --show-toplevel)
PLATFORM ?= xilinx_vck5000_air
CWD=$(shell pwd)
XSA_DIR ?= $(CWD)/vivado

.phony: all

blackparrot_ip:
	cp $(TOP)/platforms/ip/black-parrot/blackparrot_ip.tar.gz vivado/
	cd vivado && tar xzf blackparrot_ip.tar.gz
	cp $(TOP)/runtime_lib/controller/bp/exe/main_vivado.mem vivado/main.mem

blackparrot_ip_clean:
	rm -rf vivado/blackparrot_ip*
	rm -f vivado/main.mem

$(PLATFORM): all
all: $(PLATFORM)

$(PLATFORM): xsa platform
	@echo 'Platform build complete'

xsa: blackparrot_ip
	$(MAKE) -C vivado PLATFORM=$(PLATFORM)

xsa_clean: blackparrot_ip_clean
	$(MAKE) -C vivado clean PLATFORM=$(PLATFORM)
	$(MAKE) -C vivado ultraclean PLATFORM=$(PLATFORM)

platform: vivado/xilinx_vck5000_air.xsa
	$(MAKE) -C aie_platform all

platform_clean:
	$(MAKE) -C aie_platform clean
	$(MAKE) -C aie_platform ultraclean

clean: xsa_clean platform_clean
	${RM} -r platform_repo
	

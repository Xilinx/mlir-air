# Copyright(C) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

TOP  := $(shell git rev-parse --show-toplevel)
BASE_DIR := $(abspath $(TOP)/platforms/ip/black-parrot)

# Export variables required by the vivado tcl script
export PART       ?= xcvc1902-vsva2197-2MP-e-S
export IP_NAME     = blackparrot
export TOP_MODULE  = blackparrot
export PROJECT_NAME = $(IP_NAME)_ip_proj
export PROJECT_DIR = $(BASE_DIR)/$(PROJECT_NAME)
export PROJECT_IP_DIR = $(BASE_DIR)/$(IP_NAME)_ip

export BASE_FLIST   = $(BASE_DIR)/base.flist
export FLIST        = $(BASE_DIR)/ip.flist

export BLACKPARROT_DIR      ?= $(BASE_DIR)/black-parrot
export BLACKPARROT_SUB_DIR  ?= $(BASE_DIR)/black-parrot-subsystems
export BASEJUMP_STL_DIR      = $(BLACKPARROT_DIR)/external/basejump_stl
export BP_FE_DIR             = $(BLACKPARROT_DIR)/bp_fe
export BP_ME_DIR             = $(BLACKPARROT_DIR)/bp_me
export BP_BE_DIR             = $(BLACKPARROT_DIR)/bp_be
export BP_TOP_DIR            = $(BLACKPARROT_DIR)/bp_top
export BP_COMMON_DIR         = $(BLACKPARROT_DIR)/bp_common
export HARDFLOAT_DIR         = $(BLACKPARROT_DIR)/external/HardFloat
export IP_SRC_DIR            = $(BASE_DIR)/src
export TCL_DIR               = $(BASE_DIR)/tcl
export BP_SDK_DIR            = $(BASE_DIR)/bp-sdk

# override either of these variables to point to an installation of Verilator for linting
export BLACKPARROT_TOOLS_DIR ?= $(abspath $(TOP)/../black-parrot-tools)
export VERILATOR             ?= $(BLACKPARROT_TOOLS_DIR)/install/bin/verilator
LINT_TOP ?= blackparrot_wrapper

PATCH_DIR = $(BASE_DIR)/patch
BLACKPARROT_PATCH = $(PATCH_DIR)/black-parrot.diff
BLACKPARROT_SUBSYSTEMS_PATCH = $(PATCH_DIR)/black-parrot-subsystems.diff

ifdef SIMULATION
SIMDEF=+define+SIMULATION=1
else
SIMDEF=
endif

checkout:
	git submodule update --init --checkout -- $(BLACKPARROT_DIR)
	cd $(BLACKPARROT_DIR) && git submodule update --init
	git submodule update --init --checkout -- $(BLACKPARROT_SUB_DIR)

deinit:
	git submodule deinit -- $(BLACKPARROT_DIR)
	git submodule deinit -- $(BLACKPARROT_SUB_DIR)

patch_bp:
	cd $(BLACKPARROT_DIR) && git apply $(BLACKPARROT_PATCH)
	cd $(BLACKPARROT_SUB_DIR) && git cherry-pick 760a888 && git apply $(BLACKPARROT_SUBSYSTEMS_PATCH)

bleach_patch:
	cd $(BLACKPARROT_DIR) && git restore .
	cd $(BASE_DIR) && git submodule update -- $(BLACKPARROT_DIR)
	cd $(BLACKPARROT_SUB_DIR) && git restore .
	cd $(BASE_DIR) && git submodule update -- $(BLACKPARROT_SUB_DIR)

flist: $(FLIST)
$(FLIST): $(BLACKPARROT_DIR)/bp_top/syn/flist.vcs $(BASE_FLIST)
	echo "+incdir+$(IP_SRC_DIR)" > $@
	# substitute paths in BP flist
	cat $^ | envsubst >> $@
	# Replace hardened memory which is incorrectly inferred on some Xilinx FPGAs
	sed -i "/bsg_mem_1rw_sync_mask_write_bit.v/d" $@
	sed -i "/bsg_mem_1rw_sync_mask_write_bit_synth.v/d" $@
	echo "$(BASEJUMP_STL_DIR)/hard/ultrascale_plus/bsg_mem/bsg_mem_1rw_sync_mask_write_bit.v" >> $@
	sed -i "/bsg_mem_1rw_sync_mask_write_byte.v/d" $@
	sed -i "/bsg_mem_1rw_sync_mask_write_byte_synth.v/d" $@
	echo "$(IP_SRC_DIR)/bsg_mem_1rw_sync_mask_write_byte.v" >> $@
	echo "$(SIMDEF)" >> $@

# lint with Verilator
lint_flist: $(FLIST)
	sed -i "/blackparrot.v/d" $^

lint: lint_flist
	cat lint.cfg | envsubst > config.vlt
	$(VERILATOR) --lint-only --Wall --Wno-unoptflat -Wno-timescalemod --top-module $(LINT_TOP) \
		-f $(FLIST) config.vlt 2>&1 | tee -i lint.log

clean_lint: clean_flist
	rm -f lint.log
	rm -f config.vlt

build_ip: clean_ip clean_project $(FLIST)
	vivado -source build-ip.tcl -mode batch

pack_ip:
	tar -czvf $(IP_NAME)_ip.tar.gz $(IP_NAME)_ip

clean_ip:
	rm -rf $(PROJECT_IP_DIR)
	rm -f $(PROJECT_IP_DIR).tar.gz

clean_project:
	rm -rf $(PROJECT_DIR)

clean_flist:
	rm -f $(FLIST)

clean: clean_ip clean_project clean_lint
	rm -f *.log
	rm -f *.jou

BP_SDK_URL = https://github.com/black-parrot-sdk/black-parrot-sdk/releases/download
BP_SDK_TAG = v0.2
get_sdk:
	@[ -d "bp-sdk" ] && echo "bp-sdk exists!" \
		|| curl -L $(BP_SDK_URL)/$(BP_SDK_TAG)/tools.tar.gz | tar -xz

bleach_sdk:
	rm -rf bp-sdk

bleach_all: clean bleach_sdk bleach_patch deinit


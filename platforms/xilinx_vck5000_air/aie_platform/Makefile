# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

SYSROOT = ../petalinux/build/petalinux/images/linux/sdk/sysroots/cortexa72-cortex53-xilinx-linux
IMAGE   = ../petalinux/build/petalinux/images/linux/Image
ROOTFS  = ../petalinux/build/petalinux/images/linux/rootfs.ext4
VIVADO=$(XILINX_VIVADO)/bin/vivado


#------------------------------------------------------------------------------


ROOT_DIR=$(realpath $(dir $(lastword $(MAKEFILE_LIST))))


.PHONY: all pfm prep_sd_dir prep_sw_comp prep_sysroot ultraclean clean sd_card ${XCLBIN} ${HOST_EXE} package 

# Target: sw_comp
# Copy PetaLinux genreated software components to platform preparation directory
prep_sd_dir:
	mkdir -p sd_dir && cp ../petalinux/build/petalinux/images/linux/boot.scr ./sd_dir/

# Target: prep_sw_comp
# Copy rootfs and kernel image to sw_comp directory
prep_sw_comp:
	mkdir -p sw_comp && \
	cp ../petalinux/build/petalinux/images/linux/rootfs.ext4 ./sw_comp && \
	cp ../petalinux/build/petalinux/images/linux/Image ./sw_comp

# Target: prep_sysroot
# Install sysroot to sw_comp directory
prep_sysroot: ../petalinux/build/petalinux/images/linux/sdk.sh
	rm -rf ./sysroot && ../petalinux/build/petalinux/images/linux/sdk.sh -d ./sw_comp -y

# Target: pfm
# Generate the Vitis platform with XSCT
# Input: vck5000_custom.xsa in ../step1_vivado/build/vivado directory
pfm:
	xsct xsct_create_pfm.tcl xilinx_vck5000_air ../vivado && \
	echo "Platform created:" && \
	find . -name "*.xpfm"


#------------------------------------------------------------------------------


#TARGET   = hw_emu
TARGET   = hw
MODE	 = linux
#PLATFORM = ${PLATFORM_REPO_PATHS}/xilinx_vck190_es1_base_202110_1/xilinx_vck190_es1_base_202110_1.xpfm
#PLATFORM = /group/xrlabs2/jackl/proj/acdc/vitis_2021.2_bram_platform/platforms/vck190_bare_2021.2/vck190_bare_2021.2.xpfm
#PLATFORM = /group/xrlabs2/jackl/proj/acdc/vitis_2021.2_bram_platform/platforms/vck190_bare_2021_2/vck190_bare_2021_2.xpfm
#PLATFORM = ./platform_repo/xilinx_vck190_prod_bare/export/xilinx_vck190_prod_bare/xilinx_vck190_prod_bare.xpfm
PLATFORM = ./platform_repo/xilinx_vck5000_air/export/xilinx_vck5000_air/xilinx_vck5000_air.xpfm

XCLBIN   = vck5000_aie_base_graph_${TARGET}.xsa
XSA	 = vck5000_aie_base_graph_${TARGET}.xsa
XPFM	 = vck5000_xpfm
HOST_EXE = host.exe
SDKTARGETSYSROOT ?= ${SYSROOT}

GRAPH    = aie/graph.cpp
LIBADF  = libadf.a
AIE_CMPL_CMD = aiecompiler -platform=${PLATFORM} -include="./aie" -workdir=./Work ${GRAPH} 
AIE_SIM_CMD = aiesimulator --pkg-dir=./Work --dump-vcd foo
EMU_CMD = ./launch_hw_emu.sh

##########################################################################################################################################################
### DO NOT MODIFY BELOW THIS LINE UNLESS NECESSARY
################################################################################################################################################


VCC      = v++
VPP_SPEC =system.cfg
VPP_FLAGS=--save-temps --verbose --config ${VPP_SPEC} 
VPP_STRATEGY=--vivado.impl.strategies "Performance_ExplorePostRoutePhysOpt,Performance_NetDelay_high,Performance_WLBlockPlacementFanoutOpt"
LDCLFLAGS=

.PHONY: clean

###
# Guarding Checks. Do not modify.
###
check_defined = \
	$(strip $(foreach 1,$1, \
		$(call __check_defined,$1,$(strip $(value 2)))))

__check_defined = \
	$(if $(value $1),, \
		$(error Undefined $1$(if $2, ($2))))

guard-PLATFORM_REPO_PATHS:
	$(call check_defined, PLATFORM_REPO_PATHS, Set your where you downloaded xilinx_vck190_es1_base_202110_1)

guard-ROOTFS:
	$(call check_defined, ROOTFS, Set to: xilinx-versal-common-v2021.1/rootfs.ext4)

guard-IMAGE:
	$(call check_defined, IMAGE, Set to: xilinx-versal-common-v2021.1/Image)

guard-CXX:
	$(call check_defined, CXX, Run: xilinx-versal-common-v2021.1/environment-setup-cortexa72-cortexa53-xilinx-linux)

guard-SDKTARGETSYSROOT:
	$(call check_defined, SDKTARGETSYSROOT, Run: xilinx-versal-common-v2021.1/environment-setup-cortexa72-cortexa53-xilinx-linux)

###

all: pfm ${LIBADF} ${XCLBIN} package_${TARGET}

aie: guard-PLATFORM_REPO_PATHS ${LIBADF}
${LIBADF}: aie/*
	${AIE_CMPL_CMD}

aiesim: ${LIBADF}
	${AIE_SIM_CMD}

xclbin: guard-PLATFORM_REPO_PATHS ${XCLBIN}

${XCLBIN}: ${LIBADF} ${VPP_SPEC} 
	${VCC} -g -l --platform ${PLATFORM} ${LIBADF} -t ${TARGET} ${VPP_STRATEGY} ${VPP_FLAGS} -o $@

host: guard-CXX guard-SDKTARGETSYSROOT ${HOST_EXE}
${HOST_EXE}: ${GRAPH} ./Work/ps/c_rts/aie_control_xrt.cpp
	$(MAKE) -C sw/

package_${TARGET}: ${LIBADF} ${XCLBIN}  
	bootgen -arch versal -image vck5000_bootgen.bif -w on -o final_vck5000.pdi -log error

#	${VCC} -p -t ${TARGET} -f ${PLATFORM} \
#		--package.boot_mode=ospi \
#		--package.defer_aie_run \
#    --package.ps_elf airrt_cpp.elf,a72-0 \
#		${XCLBIN} ${LIBADF}

pdi/microblaze_reset.cdo: pdi/microblaze_reset.txt
	cdoutil -output-binary-be -output-file pdi/microblaze_reset.cdo pdi/microblaze_reset.txt

prep_package: bl31.elf plm.elf psmfw.elf system.dtb u-boot.elf pdi/microblaze_reset.cdo

program_vck5000:
	$(VIVADO) -mode batch -source program_vck5000.tcl 

clean:
	$(RM) -r vitis_* xrc.log ./sd_dir/boot.scr rootfs ./sd_card/* *.ltx
	$(RM) -rf build sd_card.img BOOT.BIN boot_image.bif aie.merged.cdo.bin a.xclbin *.pdi
	$(RM) -rf _x v++_* ${XOS} ${OS} ${LIBADF} *.o.* *.o *.xpe *.xo.* \
	       vck5000_aie_base*.xclbin* *.xsa *.log *.package_summary *.jou xnwOut Work Map_Report.csv \
	       ilpProblem* sol.db drivers .Xil *.txt *.link_summary
	$(MAKE) -C sw clean

ultraclean:
	$(RM) -r vitis_* xrc.log build platform_repo ./sd_dir/boot.scr output rootfs sw_comp


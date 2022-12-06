# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

SYSROOT = ../petalinux/sysroot/sysroots/cortexa72-cortexa53-xilinx-linux
IMAGE   = ../petalinux/images/linux/Image
ROOTFS  = ../petalinux/images/linux/rootfs.ext4


#------------------------------------------------------------------------------


ROOT_DIR=$(realpath $(dir $(lastword $(MAKEFILE_LIST))))


.PHONY: all pfm prep_sd_dir prep_sw_comp prep_sysroot ultraclean clean sd_card

# Target: sw_comp
# Copy PetaLinux genreated software components to platform preparation directory
prep_sd_dir:
	mkdir -p sd_dir && cp ../petalinux/images/linux/boot.scr ./sd_dir/

# Target: prep_sw_comp
# Copy rootfs and kernel image to sw_comp directory
prep_sw_comp:
	mkdir -p sw_comp && \
	cp ../petalinux/images/linux/rootfs.ext4 ./sw_comp && \
	cp ../petalinux/images/linux/Image ./sw_comp

# Target: prep_sysroot
# Install sysroot to sw_comp directory
prep_sysroot: ../petalinux/images/linux/sdk.sh
	rm -rf ./sysroot && ../petalinux/images/linux/sdk.sh -d ./sw_comp -y

# Target: pfm
# Generate the Vitis platform with XSCT
# Input: vck190_custom.xsa in ../step1_vivado/build/vivado directory
pfm:
	xsct xsct_create_pfm.tcl xilinx_vck190_air ../vivado && \
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
PLATFORM = ./platform_repo/xilinx_vck190_air/export/xilinx_vck190_air/xilinx_vck190_air.xpfm

XCLBIN   = vck190_aie_base_graph_${TARGET}.xclbin
XSA	 = vck190_aie_base_graph_${TARGET}.xsa
XPFM	 = vck190_xpfm
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

all: prep_sd_dir prep_sw_comp pfm prep_sysroot ${XCLBIN} ${HOST_EXE} prep_package package
sd_card: prep_sd_card sd_card.img

sd_card.img:
	${XILINX_VITIS}/scripts/vitis/util/mkfsImage.sh -s ./sd_card/ -o ./sd_card.img -e ../petalinux/images/linux/rootfs.ext4

aie: guard-PLATFORM_REPO_PATHS ${LIBADF}
${LIBADF}: aie/*
	${AIE_CMPL_CMD}

aiesim: ${LIBADF}
	${AIE_SIM_CMD}

xclbin: guard-PLATFORM_REPO_PATHS ${XCLBIN}


${XCLBIN}: ${LIBADF} ${VPP_SPEC} 
	${VCC} -g -l --platform ${PLATFORM} ${LIBADF} -t ${TARGET} ${VPP_FLAGS} -o $@

host: guard-CXX guard-SDKTARGETSYSROOT ${HOST_EXE}
${HOST_EXE}: ${GRAPH} ./Work/ps/c_rts/aie_control_xrt.cpp
	$(MAKE) -C sw/

package: guard-ROOTFS guard-IMAGE package_${TARGET}
package_${TARGET}: ${LIBADF} ${XCLBIN} ${HOST_EXE} 
	${VCC} -p -t ${TARGET} -f ${PLATFORM} \
		--package.rootfs ${ROOTFS} \
		--package.kernel_image ${IMAGE} \
		--package.boot_mode=sd \
		--package.no_image \
		--package.image_format=ext4 \
		--package.defer_aie_run \
		--package.sd_file ${HOST_EXE} \
		${XCLBIN} ${LIBADF}

prep_sd_card: ../bootgen/BOOT.BIN
	cp $< ./sd_card/

sd_card_clean:
	rm -rf sd_card.img

%.elf: ../petalinux/images/linux/%.elf
	cp $< ./platform_repo/xilinx_vck190_air/export/xilinx_vck190_air/sw/ 

%.dtb: ../petalinux/images/linux/%.dtb
	cp $< ./platform_repo/xilinx_vck190_air/export/xilinx_vck190_air/sw/ 

pdi/microblaze_reset.cdo: pdi/microblaze_reset.txt
	cdoutil -output-binary-be -output-file pdi/microblaze_reset.cdo pdi/microblaze_reset.txt

prep_package: bl31.elf plm.elf psmfw.elf system.dtb u-boot.elf pdi/microblaze_reset.cdo

run_hw_emu: launch_hw_emu.sh
launch_hw_emu.sh: package_hw_emu
	$(EMU_CMD)

clean:
	$(RM) -r vitis_* xrc.log ./sd_dir/boot.scr rootfs ./sd_card/*
	$(RM) -rf sd_card.img BOOT.BIN boot_image.bif aie.merged.cdo.bin a.xclbin 
	$(RM) -rf _x v++_* ${XOS} ${OS} ${LIBADF} *.o.* *.o *.xpe *.xo.* \
	       vck190_aie_base*.xclbin* *.xsa *.log *.package_summary *.jou xnwOut Work Map_Report.csv \
	       ilpProblem* sol.db drivers .Xil
	$(MAKE) -C sw clean

ultraclean:
	$(RM) -r vitis_* xrc.log build platform_repo ./sd_dir/boot.scr rootfs sw_comp


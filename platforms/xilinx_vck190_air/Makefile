PLATFORM ?= xilinx_vck190_air
CWD=$(shell pwd)
XSA_DIR ?= $(CWD)/vivado
OUTPUT_PATH ?= $(CWD)/platform_repo
XSCT = $(XILINX_VITIS)/bin/xsct
CPU_ARCH=a72
export COMMON_RFS_KRNL_SYSROOT = TRUE

.phony: all

$(PLATOFORM): all
all: $(PLATFORM)

$(PLATFORM): xsa petalinux_build
	@echo 'Platform build complete'

xsa:
	$(MAKE) -C vivado PLATFORM=$(PLATFORM)

xsa_clean:
	$(MAKE) -C vivado clean PLATFORM=$(PLATFORM)

petalinux_build:
	$(MAKE) -C petalinux all XSA_DIR=$(XSA_DIR) PLATFORM=$(PLATFORM)

petalinux_sysroot:
	$(MAKE) -C petalinux sysroot

petalinux_clean:
	$(MAKE) -C petalinux ultraclean PLATFORM=$(PLATFORM)

ifeq ($(COMMON_RFS_KRNL_SYSROOT), TRUE)
pfm:
	$(XSCT) -nodisp -sdx scripts/${PLATFORM}_pfm.tcl ${PLATFORM} ${XSA_DIR} ${OUTPUT_PATH}
	mkdir platform_repo/${PLATFORM}/export/${PLATFORM}/sw/${PLATFORM}/pre-built/
	cp -rf platform_repo/tmp/sw_components/src/${CPU_ARCH}/BOOT.BIN platform_repo/${PLATFORM}/export/${PLATFORM}/sw/${PLATFORM}/pre-built/
else
pfm:
	$(XSCT) -nodisp -sdx scripts/${PLATFORM}_pfm.tcl ${PLATFORM} ${XSA_DIR} ${OUTPUT_PATH}
	mkdir platform_repo/${PLATFORM}/export/${PLATFORM}/sw/${PLATFORM}/pre-built/
	cp -rf platform_repo/tmp/sw_components/src/${CPU_ARCH}/BOOT.BIN platform_repo/${PLATFORM}/export/${PLATFORM}/sw/${PLATFORM}/pre-built/
	cp -rf platform_repo/tmp/sw_components/src/${CPU_ARCH}/xrt/filesystem platform_repo/${PLATFORM}/export/${PLATFORM}/sw/${PLATFORM}/
endif

cleanall: petalinux_clean xsa_clean
	${RM} -r platform_repo
	

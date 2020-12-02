PLATFORM ?= xilinx_vck190_air
CWD=$(shell pwd)
XSA_DIR ?= $(CWD)/vivado

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

cleanall: petalinux_clean xsa_clean
	${RM} -r platform_repo
	

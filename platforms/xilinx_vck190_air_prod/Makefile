PLATFORM ?= xilinx_vck190_air
CWD=$(shell pwd)
XSA_DIR ?= $(CWD)/vivado

.phony: all

$(PLATOFORM): all
all: $(PLATFORM)

$(PLATFORM): xsa petalinux_build aie_platform bootbin sd_card
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
	$(MAKE) -C petalinux clean PLATFORM=$(PLATFORM)

petalinux_bsp:
	$(MAKE) -C petalinux bsp

aie_platform:
	$(MAKE) -C aie_platform all

aie_platform_clean:
	$(MAKE) -C aie_platform clean
	$(MAKE) -C aie_platform ultraclean

bootbin:
	$(MAKE) -C bootgen

bootbin_clean:
	$(MAKE) -C bootgen clean

sd_card:
	$(MAKE) -C aie_platform sd_card

sd_card_clean:
	$(MAKE) -C aie_platform sd_card_clean

clean: petalinux_clean xsa_clean bootbin_clean aie_platform_clean sd_card_clean
	${RM} -r platform_repo
	

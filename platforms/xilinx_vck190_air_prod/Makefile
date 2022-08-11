PLATFORM ?= xilinx_vck190_air
CWD=$(shell pwd)
XSA_DIR ?= $(CWD)/vivado

#.phony: all xsa petalinux_build aie_platform 
.PHONY: all xsa petalinux_build aie_platform pynq

$(PLATOFORM): all
#all: $(PLATFORM)
all: pynq

pynq : xsa petalinux_build petalinux_bsp aie_platform bootbin 
	-cp ./aie_platform/vck190_aie_base_graph_hw.xsa ../../pynq/vck190_air/xilinx_vck190_air.xsa
	-cp ./petalinux/xilinx_vck190_air.bsp ../../pynq/vck190_air/vck190_air.bsp
	@echo 'Pynq Platform build complete'

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
	

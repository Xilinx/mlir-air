# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

OUTPUT=../platform_repo/tmp
HW_PATH=$(OUTPUT)/vivado
VIVADO=$(XILINX_VIVADO)/bin/vivado

all: hw

hw:
		mkdir -p $(HW_PATH)
		$(VIVADO) -mode batch -notrace -source $(PLATFORM)_xsa.tcl

clean:
		$(RM) -r vivado.* $(PLATFORM) $(PLATFORM).xsa myproj *.backup.* NOC_Power.xpe .Xil

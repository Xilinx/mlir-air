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

#VITIS_BIN := /proj/xbuilds/SWIP/2020.1_0602_1208/installs/lin64/Vitis/2020.1/bin
#VIVADO_BIN := /proj/xbuilds/SWIP/2020.1_0602_1208/installs/lin64/Vivado/2020.1/bin

#PROJECT := ../../platforms/xilinx_vck5000_air/vivado/xilinx_vck5000_air.xsa
#PROJECT_SOURCES :=  ../../platforms/xilinx_vck5000_air/vivado/myproj/project_1.srcs/sources_1
PROJECT := ../../platforms/xilinx_vck190_air/vivado/xilinx_vck190_air.xsa
PROJECT_SOURCES :=  ../../platforms/xilinx_vck190_air/vivado/myproj/project_1.srcs/sources_1

# would be better if I stripped off the trailing /
SCRIPT_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
BUILD_DIR := build
APP_NAME := acdc_agent

#all: ${BUILD_DIR}/${APP_NAME}/acdc_agent.elf fixup
all: ${BUILD_DIR}/${APP_NAME}/acdc_agent.elf

#${BUILD_DIR}/${APP_NAME}/acdc_agent.elf : ../../platforms/xilinx_vck5000_air/vivado/xilinx_vck5000_air.xsa
${BUILD_DIR}/${APP_NAME}/acdc_agent.elf : ../../platforms/xilinx_vck190_air/vivado/xilinx_vck190_air.xsa
	rm -rf build && xsct build.tcl

fixup: ${BUILD_DIR}/${APP_NAME}/src/lscript.ld
	echo -e "\nModifying $< to increase the stack and heap size\n"; \
	sed -i 's/_SIZE : 0x2000/_SIZE : 0x4000/g' $<; \
	cd ${BUILD_DIR}/${APP_NAME}/Debug && make acdc_agent.elf

clean:
	rm -rf build

#VITIS_BIN := /proj/xbuilds/SWIP/2020.1_0602_1208/installs/lin64/Vitis/2020.1/bin
#VIVADO_BIN := /proj/xbuilds/SWIP/2020.1_0602_1208/installs/lin64/Vivado/2020.1/bin

PROJECT := ../../platforms/xilinx_vck190_air/vivado/xilinx_vck190_air.xsa
PROJECT_SOURCES :=  ../../platforms/xilinx_vck190_air/vivado/myproj/project_1.srcs/sources_1

# would be better if I stripped off the trailing /
SCRIPT_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
BUILD_DIR := build
APP_NAME := acdc_agent

all: ${BUILD_DIR}/${APP_NAME}/acdc_agent.elf 

${BUILD_DIR}/${APP_NAME}/acdc_agent.elf : ../../platforms/xilinx_vck190_air/vivado/xilinx_vck190_air.xsa
	rm -rf build && xsct build.tcl

clean:
	rm -rf build
# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

AIE_SRC_DIR = acdc_project
ARM_SRC_DIR = .
ARM_OBJ_DIR = .

AIE_OPT = aie-opt
AIE_XLATE = aie-translate
AIR_OPT = air-opt
ATEN_OPT = aten-opt
AIECC = aiecc.py

AIR_INSTALL_PATH = $(dir $(shell which air-opt))/..
AIE_INSTALL_PATH = $(dir $(shell which aie-opt))/..

LDFLAGS = -fuse-ld=lld \
		-rdynamic \
		-lxaiengine \
		-Wl,--whole-archive -lairhost -Wl,--no-whole-archive \
		-lstdc++ \
		-ldl

CC = clang
CFLAGS += -g -I/opt/xaiengine/include
LDFLAGS += -L/opt/xaiengine/lib

CFLAGS += -std=c++17 \
		-I$(AIR_INSTALL_PATH)/runtime_lib/airhost/include \
		-I${AIE_INSTALL_PATH}/runtime_lib \
		-DAIR_LIBXAIE_ENABLE \
		-DLIBXAIENGINEV2

LDFLAGS += -L$(AIR_INSTALL_PATH)/runtime_lib/airhost

default: all

test_library.o: ${AIE_INSTALL_PATH}/runtime_lib/test_library.cpp
	$(CC) $^ $(CFLAGS) -c -o $@

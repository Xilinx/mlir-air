# Copyright (C) 2020-2022, Xilinx Inc.
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

AIE_SRC_DIR = chess
ARM_SRC_DIR = .
ARM_OBJ_DIR = .

MLIR_CPP_FILES = $(patsubst %.mlir,%.inc,$(filter %.mlir, $(SOURCE_FILES)))
BUILD_CPP_FILES = $(filter %.cpp, $(SOURCE_FILES))

OBJ_FILES = $(filter %.o, $(SOURCE_FILES))
OBJ_FILES += $(patsubst %.cpp,%.o,$(BUILD_CPP_FILES))
OBJ_FILES += test_library.o

AIE_OPT = aie-opt
AIE_XLATE = aie-translate

ACDC_AIR = ${ACDC}/air

CFLAGS += -I../../../aie/runtime_lib -DAIR_LIBXAIE_ENABLE -DLIBXAIENGINEV2

uname_p := $(shell uname -p)
ifeq ($(uname_p),aarch64)
	CFLAGS += -I/opt/xaienginev2/include 
	LDFLAGS += -L/opt/xaienginev2/lib 
else 
  SYSROOT = /group/xrlabs/platforms/pynq-vck190-sysroot/
  CC = clang
  CFLAGS += --target=aarch64-linux-gnu --sysroot=$(SYSROOT) -g 
  CFLAGS += -I$(SYSROOT)/opt/xaienginev2/include
  LDFLAGS += --target=aarch64-linux-gnu --sysroot=$(SYSROOT) -L$(SYSROOT)/opt/xaienginev2/lib
endif

CFLAGS += -std=c++17 -I$(ACDC)/runtime_lib/airhost/include
LDFLAGS += -L${ACDC}/runtime_lib/airhost

.PHONY: all
all:

CC=clang-8

$(BUILD_CPP_FILES):  $(MLIR_CPP_FILES)

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: $(OBJ_FILES)
	$(CC) $^ \
		$(LDFLAGS) \
		-rdynamic \
		-lxaiengine \
		-lmetal \
		-lopen_amp \
		-Wl,--whole-archive -lairhost -Wl,--no-whole-archive \
		-lstdc++ \
		-ldl \
		-o $@

%.inc: %.mlir
	$(AIE_OPT) --convert-scf-to-cf --aie-create-pathfinder-flows --aie-find-flows --aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

%.elf: $(AIE_SRC_DIR)/main.cc
	cd ./$(AIE_SRC_DIR) && \
	xchessmk $*.prx && \
	cp work/Release_LLVM/$*.prx/$* ../$*.elf

%.elf: $(AIE_SRC_DIR)/%.cc
	cd ./$(AIE_SRC_DIR) && \
	xchessmk $*.prx && \
	cp work/Release_LLVM/$*.prx/$* ../$*.elf

test_library.o: ../../../aie/runtime_lib/test_library.cpp
	$(CC) $^ $(CFLAGS) -c -o $@

clean::
	rm -rf $(MLIR_CPP_FILES) $(OBJ_FILES) *.exe *.elf ./$(AIE_SRC_DIR)/work acdc_project

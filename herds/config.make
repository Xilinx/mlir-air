
AIE_SRC_DIR = acdc_project
ARM_SRC_DIR = .
ARM_OBJ_DIR = .

AIE_OPT = aie-opt
AIE_XLATE = aie-translate
ATEN_OPT = aten-opt

AIECC = aiecc.py

LDFLAGS = -fuse-ld=lld -rdynamic \
		-lxaiengine \
		-lmetal \
		-lopen_amp \
		-Wl,--whole-archive -lairhost -Wl,--no-whole-archive \
		-lstdc++ \
		-ldl

uname_p := $(shell uname -p)
ifeq ($(uname_p),aarch64)
	CC = clang -std=c++11 -g
	CFLAGS += -I/opt/xaiengine/include
	LDFLAGS += -L/opt/xaiengine/lib
else
	SYSROOT = /group/xrlabs/platforms/pynq-vck190-sysroot/
	CC = clang --target=aarch64-linux-gnu -std=c++11 --sysroot=$(SYSROOT) -g
	CFLAGS += -I$(SYSROOT)/opt/xaiengine/include
	LDFLAGS += -L$(SYSROOT)/opt/xaiengine/lib
endif

ACDC_AIR = $(dir $(shell which aie-opt))/..

CFLAGS += -std=c++11 -I$(ACDC_AIR)/runtime_lib/airhost/include -I../../../aie/runtime_lib \
		-DAIR_LIBXAIE_ENABLE
LDFLAGS += -L$(ACDC_AIR)/runtime_lib
#../../../build/air

default: all

test_library.o: ../../../aie/runtime_lib/test_library.cpp
	$(CC) $^ $(CFLAGS) -c -o $@

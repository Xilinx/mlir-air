
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
		-ldl

uname_p := $(shell uname -p)
ifeq ($(uname_p),aarch64)
	CC = clang -std=c++11
	CFLAGS += -I/opt/xaiengine/include
	LDFLAGS += -L/opt/xaiengine/lib
else
	SYSROOT = /group/xrlabs/platforms/pynq-vck190-sysroot/
	CC = clang --target=aarch64-linux-gnu -std=c++11 --sysroot=$(SYSROOT)
	CFLAGS += -I$(SYSROOT)/opt/xaiengine/include
	LDFLAGS += -L$(SYSROOT)/opt/xaiengine/lib
endif

CFLAGS += -I../../lib/include

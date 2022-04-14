# General Makefile for tests

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

CFLAGS += -I../../../aie/runtime_lib -DAIR_LIBXAIE_ENABLE

uname_p := $(shell uname -p)
ifeq ($(uname_p),aarch64)
	CFLAGS += -I/opt/xaiengine/include 
	LDFLAGS += -L/opt/xaiengine/lib 
else 
  SYSROOT = /group/xrlabs/platforms/pynq-vck190-sysroot/
  CC = clang
  CFLAGS += --target=aarch64-linux-gnu --sysroot=$(SYSROOT) -g 
  CFLAGS += -I$(SYSROOT)/opt/xaiengine/include
  LDFLAGS += --target=aarch64-linux-gnu --sysroot=$(SYSROOT) -L$(SYSROOT)/opt/xaiengine/lib
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
	$(AIE_OPT) --convert-scf-to-cf --aie-create-flows --aie-find-flows --aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

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

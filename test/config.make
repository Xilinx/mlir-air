# General Makefile for tests

AIE_SRC_DIR = chess
ARM_SRC_DIR = .
ARM_OBJ_DIR = .

MLIR_CPP_FILES = $(patsubst %.mlir,%.cpp,$(filter %.mlir, $(SOURCE_FILES)))
BUILD_CPP_FILES = $(filter %.cpp, $(SOURCE_FILES))

OBJ_FILES=$(patsubst %.cpp,%.o,$(BUILD_CPP_FILES))

AIE_OPT = aie-opt
AIE_XLATE = aie-translate

uname_p := $(shell uname -p)
ifeq ($(uname_p),aarch64)
	CFLAGS += -I/opt/xaiengine/include -I../../lib/include
	LDFLAGS += -L/opt/xaiengine/lib -L../../../build/air
endif

.PHONY: all
all:

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
		-ldl \
		-lairhost \
		-o $@

%.cpp: %.mlir
	$(AIE_OPT) --aie-create-flows --aie-find-flows --aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

%.elf: $(AIE_SRC_DIR)/main.cc
	cd ./$(AIE_SRC_DIR) && \
	xchessmk $*.prx && \
	cp work/Release_LLVM/$*.prx/$* ../$*.elf

%.elf: $(AIE_SRC_DIR)/%.cc
	cd ./$(AIE_SRC_DIR) && \
	xchessmk $*.prx && \
	cp work/Release_LLVM/$*.prx/$* ../$*.elf

clean:
	rm -rf $(MLIR_CPP_FILES) $(OBJ_FILES) *.exe *.elf ./$(AIE_SRC_DIR)/work

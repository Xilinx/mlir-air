srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

LQP ?= 64
LKP ?= 64
DK ?= 64
DV ?= 64
BUILD_DIR := build

compile-kernel:
	mkdir -p $(BUILD_DIR)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) \
		-c $(srcdir)/kernels.cc -o $(BUILD_DIR)/kernels.o \
		-Dlqp=$(LQP) -Dlkp=$(LKP) -Ddk=$(DK) -Ddv=$(DV) \
		-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16

test1: compile-kernel
	cd $(BUILD_DIR) && python3 $(srcdir)/test_matmul.py

test2: compile-kernel
	cd $(BUILD_DIR) && python3 $(srcdir)/test_matmul_exp.py

test3: compile-kernel
	cd $(BUILD_DIR) && python3 $(srcdir)/test_single_chunk_attn.py

clean:
	rm -rf $(BUILD_DIR)

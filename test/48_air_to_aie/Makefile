include ../../../mlir-aie/utils/vitisVariables.config
#include ../../../mlir-aie/tutorials/makefile-common

# The following flags are passed to both AI core and host compilation for
# aiecc.py invocations.
AIECC_FLAGS += --sysroot=${MLIR_AIE_SYSROOT} --host-target=aarch64-linux-gnu

CHESSCC_FLAGS = -p me -P ${VITIS_AIE_INCLUDE_DIR}
CHESS_FLAGS = -P ${VITIS_AIE_INCLUDE_DIR}

# The following additional flags are only applied for host code. We add the
# necessary search paths for the sysroot so clang++ can find the aarch64
# includes and libraries.
AIECC_HOST_FLAGS += -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp -I${VITIS_AIETOOLS_DIR}/include/drivers/aiengine -I${MLIR_AIE_SYSROOT}/usr/include

.PHONY: all clean

AIR_INSTALL ?= $(shell realpath $(dir $(shell which air-opt))/../runtime_lib/airhost)

AIECC_HOST_FLAGS += -I$(AIR_INSTALL)/include -lairhost_shared -lstdc++ -L$(AIR_INSTALL) 

all: test.exe

test.exe: test.cpp air.mlir
	aiecc.py -v -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

clean:
	rm -rf acdc_project sim aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts .AIE_SIM_CMD_LINE_OPTIONS

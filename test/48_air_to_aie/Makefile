.PHONY: all clean

all: test.exe

test.exe: test.cpp air.mlir
	aiecc.py  -v --host-target=aarch64-linux-gnu --sysroot=/group/xrlabs/platforms/vck190-pynq-v2.7/sysroot $(word 2,$^) \
	-I/scratch/andrab/air/mlir-air/runtime_lib/airhost/include -I/scratch/andrab/air/mlir-air/build-pcie/runtime_lib/aarch64/airhost/include -I/scratch/andrab/air/mlir-aie/runtime_lib /scratch/andrab/air/mlir-aie/runtime_lib/test_library.cpp \
	-L/scratch/andrab/air/mlir-air/build-pcie/runtime_lib/aarch64/airhost $< -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -o $@

clean:
	rm -rf acdc_project sim aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts .AIE_SIM_CMD_LINE_OPTIONS

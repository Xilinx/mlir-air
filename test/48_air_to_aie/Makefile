# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

all: test.elf

test.elf: test.cpp air.mlir
    aiecc.py -v --host-target=aarch64-linux-gnu --sysroot=/group/xrlabs/platforms/vck190-pynq-v2.7/sysroot ./air.mlir \
	-I/scratch/andrab/air/mlir-air/runtime_lib/airhost/include -I/scratch/andrab/air/mlir-air/build-pcie/runtime_lib/aarch64/airhost/include -I/scratch/andrab/air/mlir-aie/runtime_lib /scratch/andrab/air/mlir-aie/runtime_lib/test_library.cpp \
    -L/scratch/andrab/air/mlir-air/build-pcie/runtime_lib/aarch64/airhost ./test.cpp -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -o

clean:
    rm -rf acdc_project *.elf
    
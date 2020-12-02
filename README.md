# This directory contains examples of the AIR architecture
```
air
├── herds                     Example herds
├── lib                       Runtime libries for ARM and MicroBlaze
├── platforms                 Hardware platforms
│   └── xilinx_vck190_air
├── pynq                      Board repo for building Pynq images
│   └── vck190_air
├── segment-architecture      Submodule containing AIR IP blocks
└── test                      testing for AIR components
```
# Building the VCK190 AIR platform

## 1. Build the XSA

Configure the environment for Vitis (Vivado) and petalinux:

    $ source /path/to/Vitis/2020.1/settings64.sh
    $ source /path/to/petalinux/2020.1/settings.sh

Build the XSA file:

    $ cd platforms/xilinx_vck190_air
    $ make xsa

The output file is `platforms/xilinx_vck190_air/vivado/xilinx_vck190_air.xsa`

## 2. Build petalinux boot images

After building the XSA file:

    $ cd platforms/xilinx_vck190_air
    $ make petalinux_build

The boot files will be in `platforms/xilinx_vck190_air/petalinux/images/linux`
## 3. Building petalinux sysroot

After building petalinux:

    $ cd platforms/xilinx_vck190_air
    $ make petalinux_sysroot

The sysroot will be in `platforms/xilinx_vck190_air/petalinux/sysroot`

Using the sysroot for cross-compile:

    $ source platforms/xilinx_vck190_air/petalinux/sysroot/environment-setup-aarch64-xilinx-linux
    $ echo $CC
    aarch64-xilinx-linux-gcc -march=armv8-a+crc -mtune=cortex-a72.cortex-a53 --sysroot=/path/to/training-architectures/src/acdc/air/platforms/xilinx_vck190_air/petalinux/sysroot/sysroots/aarch64-xilinx-linux

Now build a test:

    $ cd test/01_simple_shim_dma
    $ make test.exe
    aarch64-xilinx-linux-gcc  -march=armv8-a+crc -mtune=cortex-a72.cortex-a53 --sysroot=/path/to/training-architectures/src/acdc/air/platforms/xilinx_vck190_air/petalinux/sysroot/sysroots/aarch64-xilinx-linux  -O2 -pipe -g -feliminate-unused-debug-types  -c -o test.o test.cpp
    aarch64-xilinx-linux-gcc  -march=armv8-a+crc -mtune=cortex-a72.cortex-a53 --sysroot=/path/to/training-architectures/src/acdc/air/platforms/xilinx_vck190_air/petalinux/sysroot/sysroots/aarch64-xilinx-linux test.o \
        -rdynamic \
        -lxaiengine \
        -lmetal \
        -lopen_amp \
        -ldl \
        -o test.exe

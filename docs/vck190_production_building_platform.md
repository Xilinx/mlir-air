
# Building the VCK190 Prodcuction AIR platform

There are three major components to the vck190 platform:
1. A Vivado project describing the hardware configuration.
2. A Petalinux configuration capturing the device-tree and device driver
required by the platform.
3. A Vitis project integrating the AIE array into the platform. 

From the Vivado project we generate a Xilinx xsa file containing configuration
information used to program the device at runtime (e.g. bitstreams). This is
passed to Petalinux to create the boot files for the system. The resulting
Petalinux build can be used to boot the board or to generate a sysroot
for cross-compilation. This build will contain some low-level libraries and
tools, but it will not contain the full set of dependencies needed to build
and run most AIR programs. 

NOTE: While not currently supported, the [Pynq based environment](vck190_building_pynq.md) 
is preferred for most use cases. Support for PYNQ on the production silicon 
board is a work in progress. 

From the Vitis project we modify the plaform to link connections to the 
AIE array and generate an xclbin file. The Vitis project is used to 
generate the `sd_card.img` file for the Versal bootable SD card. 

#
## Prerequisites

Configure the environment for Vitis (Vivado) 2021.2 and petalinux 2021.2:

    $ source /path/to/Vitis/2021.2/settings64.sh
    $ source /path/to/petalinux/2021.2/settings.sh

Additional tools licenses might be required for Versal devices.

Ensure the directory containing `parted` is on your PATH environment variable.
Typically `/sbin`.

#
## Building the SD Card Image

After building the AIE Platform:

    $ cd air/platforms/xilinx_vck190_air_prod
    $ make all

The sd_card image will be `air/platforms/xilinx_vck190_air_prod/aie_platform/sd_card.img`.

#
## Building a Petalinux sysroot (optional)

After building petalinux it is possible to create a basic sysroot for
cross-compilation:

    $ cd air/platforms/xilinx_vck190_air_prod
    $ make petalinux_sysroot

The sysroot will be in `air/platforms/xilinx_vck190_air_prod/petalinux/sysroot`

Using the sysroot for cross-compile:

    $ source platforms/xilinx_vck190_air_prod/petalinux/sysroot/environment-setup-aarch64-xilinx-linux
    $ echo $CC
    aarch64-xilinx-linux-gcc -march=armv8-a+crc -mtune=cortex-a72.cortex-a53 --sysroot=/path/to/acdc/air/platforms/xilinx_vck190_air_prod/petalinux/sysroot/sysroots/aarch64-xilinx-linux

Now build a test:

    $ cd test/01_simple_shim_dma
    $ make test.exe
    aarch64-xilinx-linux-gcc  -march=armv8-a+crc -mtune=cortex-a72.cortex-a53 --sysroot=/path/to/acdc/air/platforms/xilinx_vck190_air_prod/petalinux/sysroot/sysroots/aarch64-xilinx-linux  -O2 -pipe -g -feliminate-unused-debug-types  -c -o test.o test.cpp
    aarch64-xilinx-linux-gcc  -march=armv8-a+crc -mtune=cortex-a72.cortex-a53 --sysroot=/path/to/acdc/air/platforms/xilinx_vck190_air_prod/petalinux/sysroot/sysroots/aarch64-xilinx-linux test.o \
        -rdynamic \
        -lxaiengine \
        -lmetal \
        -lopen_amp \
        -ldl \
        -o test.exe

#
## Building step-by-step (optional):

#
### Building the XSA

Build the XSA file:

    $ cd air/platforms/xilinx_vck190_air_prod
    $ make xsa

The output file is `air/platforms/xilinx_vck190_air_prod/vivado/xilinx_vck190_air.xsa`.

#
### Building Petalinux

After building the XSA file:

    $ cd air/platforms/xilinx_vck190_air_prod
    $ make petalinux_build

The boot files will be in `air/platforms/xilinx_vck190_air_prod/petalinux/images/linux`.

To rebuild Pynq it's necessary to generate a BSP file:

    $ cd air/platforms/xilinx_vck190_air_prod
    $ make petalinux_bsp

The output file is `air/platforms/xilinx_vck190_air_prod/petalinux/xilinx_vck190_air.bsp`.

#
### Building the AIE Platform

After building the XSA and Petalinux files:

    $ cd air/platforms/xilinx_vck190_air_prod
    $ make platform

The partial sd_card image files will be in `air/platforms/xilinx_vck190_air_prod/aie_platform/sd_card`.

#
### Building the SD Card Image

After building the AIE Platform:

    $ cd air/platforms/xilinx_vck190_air_prod
    $ make bootbin
    $ make sd_card

The sd_card image will be `air/platforms/xilinx_vck190_air_prod/aie_platform/sd_card.img`.

-----

<p align="center">Copyright&copy; 2019-2022 AMD/Xilinx</p>
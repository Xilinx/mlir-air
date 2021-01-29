
# Building the VCK190 AIR platform

There are two major components to the vck190 platform:
1. A Vivado project describing the hardware configuration.
2. A Petalinux configuration capturing the device-tree and device driver
required by the platform.

From the Vivado project we generate a Xilinx xsa file containing configuration
information used to program the device at runtime (e.g. bitstreams). This is
passed to Petalinux to create the boot files for the system. The resulting
Petalinux build can be used to boot the board or to generate a sysroot
for cross-compilation. This build will contain some low-level libraries and
tools, but it will not contain the full set of dependencies needed to build
and run most AIR programs. The
[Pynq based environment](docs/vck190_building_pynq.md) if preferred for most
use cases.

#
## Prerequisites

Configure the environment for Vitis (Vivado) 2020.1 and petalinux 2020.1:

    $ source /path/to/Vitis/2020.1/settings64.sh
    $ source /path/to/petalinux/2020.1/settings.sh

Additional tools licenses might be required for Versal and ES devices.

#
## Building the XSA

Build the XSA file:

    $ cd air/platforms/xilinx_vck190_air
    $ make xsa

The output file is `air/platforms/xilinx_vck190_air/vivado/xilinx_vck190_air.xsa`.

#
## Building Petalinux

After building the XSA file:

    $ cd air/platforms/xilinx_vck190_air
    $ make petalinux_build

The boot files will be in `air/platforms/xilinx_vck190_air/petalinux/images/linux`.

To rebuild Pynq it's necessary to generate a BSP file:

    $ cd air/platforms/xilinx_vck190_air
    $ make petalinux_bsp

The output file is `air/platforms/xilinx_vck190_air/petalinux/xilinx_vck190_air.bsp`.

#
## Building a Petalinux sysroot (optional)

After building petalinux it is possible to create a basic sysroot for
cross-compilation:

    $ cd air/platforms/xilinx_vck190_air
    $ make petalinux_sysroot

The sysroot will be in `air/platforms/xilinx_vck190_air/petalinux/sysroot`

Using the sysroot for cross-compile:

    $ source platforms/xilinx_vck190_air/petalinux/sysroot/environment-setup-aarch64-xilinx-linux
    $ echo $CC
    aarch64-xilinx-linux-gcc -march=armv8-a+crc -mtune=cortex-a72.cortex-a53 --sysroot=/path/to/acdc/air/platforms/xilinx_vck190_air/petalinux/sysroot/sysroots/aarch64-xilinx-linux

Now build a test:

    $ cd test/01_simple_shim_dma
    $ make test.exe
    aarch64-xilinx-linux-gcc  -march=armv8-a+crc -mtune=cortex-a72.cortex-a53 --sysroot=/path/to/acdc/air/platforms/xilinx_vck190_air/petalinux/sysroot/sysroots/aarch64-xilinx-linux  -O2 -pipe -g -feliminate-unused-debug-types  -c -o test.o test.cpp
    aarch64-xilinx-linux-gcc  -march=armv8-a+crc -mtune=cortex-a72.cortex-a53 --sysroot=/path/to/acdc/air/platforms/xilinx_vck190_air/petalinux/sysroot/sysroots/aarch64-xilinx-linux test.o \
        -rdynamic \
        -lxaiengine \
        -lmetal \
        -lopen_amp \
        -ldl \
        -o test.exe

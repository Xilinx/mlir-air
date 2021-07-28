
# Building a PYNQ SD Image for the VCK190 AIR platform


#
## Prerequisites

1. Build the XSA file, BSP file, and Petalinux for the VCK190 AIR platform. ([Documentation](vck190_building_platform.md))

2. Setup an envrionment to build Pynq.
([Documentation](https://pynq.readthedocs.io/en/latest/pynq_sd_card.html#prepare-the-building-environment))

3. Get pre-built board-agnositc image at `/group/xrlabs2/pynq/public/v2.6.0_images/bionic.aarch64.2.6.0_2020_09_21.zip`.
(is there an external link?)

#
## Building

1. Copy the XSA and BSP to the Pynq board repository for VCK190 AIR:
```
    cp air/platforms/xilinx_vck190_air/vivado/xilinx_vck190_air.xsa air/pynq/vck190_air
    cp air/platforms/xilinx_vck190_air/petalinux/xilinx_vck190_air.bsp air/pynq/vck190_air/vck190_air.bsp
```
2. Clone [this PYNQ repo](https://gitenterprise.xilinx.com/jefff/PYNQ),
checkout the versal_acdc branch.

3. Go into the `sdbuild` folder of the PYNQ repostory and unzip the pre-built.
image into a new directory called `output`

4. Run `make PREBUILT=output/bionic.aarch64.*.img BOARDDIR=<path to acdc>/air/pynq nocheck_images`

5. Write resulting image to SD card

6. Copy the files `bl31.elf plm.elf psmfw.elf system.dtb u-boot.elf` from the PYNQ petalinux build area:
  `<path to PYNQ>/sdbuild/build/vck190_air/petalinux_project/images/linux`,
to the platform bootgen directory:
  `air/platforms/xilinx_vck190_air/bootgen`.

7. Run make in the platform bootgen directory `air/platforms/xilinx_vck190_air/bootgen`.

8. Copy the resulting BOOT.BIN file to the boot partition of the Pynq SD card writen in step 5.

Steps 6-8 configure the BOOT.BIN to bring the microblaze control processor out of reset at boot time.

#
## Updating the SD Card with a new hardware design

For changes to the Vivado design that don't require Linux driver changes, it
is often easier to update just the BOOT.BIN on the SD Card.

1. Update the Vivado design and generate a new XSA file.
2. Run make in the `air/platforms/xilinx_vck190_air/bootgen` directory
3. Copy the resulting BOOT.BIN file to the boot partition of the Pynq SD card.

For hardware changes requiring device driver or device-tree modifications, it
is best to update the Petalinux project and rebuild Pynq from scratch.

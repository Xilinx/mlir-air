
# Building and updating VCK190 AIR MicroBlaze firmware

## Prerequisites

1. Build the XSA file. ([Documentation](vck190_building_platform.md))

## Building

After building the XSA:

    $ cd air/lib/mb
    $ make

This will produce the microblaze executable `air/lib/mb/build/acdc_agent/Debug/acdc_agent.elf`

#
## Updating a live system with the new ELF
Connect to the board with XSCT

```
xsct% connect
attempting to launch hw_server

****** Xilinx hw_server v2020.1
  **** Build date : May 27 2020 at 20:33:44
    ** Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.

INFO: hw_server application started
INFO: Use Ctrl-C to exit hw_server application

INFO: To connect to this hw_server instance use url: TCP:127.0.0.1:3121

tcfchan#0
xsct% targets
  1  Versal xcvc1902
     2  RPU
        3  Cortex-R5 #0 (Halted)
        4  Cortex-R5 #1 (Lock Step Mode)
     5  APU
        6  Cortex-A72 #0 (Running)
        7  Cortex-A72 #1 (Running)
     8  PPU
       15  MicroBlaze PPU (Sleeping)
     9  PSM
       16  MicroBlaze PSM (Running)
    10  PMC
    11  PL
       12  MicroBlaze Debug Module at USER2
          13  MicroBlaze #0 (Running)
 14  DPC
xsct%
```
Select and stop the MicroBlaze
```
xsct% target 13
xsct% stop
Info: MicroBlaze #0 (target 13) Stopped at 0x40006d4c (Stop)
xsct%
```
Download the new ELF
```
xsct% dow build/acdc_agent/Debug/acdc_agent.elf                                                                                                                        
Downloading Program -- /work/acdc/air/lib/mb/build/acdc_agent/Debug/acdc_agent.elf
        section, .vectors.reset: 0x40000000 - 0x40000007
        section, .vectors.sw_exception: 0x40000008 - 0x4000000f
        section, .vectors.interrupt: 0x40000010 - 0x40000017
        section, .vectors.hw_exception: 0x40000020 - 0x40000027
        section, .text: 0x40000050 - 0x4000877f
        section, .init: 0x40008780 - 0x400087b7
        section, .fini: 0x400087b8 - 0x400087d7
        section, .rodata: 0x400087d8 - 0x40009837
        section, .sdata2: 0x40009838 - 0x40009837
        section, .data: 0x40009838 - 0x4000a48f
        section, .sdata: 0x4000a490 - 0x4000a48f
        section, .sbss: 0x4000a490 - 0x4000a48f
        section, .bss: 0x4000a490 - 0x4000bbf3
        section, .heap: 0x4000bbf4 - 0x4000c3f7
        section, .stack: 0x4000c3f8 - 0x4000c7f7
100%    0MB   0.2MB/s  00:00                                                                                                                                           
Setting PC to Program Start Address 0x40000000
Successfully downloaded /work/acdc/air/lib/mb/build/acdc_agent/Debug/acdc_agent.elf
xsct%
```
Restart the MicroBlaze
```
xsct% con
Info: MicroBlaze #0 (target 13) Running
xsct%
```
When the MicroBlaze restarts, it should print its build information to its serial console
```
MB 0 firmware 1.0.1 created on May 25 2021 at 16:14:22 GMT                      
setup_queue, 50 bytes + 64 64 byte packets                                      
Created queue @ 0x20100000040                                                   
```
The update will be lost when the board is turned off.
To make the update persist, it is necessary to update the hardware design.

## Update hardware design with new ELF

To update the hardware with the new ELF with Vivado 2020.1 it is necessary to rebuild the XSA:

1. Update the Vivado project
    ```
    $ cd acdc/air
    $ cp lib/mb/build/acdc_agent/Debug/acdc_agent.elf platforms/xilinx_vck190_air/lib/mb/acdc_agent.elf
    ```

2. Rebuild the XSA

## Updating the SD Card with the new hardware design

1. Run make in the `air/platforms/xilinx_vck190_air/bootgen` directory
2. Copy the resulting BOOT.BIN file to the boot partition of the Pynq SD card.

-----

<p align="center">Copyright&copy; 2019-2022 Advanced Micro Devices, Inc.</p>
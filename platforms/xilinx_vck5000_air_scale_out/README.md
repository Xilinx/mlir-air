# VCK5000 AIR Scale Out platform (Vitis 2022.1) 

This platform is an extension of the VCK5000 AIR Platform with scale out networking functionality. Specifically, there are two [Embedded RDMA Enabled NICs (ERNICs)](https://www.xilinx.com/products/intellectual-property/ef-di-ernic.html) which provide RoCEv2 interfaces over the two QSFP cages of the VCK5000. The ARM processor acts as an HSA AQP packet processor that, in addition to managing the AIEs as done in the VCK5000 AIR platform, manages the two ERNICs by posting RDMA operations to the ERNIC. More details of the RDMA-oriented HSA packets can be found in the ARM source code as well and the tests that utilize that functionality.

We have provided a PDI with the platform in aie_platform/final_vck5000.pdi. The provided PDI is compatible with the standard VCK5000 AIR platform ARM firmware.

## Prerequisites
Vivado 2022.1

The pdi is loaded to the card over JTAG, the USB-JTAG cable must be connected to the micro-USB input on the VCK5000 card and a programming machine (this can be the x86 host). The Xilinx Cable drivers must be [installed](https://docs.xilinx.com/r/en-US/ug973-vivado-release-notes-install-license/Installing-Cable-Drivers) on the programming machine. 

## Programming steps
Once run the top-level make completes, you should have generated aie_platform/final_vck5000.pdi containing the VCK5000 platform. The pdi can be loaded to the card by calling:
```
cd aie_platform
vivado -mode batch -source program_vck5000.tcl
```
After programming the host should undergo a **warm reboot**.
NOTE: the machine hosting the VCK5000 card will most likely crash after programming the card. This is because the PCIe link is lost during reconfiguration and the host may report an error. This is normal, and the card will be reenumerated on the PCIe bus after a **warm reboot**.

## Verification
After a warm reboot, you can verify that the card has been programmed properly with the VCK5000 AIR platfrorm by executing this command:
```
sudo lspci -vd 10ee:
```
The output should match the following (perhaps with a different bus ID):
```
21:00.0 Memory controller: Xilinx Corporation Device b034
        Subsystem: Xilinx Corporation Device 0007
        Flags: bus master, fast devsel, latency 0, IRQ 5, NUMA node 0
        Memory at c0000000 (64-bit, non-prefetchable) [size=64M]
        Memory at a0000000 (64-bit, non-prefetchable) [size=512M]
        Memory at c4000000 (64-bit, non-prefetchable) [size=2M]
        Capabilities: [40] Power Management version 3
        Capabilities: [70] Express Endpoint, MSI 00
        Capabilities: [100] Advanced Error Reporting
        Capabilities: [1c0] Secondary PCI Express
        Capabilities: [1f0] Virtual Channel
```

## Running Tests
Tests that utilize the VCK5000 Scale Out platform are located in mlir-air/test and have the phrase air_scale_out in the title. We also have a standalone test in mlir-air/runtime_lib/test/7_pcie_ernic_mrmac_standalone which bypasses the ARM and AIEs, and has the userpsace program directly post RDMA commands to the ERNICs. This can be useful to test and debug network connectivity prior to incorporating the rest of AIR. Each test will have documentation of how the cards should be connected and how the tests should be run.


-----

<p align="center">Copyright&copy; 2019-2022 Advanced Micro Devices, Inc.</p>

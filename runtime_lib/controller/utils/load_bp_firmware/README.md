NOTE: Currently, this is only used from an x86 host which will send the packet to the device controller to load the firmware of the BlackParrot cores.

This is used to dynamically load the firmware running on the BlackParrots. This program reads a local `.mem` file, loads it into device memory, and then notifies the ARM, via a packet to program all of the BlackParrots in the system. It will default to `main.mem` if a `.mem` file is not provided. To build the BlackParrot firmware, run `make -f Makefile.bp` in `mlir-air/runtime_lib/controller/`.


<p align="center">Copyright&copy; 2019-2022 AMD/Xilinx</p>

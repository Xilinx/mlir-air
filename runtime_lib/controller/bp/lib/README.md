# BlackParrot VCK firmware library

This directory contains the necessary library code for the UartLite and Mutex IP blocks
in the VCK+BP platform.

Much of the code is used directly from the [Xilinx Embedded SW](https://github.com/Xilinx/embeddedsw)
repository at the `xilinx_v2021.2` tag. A few files have been copied and modified for use by the
BlackParrot processor. These files are found in the `src` and `include` directories.

The `Makefile.frag` file defines the source files required for the library. The `XIL_LIB_SRC`
files are unmodified files from the Embedded SW repository, while `BP_FW_LIB_OVERRIDE_SRC` are
files that were copied and modified from the Embedded SW repository.

Below is a brief summary of changes made to the modified files.
- `outbyte.c` - modified to use uartlite
- `bspconfig.h` - modified (make empty)
- `xmutex.c` - replace static `XPAR_CPU_ID` parameter with dynamic CpuId argument to functions

-----

<p align="center">Copyright&copy; 2022 Advanced Micro Devices, Inc.</p>

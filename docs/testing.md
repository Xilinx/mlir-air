
# Testing

Testing is implementing using the [https://llvm.org/docs/CommandGuide/lit.html](lit framework).  The goal of this testing is to test both individual passes as part of unit testing, and end-to-end functionality of different parts of the toolchain as part of integration testing, and eventually to measure and track performance of each component.

Tests are generally run from a build directory using ninja:
```
$ cd build
$ ninja check-air
```

## Testing an Install Area

It is almost always much faster to cross-compile these tools for embedded processors (e.g. ARM/AArch64) rather than compiling locally.  To test a cross-compiled build, the tests can be configured using cmake independently from the rest of the source code.  This leverages standard cmake mechanisms to export information about an install area.

```
$ cd aie/test
$ mkdir build
$ cd build
$ cmake -GNinja .. -DCMAKE_MODULE_PATH=/home/xilinx/acdc/cmakeModules/cmakeModulesXilinx/
```
Note that CMAKE_MODULE_PATH needs to be an absolute path at the moment

## Unit Testing

Most unit tests check the behavior of individual compilation passes.  In general, we follow [https://llvm.org/docs/TestingGuide.html] best practices from LLVM, such as `FileCheck`.

```
// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T23:.*]] = AIE.tile(2, 3)
```

## On-board Integration Testing

If no board is available, then designs will still be compiled (enabling some minimal testing).  However, on a board, the tests will automatically be run as well.  This is controlled by the cmake `ENABLE_BOARD_TESTS` option, the lit configuration and the `%run_on_board` substitution:
```
$ cmake -GNinja .. -DCMAKE_MODULE_PATH=/home/xilinx/acdc/cmakeModules/cmakeModulesXilinx/ -DENABLE_BOARD_TESTS=ON
```
```
// RUN: clang ... -o %T/test.elf
// RUN: %run_on_board %T/test.elf
```

The default for `ENABLE_BOARD_TESTS` is based on the processor architecture you're compiling on.
When compiling under QEMU, you might have to explicitly disable this CMAKE option.

When a board is available, `%run_on_board` becomes `sudo`, executing the elf file.  If the execution fails (i.e., returns a negative return value), then the test will fail.  If no board is available then `%run_on_board` becomes `echo`.  Note that this mechanism means that the executable must be self-checking and cannot use the common `FileCheck`
mechanism.

Board tests must also be serialized.  Currently no in-system mechanism is provided to arbitrate access to the AIR platform.  Board tests configure lit (in `lit.cfg.py`) as below, in order to ensure serial access to the AIR platform:
```
lit_config.parallelism_groups["board"] = 1
config.parallelism_group = "board"
```

-----

<p align="center">Copyright&copy; 2019-2022 Advanced Micro Devices, Inc.</p>
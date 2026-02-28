
# Testing

Testing is implementing using the [https://llvm.org/docs/CommandGuide/lit.html](lit framework).  The goal of this testing is to test both individual passes as part of unit testing, and end-to-end functionality of different parts of the toolchain as part of integration testing, and eventually to measure and track performance of each component.

Tests are generally run from a build directory using ninja:
```
$ cd build
$ ninja check-air-mlir
$ ninja check-air-e2e-peano
$ ninja check-air-e2e-chess
$ ninja check-programming-examples-peano
$ ninja check-programming-examples-chess
```

## Unit Testing

Most unit tests check the behavior of individual compilation passes.  In general, we follow [https://llvm.org/docs/TestingGuide.html] best practices from LLVM, such as `FileCheck`.

```
// RUN: air-opt --air-dma-to-channel %s | FileCheck %s
// CHECK: %[[VAL1:.*]] = air.channel.get
```

-----

<p align="center">Copyright&copy; 2019-2024 Advanced Micro Devices, Inc.</p>
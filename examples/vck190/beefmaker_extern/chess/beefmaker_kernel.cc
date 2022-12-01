//===- beefmaker_kernel.cc -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===---------------------------------------------------------------------===//

extern "C" {

void beefmaker_kernel(uint32_t *buffer) {
  buffer[0] = 0xdeadbeef;
  buffer[1] = 0xcafecafe;
  buffer[2] = 0x000decaf;
  buffer[3] = 0x5a1ad000;
  return;
}
}
//===- beefmaker_kernel.cc -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===---------------------------------------------------------------------===//

extern "C" {

void beefmaker_kernel(uint32_t *buffer) {
  for (int i = 0; i < 1024; i += 4) {
    buffer[i + 0] = 0xdeadbeef;
    buffer[i + 1] = 0xcafecafe;
    buffer[i + 2] = 0x000decaf;
    buffer[i + 3] = 0x5a1ad000 + i;
  }
  return;
}
}
//===- airbin.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIRBIN_H
#define AIRBIN_H

#include <stdint.h>

struct airbin_size {
  uint8_t start_col = 0;
  uint8_t num_cols = 1;
  uint8_t start_row = 1;
  uint8_t num_rows = 2;
};
#endif

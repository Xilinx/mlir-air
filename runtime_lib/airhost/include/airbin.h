//===- airbin.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIRBIN_H
#define AIRBIN_H

/*
  Each entry describes a loadable section in device memory. The device uses
  this information to load the data into AIE memory. This definition is shared
  with the device firmware.
*/
struct airbin_table_entry {
  uint32_t offset; // offset into allocated device memory
  uint32_t size;   // size of the loadable section
  uint64_t addr;   // base address to load the data
};

#endif

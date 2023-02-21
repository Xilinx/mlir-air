//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022-2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "air.hpp"

int main(int argc, char *argv[]) {
  uint64_t row = 6;
  uint64_t col = 16;

  air_init();

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_init_libxaie();
  if (xaie == NULL) {
    std::cout << "Error initializing libxaie" << std::endl;
    return -1;
  }

  uint64_t core_addr = air_get_tile_addr(col, row);
  if (core_addr != 0x8180000) {
    std::cout << "FAIL: air_get_tile_addr did not return 0x8180000."
              << std::endl;
    return 1;
  }

  uint64_t tile_mem_offset = 0x1100;

  air_write32(core_addr + tile_mem_offset, 0xACDC);
  uint32_t value = air_read32(core_addr + tile_mem_offset);

  if (value != 0xACDC) {
    std::cout
        << "FAIL: air_read32 did not return the air_write32 value: 0xACDC."
        << std::endl;
    return 1;
  }

  std::cout << "Pass!" << std::endl;
  return 0;
}

// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef BP_PL_H
#define BP_PL_H

#include "bp.h"

uint32_t bp_get_global_id() {
  uint32_t* p = (uint32_t*)(BP_ID_OFFSET);
  return *p;
}

uint64_t bp_get_mimpid() {
  uint64_t mimpid;
  __asm__ volatile("csrr %0, mimpid": "=r"(mimpid): :);
  return mimpid;
}

#endif // BP_PL_H

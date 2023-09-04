//===- cdma.h   -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

void cdma_sg_init(void);
int cdma_sg_set(uint32_t idx, uint64_t dest, uint64_t src, uint32_t len);
void cdma_sg_start(uint32_t head, uint32_t tail);
uint32_t cdma_sg_start_sync(uint32_t head, uint32_t tail);
void cdma_print_status(void);

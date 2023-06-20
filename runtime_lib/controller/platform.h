//===- platform.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef __PLATFORM_H_
#define __PLATFORM_H_

#include "platform_config.h"

void init_platform();
void cleanup_platform();

/*
        Return the base address of the interface data structures
*/
uint64_t get_base_address(void);

#endif

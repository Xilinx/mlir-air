//===- utility.hpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

/*
 * -*- coding: utf-8 -*-
 *
 *
 * (c) 2016 by the author(s)
 *
 * Author(s):
 *    Andre Richter, andre.richter@tum.de
 */

#include <stdint.h>
#include <string>
#include <vector>

namespace utility {
void get_pci_dbdf(std::vector<std::string> *bdf_vect, uint32_t vendor,
                  uint32_t device, unsigned int func_num);
}

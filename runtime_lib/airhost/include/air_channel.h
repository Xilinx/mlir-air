//===- air_channel.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===--------------------------------------------------------------------===//

#ifndef AIR_CHANNEL_H
#define AIR_CHANNEL_H

#include <stdlib.h>
#include <mutex>

template <typename T> struct channel_t {
  T *data;
  std::mutex mtx;
  bool _is_full;

  channel_t() {
    data = nullptr;
    _is_full = false;
  }
};

#endif

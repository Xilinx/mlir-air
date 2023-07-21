//===- air_channel.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===--------------------------------------------------------------------===//

#ifndef AIR_CHANNEL_H
#define AIR_CHANNEL_H

#include <condition_variable>
#include <mutex>
#include <stdlib.h>

template <typename T> struct channel_t {
  T *data;
  bool _is_full;
  std::mutex mtx;
  std::condition_variable cv;

  channel_t(size_t sizes[4]) {
    data = new T[sizes[0] * sizes[1] * sizes[2] * sizes[3]];
    _is_full = false;
    cv.notify_all();
  }
};

#endif

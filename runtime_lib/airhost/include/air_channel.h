//===- air_channel.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===--------------------------------------------------------------------===//

#ifndef AIR_CHANNEL_H
#define AIR_CHANNEL_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdlib.h>

template <typename T> struct channel_t {
  T *data;
  size_t bcast_ratio[2];
  std::atomic<int> token;
  std::mutex mtx;
  std::condition_variable cv;

  channel_t(size_t sizes[4], size_t ratio[2]) {
    data = new T[sizes[0] * sizes[1] * sizes[2] * sizes[3]];
    bcast_ratio[0] = ratio[0];
    bcast_ratio[1] = ratio[1];
    token.store(0);
    cv.notify_all();
  }
};

#endif

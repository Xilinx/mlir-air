// SPDX-License-Identifier: MIT
// Copyright (C) 2023, Advanced Micro Devices, Inc.

#ifndef DEBUG_H_
#define DEBUG_H_

#include <iostream>
#include <ostream>
#include <type_traits>

template <bool enable> class DebugPrinter__ {
public:
  template <typename... Args> void operator()(Args &&...args) {
    Print(args...);
  }

private:
  template <typename T = void, typename... Args>
  static typename std::enable_if_t<enable, T> Print(Args &&...args) {
    std::ostream &debug_stream_(std::cerr);
    (debug_stream_ << ... << args);
    debug_stream_ << std::endl;
    debug_stream_.flush();
  }

  template <typename T = void, typename... Args>
  static typename std::enable_if_t<!enable, T>
  Print([[maybe_unused]] Args &&...args) {}
};

#ifdef NDEBUG
using DebugPrinter = DebugPrinter__<false>;
#else
using DebugPrinter = DebugPrinter__<true>;
#endif

inline DebugPrinter debug_print;

#endif // DEBUG_H_

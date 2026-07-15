//===- test_dequant.cpp ---------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// XRT (ELF) profiling harness for AWQ int4 -> bf16 dequantization.
//   out[i] = (q[i] - z[group(i)]) * s[group(i)]
// 2-arg kernel: (PACKED, OUT), order matches dequant_awq.py.
//
// Per-tile PACKED layout (uint8): [ Q | S(bf16) | Z(uint8) | pad ].
// The pad bytes round each tile to a 4-byte boundary (aie.dma_bd length req);
// the kernel reads only [0, Q_BYTES + S_BYTES + Z_BYTES).

#include "cxxopts.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <stdfloat>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>

using DT_BF16 = std::bfloat16_t;

static size_t round_up_4(size_t n) { return (n + 3) & ~size_t{3}; }

int main(int argc, const char *argv[]) {
  cxxopts::Options options("AWQ int4->bf16 Dequant ELF Profiling");
  options.add_options()("help,h", "produce help message")(
      "elf,e", "elf path", cxxopts::value<std::string>())(
      "kernel,k", "kernel <kernel>:<instance>", cxxopts::value<std::string>())(
      "verbosity,v", "verbosity", cxxopts::value<int>()->default_value("0"))(
      "size_n,N", "elements N", cxxopts::value<int>()->default_value("1024"))(
      "group_size,G", "AWQ group size",
      cxxopts::value<int>()->default_value("128"))(
      "herd_n,H", "compute tiles in herd",
      cxxopts::value<int>()->default_value("4"))(
      "iterations", "timed iters", cxxopts::value<int>()->default_value("50"))(
      "warmup", "warmup iters", cxxopts::value<int>()->default_value("20"));

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int N = vm["size_n"].as<int>();
  int GS = vm["group_size"].as<int>();
  int HERD_N = vm["herd_n"].as<int>();
  int n_iter_arg = vm["iterations"].as<int>();
  int n_warm_arg = vm["warmup"].as<int>();

  if (N <= 0 || GS <= 0 || HERD_N <= 0) {
    std::cerr << "Error: N, group_size, and herd_n must all be positive\n";
    return 1;
  }
  if (n_iter_arg <= 0) {
    std::cerr << "Error: iterations must be positive (divides total time)\n";
    return 1;
  }
  if (n_warm_arg < 0) {
    std::cerr << "Error: warmup must be non-negative\n";
    return 1;
  }
  if (N % HERD_N != 0) {
    std::cerr << "Error: N must be divisible by HERD_N\n";
    return 1;
  }
  int N_TILE = N / HERD_N;
  if (N_TILE % GS != 0) {
    std::cerr << "Error: N/HERD_N must be divisible by GROUP_SIZE\n";
    return 1;
  }
  int NG_TILE = N_TILE / GS;

  size_t Q_BYTES = (size_t)N_TILE / 2;
  size_t S_BYTES = (size_t)NG_TILE * sizeof(DT_BF16);
  size_t Z_BYTES = (size_t)NG_TILE;
  size_t TILE_BYTES = round_up_4(Q_BYTES + S_BYTES + Z_BYTES);
  size_t PACKED_SIZE = (size_t)HERD_N * TILE_BYTES;
  size_t OUT_SIZE = (size_t)N * sizeof(DT_BF16);

  srand(time(NULL));

  auto device = xrt::device(0);
  xrt::elf ctx_elf{vm["elf"].as<std::string>()};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, vm["kernel"].as<std::string>());

  xrt::bo bo_packed = xrt::ext::bo{device, PACKED_SIZE};
  xrt::bo bo_out = xrt::ext::bo{device, OUT_SIZE};

  {
    // Fill each per-tile slab as valid AWQ data so the profile loop runs
    // on inputs the kernel can actually process (random bytes in the S
    // region would produce bf16 NaN/Inf which destabilizes timing).
    uint8_t *base = bo_packed.map<uint8_t *>();
    memset(base, 0, PACKED_SIZE);
    for (int t = 0; t < HERD_N; t++) {
      uint8_t *p = base + (size_t)t * TILE_BYTES;
      // Q: any uint8 pattern is a valid pair of uint4 nibbles.
      for (size_t i = 0; i < Q_BYTES; i++)
        p[i] = (uint8_t)(rand() & 0xFF);
      // S: bf16 in [0.01, 0.1] (matches the Python correctness reference).
      DT_BF16 *s = reinterpret_cast<DT_BF16 *>(p + Q_BYTES);
      for (int i = 0; i < NG_TILE; i++)
        s[i] = DT_BF16(0.01f + 0.09f * ((float)rand() / RAND_MAX));
      // Z: uint4 (stored uint8) in [7, 10), per Python reference.
      uint8_t *z = p + Q_BYTES + S_BYTES;
      for (int i = 0; i < NG_TILE; i++)
        z[i] = (uint8_t)(7 + (rand() % 3));
    }
  }
  {
    DT_BF16 *p = bo_out.map<DT_BF16 *>();
    memset(p, 0, OUT_SIZE);
  }

  bo_packed.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_iter = (unsigned)n_iter_arg;
  unsigned n_warm = (unsigned)n_warm_arg;
  unsigned total = n_iter + n_warm;
  float t_total = 0, t_min = std::numeric_limits<float>::max(), t_max = 0;

  std::cout << "AWQ Dequant Benchmark (ELF, packed)\n"
            << "  N=" << N << " GS=" << GS << " HERD_N=" << HERD_N
            << " N_TILE=" << N_TILE << " TILE_BYTES=" << TILE_BYTES
            << " PACKED=" << PACKED_SIZE << " B  OUT=" << OUT_SIZE << " B\n";

  for (unsigned i = 0; i < total; i++) {
    auto run = xrt::run(kernel);
    run.set_arg(0, bo_packed);
    run.set_arg(1, bo_out);
    auto t0 = std::chrono::high_resolution_clock::now();
    run.start();
    run.wait2();
    auto t1 = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (i < n_warm)
      continue;
    float dt =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    t_total += dt;
    t_min = std::min(t_min, dt);
    t_max = std::max(t_max, dt);
  }

  // Throughput metrics: dequant has no MACs to count -- report element
  // throughput and effective output bandwidth (bf16, 2 B/elt).
  float t_avg_us = t_total / n_iter;
  float gelts_avg = (float)N / (1000.0f * t_avg_us);
  float gelts_max = (float)N / (1000.0f * t_min);
  float gbs_avg = ((float)N * sizeof(DT_BF16)) / (1000.0f * t_avg_us);
  float gbs_max = ((float)N * sizeof(DT_BF16)) / (1000.0f * t_min);

  std::cout << "\nAvg NPU time: " << t_avg_us << " us"
            << "  (" << gelts_avg << " Gelts/s, " << gbs_avg << " GB/s out)\n";
  std::cout << "Min NPU time: " << t_min << " us"
            << "  (" << gelts_max << " Gelts/s, " << gbs_max << " GB/s out)\n";
  std::cout << "Max NPU time: " << t_max << " us\n";
  return 0;
}

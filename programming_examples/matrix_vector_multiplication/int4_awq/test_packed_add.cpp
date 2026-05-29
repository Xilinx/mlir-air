//===- test_packed_add.cpp ------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// XRT (ELF) profiling harness for int4-AWQ GEMV + fused residual add.
// 4-arg kernel: (PACKED, B, R, D), order matches matvec_int4_packed_add.py.

#include "cxxopts.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

int main(int argc, const char *argv[]) {
  cxxopts::Options options("int4-AWQ GEMV+Add ELF Profiling (packed)");
  options.add_options()("help,h", "produce help message")(
      "elf,e", "elf path", cxxopts::value<std::string>())(
      "kernel,k", "kernel <kernel>:<instance>", cxxopts::value<std::string>())(
      "verbosity,v", "verbosity", cxxopts::value<int>()->default_value("0"))(
      "size_m,M", "rows M", cxxopts::value<int>()->default_value("2048"))(
      "size_k,K", "reduction K", cxxopts::value<int>()->default_value("2048"))(
      "group_size,G", "AWQ group size",
      cxxopts::value<int>()->default_value("128"))(
      "tile_m,T", "M_TILE", cxxopts::value<int>()->default_value("8"))(
      "k_chunk,C", "K_CHUNK", cxxopts::value<int>()->default_value("2048"))(
      "iterations", "timed iters", cxxopts::value<int>()->default_value("20"))(
      "warmup", "warmup iters", cxxopts::value<int>()->default_value("10"));

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int M = vm["size_m"].as<int>();
  int K = vm["size_k"].as<int>();
  int GS = vm["group_size"].as<int>();
  int M_TILE = vm["tile_m"].as<int>();
  int K_CHUNK = vm["k_chunk"].as<int>();
  int n_gpc = K_CHUNK / GS;

  size_t Q_BYTES = (size_t)M_TILE * (K_CHUNK / 2);
  size_t S_BYTES = (size_t)n_gpc * M_TILE * sizeof(DT_BF16);
  size_t Z_BYTES = (size_t)n_gpc * M_TILE;
  size_t TILE_BYTES = Q_BYTES + S_BYTES + Z_BYTES;
  size_t n_tiles_total = ((size_t)M / M_TILE) * (K / K_CHUNK);
  size_t PACKED_SIZE = n_tiles_total * TILE_BYTES;
  size_t B_SIZE = (size_t)K * sizeof(DT_BF16);
  size_t R_SIZE = (size_t)M * sizeof(DT_BF16);
  size_t D_SIZE = (size_t)M * sizeof(DT_BF16);

  srand(time(NULL));

  auto device = xrt::device(0);
  xrt::elf ctx_elf{vm["elf"].as<std::string>()};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, vm["kernel"].as<std::string>());

  xrt::bo bo_packed = xrt::ext::bo{device, PACKED_SIZE};
  xrt::bo bo_b = xrt::ext::bo{device, B_SIZE};
  xrt::bo bo_r = xrt::ext::bo{device, R_SIZE};
  xrt::bo bo_d = xrt::ext::bo{device, D_SIZE};

  {
    uint8_t *p = bo_packed.map<uint8_t *>();
    for (size_t i = 0; i < PACKED_SIZE; i++)
      p[i] = (uint8_t)(rand() & 0xFF);
  }
  {
    DT_BF16 *p = bo_b.map<DT_BF16 *>();
    for (size_t i = 0; i < (size_t)K; i++)
      p[i] = DT_BF16(4.0f * (float)rand() / RAND_MAX);
  }
  {
    DT_BF16 *p = bo_r.map<DT_BF16 *>();
    for (size_t i = 0; i < (size_t)M; i++)
      p[i] = DT_BF16(4.0f * (float)rand() / RAND_MAX);
  }
  {
    DT_BF16 *p = bo_d.map<DT_BF16 *>();
    memset(p, 0, D_SIZE);
  }

  bo_packed.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_r.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_d.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_iter = vm["iterations"].as<int>();
  unsigned n_warm = vm["warmup"].as<int>();
  unsigned total = n_iter + n_warm;
  float t_total = 0, t_min = std::numeric_limits<float>::max(), t_max = 0;
  float macs = 2.0f * (float)M * (float)K;

  std::cout << "int4-AWQ GEMV+Add Benchmark (ELF, packed cascade)\n"
            << "  M=" << M << " K=" << K << " GS=" << GS << " M_TILE=" << M_TILE
            << " K_CHUNK=" << K_CHUNK << " TILE_BYTES=" << TILE_BYTES
            << " n_tiles=" << n_tiles_total << " PACKED=" << PACKED_SIZE
            << " B\n";

  for (unsigned i = 0; i < total; i++) {
    auto run = xrt::run(kernel);
    run.set_arg(0, bo_packed);
    run.set_arg(1, bo_b);
    run.set_arg(2, bo_r);
    run.set_arg(3, bo_d);
    auto t0 = std::chrono::high_resolution_clock::now();
    run.start();
    run.wait2();
    auto t1 = std::chrono::high_resolution_clock::now();
    bo_d.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (i < n_warm)
      continue;
    float dt =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    t_total += dt;
    t_min = std::min(t_min, dt);
    t_max = std::max(t_max, dt);
  }

  std::cout << "\nAvg NPU time: " << t_total / n_iter << " us\n";
  std::cout << "Avg gflops:   " << macs / (1000.0f * t_total / n_iter) << "\n";
  std::cout << "Min NPU time: " << t_min << " us  (max gflops "
            << macs / (1000.0f * t_min) << ")\n";
  std::cout << "Max NPU time: " << t_max << " us  (min gflops "
            << macs / (1000.0f * t_max) << ")\n";
  return 0;
}

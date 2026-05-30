//===- test_o_gemv_ffn.cpp ------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// XRT (ELF) profiling harness for the 15-arg LLAMA decode ELF2 op
// (matvec_2tile_add + matvec_swiglu_rms + matvec_2tile_add fused via
// arg6-row0 subview routing). Works for both the bf16 baseline
// (o_gemv_ffn) and the full int4 variant (o_gemv_ffn_int4) — size
// the three weight BOs from the CLI.

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

int main(int argc, const char *argv[]) {
  cxxopts::Options options("ELF2 (o_gemv_ffn) Profiling");
  options.add_options()("help,h", "produce help message")(
      "elf,e", "elf path", cxxopts::value<std::string>())(
      "kernel,k", "kernel <kernel>:<instance>", cxxopts::value<std::string>())(
      "verbosity,v", "verbosity", cxxopts::value<int>()->default_value("0"))(
      "emb", "embedding dim", cxxopts::value<int>()->default_value("2048"))(
      "hidden", "hidden dim", cxxopts::value<int>()->default_value("8192"))(
      "w0_bytes", "arg0 wo bytes", cxxopts::value<size_t>())(
      "w7_bytes", "arg7 gate+up bytes", cxxopts::value<size_t>())(
      "w12_bytes", "arg12 wdown bytes", cxxopts::value<size_t>())(
      "iterations", "timed iters", cxxopts::value<int>()->default_value("50"))(
      "warmup", "warmup iters", cxxopts::value<int>()->default_value("20"));

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int emb = vm["emb"].as<int>();
  int hidden = vm["hidden"].as<int>();
  size_t w0_bytes = vm["w0_bytes"].as<size_t>();
  size_t w7_bytes = vm["w7_bytes"].as<size_t>();
  size_t w12_bytes = vm["w12_bytes"].as<size_t>();
  size_t emb_bytes = (size_t)emb * sizeof(DT_BF16);
  size_t hidden_bytes = (size_t)hidden * sizeof(DT_BF16);
  size_t rms_bytes = (size_t)2 * emb_bytes;
  size_t dead_he_bytes = (size_t)hidden * emb * sizeof(DT_BF16);

  srand(time(NULL));
  auto device = xrt::device(0);
  xrt::elf ctx_elf{vm["elf"].as<std::string>()};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, vm["kernel"].as<std::string>());

  // 15 args (see o_gemv_ffn_int4_multi.py docstring for ABI).
  xrt::bo bo0 = xrt::ext::bo{device, w0_bytes};      // wo
  xrt::bo bo1 = xrt::ext::bo{device, emb_bytes};     // attn_out
  xrt::bo bo2 = xrt::ext::bo{device, emb_bytes};     // dead
  xrt::bo bo3 = xrt::ext::bo{device, emb_bytes};     // x_residual
  xrt::bo bo4 = xrt::ext::bo{device, emb_bytes};     // dead
  xrt::bo bo5 = xrt::ext::bo{device, emb_bytes};     // dead
  xrt::bo bo6 = xrt::ext::bo{device, rms_bytes};     // rms input [2, emb]
  xrt::bo bo7 = xrt::ext::bo{device, w7_bytes};      // gate+up
  xrt::bo bo8 = xrt::ext::bo{device, hidden_bytes};  // dead
  xrt::bo bo9 = xrt::ext::bo{device, dead_he_bytes}; // dead [hidden, emb]
  xrt::bo bo10 = xrt::ext::bo{device, hidden_bytes}; // dead
  xrt::bo bo11 = xrt::ext::bo{device, hidden_bytes}; // swiglu
  xrt::bo bo12 = xrt::ext::bo{device, w12_bytes};    // wdown
  xrt::bo bo13 = xrt::ext::bo{device, emb_bytes};    // dead
  xrt::bo bo14 = xrt::ext::bo{device, emb_bytes};    // output

  // Fill weight + activation BOs with bounded random values so timing
  // isn't disturbed by NaN/Inf, but content is irrelevant for timing.
  auto fill_u8 = [](xrt::bo &bo, size_t bytes) {
    uint8_t *p = bo.map<uint8_t *>();
    for (size_t i = 0; i < bytes; i++)
      p[i] = (uint8_t)(rand() & 0xFF);
  };
  auto fill_bf16 = [](xrt::bo &bo, size_t bytes) {
    DT_BF16 *p = bo.map<DT_BF16 *>();
    size_t n = bytes / sizeof(DT_BF16);
    for (size_t i = 0; i < n; i++)
      p[i] = DT_BF16(0.01f * (float)rand() / RAND_MAX);
  };
  fill_u8(bo0, w0_bytes);
  fill_bf16(bo1, emb_bytes);
  fill_bf16(bo3, emb_bytes);
  fill_bf16(bo6, rms_bytes);
  fill_u8(bo7, w7_bytes);
  fill_u8(bo12, w12_bytes);
  memset(bo2.map<uint8_t *>(), 0, emb_bytes);
  memset(bo4.map<uint8_t *>(), 0, emb_bytes);
  memset(bo5.map<uint8_t *>(), 0, emb_bytes);
  memset(bo8.map<uint8_t *>(), 0, hidden_bytes);
  memset(bo9.map<uint8_t *>(), 0, dead_he_bytes);
  memset(bo10.map<uint8_t *>(), 0, hidden_bytes);
  memset(bo11.map<uint8_t *>(), 0, hidden_bytes);
  memset(bo13.map<uint8_t *>(), 0, emb_bytes);
  memset(bo14.map<uint8_t *>(), 0, emb_bytes);

  for (xrt::bo *b : {&bo0, &bo1, &bo2, &bo3, &bo4, &bo5, &bo6, &bo7, &bo8, &bo9,
                     &bo10, &bo11, &bo12, &bo13, &bo14})
    b->sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_iter = vm["iterations"].as<int>();
  unsigned n_warm = vm["warmup"].as<int>();
  unsigned total = n_iter + n_warm;
  float t_total = 0, t_min = std::numeric_limits<float>::max(), t_max = 0;

  std::cout << "ELF2 (o_gemv_ffn) Benchmark\n"
            << "  emb=" << emb << " hidden=" << hidden << " w0=" << w0_bytes
            << "B"
            << " w7=" << w7_bytes << "B"
            << " w12=" << w12_bytes << "B\n";

  for (unsigned i = 0; i < total; i++) {
    auto run = xrt::run(kernel);
    run.set_arg(0, bo0);
    run.set_arg(1, bo1);
    run.set_arg(2, bo2);
    run.set_arg(3, bo3);
    run.set_arg(4, bo4);
    run.set_arg(5, bo5);
    run.set_arg(6, bo6);
    run.set_arg(7, bo7);
    run.set_arg(8, bo8);
    run.set_arg(9, bo9);
    run.set_arg(10, bo10);
    run.set_arg(11, bo11);
    run.set_arg(12, bo12);
    run.set_arg(13, bo13);
    run.set_arg(14, bo14);
    auto t0 = std::chrono::high_resolution_clock::now();
    run.start();
    run.wait2();
    auto t1 = std::chrono::high_resolution_clock::now();
    bo14.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (i < n_warm)
      continue;
    float dt =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    t_total += dt;
    t_min = std::min(t_min, dt);
    t_max = std::max(t_max, dt);
  }

  // GEMV+R MACs: wo (emb*emb), gate+up (2*hidden*emb), wdown (emb*hidden).
  float macs =
      2.0f * ((float)emb * emb + 2.0f * hidden * emb + (float)emb * hidden);
  std::cout << "\nAvg NPU time: " << t_total / n_iter << " us\n";
  std::cout << "Avg gflops:   " << macs / (1000.0f * t_total / n_iter) << "\n";
  std::cout << "Min NPU time: " << t_min << " us  (max gflops "
            << macs / (1000.0f * t_min) << ")\n";
  std::cout << "Max NPU time: " << t_max << " us\n";
  return 0;
}

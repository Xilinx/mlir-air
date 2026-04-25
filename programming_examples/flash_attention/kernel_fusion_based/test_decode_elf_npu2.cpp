//===- test_decode_elf_npu2.cpp --------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Profiling harness for decode flash attention on NPU2.
// Kernel signature: decode_attention_bf16(Q, K_cache, V_cache, pos, Output)
//   Q:      [num_heads, dk]           bf16
//   K:      [num_kv_heads, lk, dk]    bf16
//   V:      [num_kv_heads, lk, dv]    bf16
//   pos:    [1]                       i32  (current_pos)
//   Output: [num_heads, dv]           bf16

#include "cxxopts.hpp"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>

using DATATYPE = std::bfloat16_t;

static inline DATATYPE random_bf16() {
  return DATATYPE(4.0f * (float)rand() / (float)RAND_MAX);
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("test_decode_elf_npu2", "Decode attention profiler");

  options.add_options()("help,h", "produce help message")(
      "elf,e", "the input elf path", cxxopts::value<std::string>())(
      "kernel,k", "kernel name (format: <kernel>:<instance>)",
      cxxopts::value<std::string>())("verbosity,v", "verbosity level",
                                     cxxopts::value<int>()->default_value("0"))(
      "lk", "KV cache sequence length",
      cxxopts::value<int>()->default_value("512"))(
      "dk", "Key head dimension", cxxopts::value<int>()->default_value("64"))(
      "dv", "Value head dimension", cxxopts::value<int>()->default_value("64"))(
      "num-heads", "Total Q heads (num_kv_heads * group_size)",
      cxxopts::value<int>()->default_value("16"))(
      "num-kv-heads", "Number of KV heads",
      cxxopts::value<int>()->default_value("4"))(
      "current-pos", "Token position for masking (default: lk-1 = full cache)",
      cxxopts::value<int>()->default_value("-1"))(
      "warmup,w", "Warmup iterations",
      cxxopts::value<int>()->default_value("10"))(
      "iterations,n", "Timed iterations",
      cxxopts::value<int>()->default_value("20"))(
      "trace-size,t", "Trace buffer size (0 to disable)",
      cxxopts::value<int>()->default_value("0"));

  auto vm = options.parse(argc, argv);
  if (vm.count("help")) {
    std::cout << options.help() << std::endl;
    return 1;
  }
  if (!vm.count("elf") || !vm.count("kernel")) {
    std::cerr << "Error: --elf and --kernel are required\n"
              << options.help() << std::endl;
    return 1;
  }

  int trace_size = vm["trace-size"].as<int>();
  int lk = vm["lk"].as<int>();
  int dk = vm["dk"].as<int>();
  int dv = vm["dv"].as<int>();
  int num_heads = vm["num-heads"].as<int>();
  int num_kv_heads = vm["num-kv-heads"].as<int>();
  int current_pos = vm["current-pos"].as<int>();
  if (current_pos < 0)
    current_pos = lk - 1;
  int verbosity = vm["verbosity"].as<int>();

  // Buffer sizes
  size_t Q_VOL = (size_t)num_heads * dk;
  size_t K_VOL = (size_t)num_kv_heads * lk * dk;
  size_t V_VOL = (size_t)num_kv_heads * lk * dv;
  size_t OUT_VOL = (size_t)num_heads * dv;
  size_t POS_SIZE = sizeof(int32_t);

  size_t Q_SIZE = Q_VOL * sizeof(DATATYPE);
  size_t K_SIZE = K_VOL * sizeof(DATATYPE);
  size_t V_SIZE = V_VOL * sizeof(DATATYPE);
  size_t OUT_SIZE = OUT_VOL * sizeof(DATATYPE);

  // XRT setup
  auto device = xrt::device(0);
  std::string elfPath = vm["elf"].as<std::string>();
  xrt::elf ctx_elf{elfPath};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  std::string kernelName = vm["kernel"].as<std::string>();
  auto kernel = xrt::ext::kernel(context, kernelName);

  // Allocate BOs: Q, K, V, pos, Output
  xrt::bo bo_q = xrt::ext::bo{device, Q_SIZE};
  xrt::bo bo_k = xrt::ext::bo{device, K_SIZE};
  xrt::bo bo_v = xrt::ext::bo{device, V_SIZE};
  xrt::bo bo_pos = xrt::ext::bo{device, POS_SIZE};
  xrt::bo bo_out =
      xrt::ext::bo{device, OUT_SIZE + static_cast<size_t>(trace_size)};

  // Fill with random data
  DATATYPE *bufQ = bo_q.map<DATATYPE *>();
  for (size_t i = 0; i < Q_VOL; i++)
    bufQ[i] = random_bf16();

  DATATYPE *bufK = bo_k.map<DATATYPE *>();
  for (size_t i = 0; i < K_VOL; i++)
    bufK[i] = random_bf16();

  DATATYPE *bufV = bo_v.map<DATATYPE *>();
  for (size_t i = 0; i < V_VOL; i++)
    bufV[i] = random_bf16();

  int32_t *bufPos = bo_pos.map<int32_t *>();
  bufPos[0] = current_pos;

  DATATYPE *bufOut = bo_out.map<DATATYPE *>();
  memset(bufOut, 0, OUT_SIZE + trace_size);

  bo_q.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_k.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_v.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_pos.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_warmup = vm["warmup"].as<int>();
  unsigned n_iter = vm["iterations"].as<int>();
  unsigned total = n_warmup + n_iter;

  float time_total = 0, time_min = std::numeric_limits<float>::max(),
        time_max = 0;

  // FLOPs: per KV head: group_size * (QK: lk*dk*2 + PV: lk*dv*2)
  int group_size = num_heads / num_kv_heads;
  float macs = (float)num_kv_heads * group_size *
               ((float)lk * dk * 2.0f + (float)lk * dv * 2.0f);

  std::cout << "Decode Attention Benchmark (ELF format)" << std::endl;
  std::cout << "  num_heads=" << num_heads << ", num_kv_heads=" << num_kv_heads
            << ", group_size=" << group_size << std::endl;
  std::cout << "  lk=" << lk << ", dk=" << dk << ", dv=" << dv
            << ", current_pos=" << current_pos << std::endl;
  std::cout << "  Q: [" << num_heads << "x" << dk << "] (" << Q_SIZE
            << " bytes)" << std::endl;
  std::cout << "  K: [" << num_kv_heads << "x" << lk << "x" << dk << "] ("
            << K_SIZE << " bytes)" << std::endl;
  std::cout << "  V: [" << num_kv_heads << "x" << lk << "x" << dv << "] ("
            << V_SIZE << " bytes)" << std::endl;
  std::cout << "  Output: [" << num_heads << "x" << dv << "] (" << OUT_SIZE
            << " bytes)" << std::endl;

  for (unsigned iter = 0; iter < total; iter++) {
    if (verbosity >= 1)
      std::cout << "Iteration " << iter << (iter < n_warmup ? " (warmup)" : "")
                << "\n";

    auto run = xrt::run(kernel);
    run.set_arg(0, bo_q);
    run.set_arg(1, bo_k);
    run.set_arg(2, bo_v);
    run.set_arg(3, bo_pos);
    run.set_arg(4, bo_out);

    auto start = std::chrono::high_resolution_clock::now();
    run.start();
    run.wait2();
    auto stop = std::chrono::high_resolution_clock::now();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup)
      continue;

    float us =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    time_total += us;
    time_min = std::min(time_min, us);
    time_max = std::max(time_max, us);
  }

  std::cout << "\nAvg NPU decode attention time: " << time_total / n_iter
            << " us" << std::endl;
  std::cout << "Avg NPU gflops: " << macs / (1000 * time_total / n_iter)
            << std::endl;
  std::cout << "\nMin NPU time: " << time_min << " us" << std::endl;
  std::cout << "Max NPU gflops: " << macs / (1000 * time_min) << std::endl;
  std::cout << "\nMax NPU time: " << time_max << " us" << std::endl;
  std::cout << "Min NPU gflops: " << macs / (1000 * time_max) << std::endl;

  if (trace_size > 0)
    test_utils::write_out_trace(((char *)bufOut) + OUT_SIZE, trace_size,
                                "trace_decode.txt");
  return 0;
}

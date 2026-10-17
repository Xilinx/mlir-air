//===- test.cpp - Fused SwiGLU profiling harness ----------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Profile harness for fused SwiGLU on NPU2 via ELF format.
// Measures e2e latency and GFLOPS over multiple iterations.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdfloat>
#include <vector>

#include "test_utils.h"

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using DATATYPE = std::bfloat16_t;

static inline std::bfloat16_t random_bfloat16_t() {
  return std::bfloat16_t(4.0f * (float)rand() / (float)(RAND_MAX));
}

int main(int argc, const char *argv[]) {

  cxxopts::Options options("Allowed options");
  options.add_options()("help,h", "produce help message")(
      "elf,e", "the input ELF path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name", cxxopts::value<std::string>())(
      "verbosity,v", "the verbosity of the output",
      cxxopts::value<int>()->default_value("0"))("size_m,M", "M dimension",
                                                 cxxopts::value<int>())(
      "size_n,N", "N dimension (output width)",
      cxxopts::value<int>())("size_k,K", "K dimension", cxxopts::value<int>())(
      "warmup,w", "Number of warmup iterations",
      cxxopts::value<int>()->default_value("10"))(
      "iterations,n", "Number of timed iterations",
      cxxopts::value<int>()->default_value("20"));

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  int verbosity = vm["verbosity"].as<int>();
  int M = vm["size_m"].as<int>();
  int K = vm["size_k"].as<int>();
  int N = vm["size_n"].as<int>();

  // x: [M, K], w_gate: [K, N], w_up: [K, N], out: [M, N]
  size_t X_SIZE = (size_t)M * K * sizeof(DATATYPE);
  size_t WGATE_SIZE = (size_t)K * N * sizeof(DATATYPE);
  size_t WUP_SIZE = (size_t)K * N * sizeof(DATATYPE);
  size_t OUT_SIZE = (size_t)M * N * sizeof(DATATYPE);

  srand(time(NULL));

  // ELF-based XRT setup
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  std::string elf_path = vm["elf"].as<std::string>();
  std::string kernel_name = vm["kernel"].as<std::string>();

  if (verbosity >= 1)
    std::cout << "Loading ELF: " << elf_path << "\n";

  auto elf = xrt::elf(elf_path);
  auto context = xrt::hw_context(device, elf);
  auto kernel = xrt::ext::kernel(context, kernel_name);

  // Use xrt::ext::bo (no group_id needed for ELF)
  xrt::bo bo_x = xrt::ext::bo(device, X_SIZE);
  xrt::bo bo_wgate = xrt::ext::bo(device, WGATE_SIZE);
  xrt::bo bo_wup = xrt::ext::bo(device, WUP_SIZE);
  xrt::bo bo_out = xrt::ext::bo(device, OUT_SIZE);

  // Fill inputs with random data
  DATATYPE *bufX = bo_x.map<DATATYPE *>();
  for (size_t i = 0; i < (size_t)M * K; i++)
    bufX[i] = random_bfloat16_t();

  DATATYPE *bufWgate = bo_wgate.map<DATATYPE *>();
  for (size_t i = 0; i < (size_t)K * N; i++)
    bufWgate[i] = random_bfloat16_t();

  DATATYPE *bufWup = bo_wup.map<DATATYPE *>();
  for (size_t i = 0; i < (size_t)K * N; i++)
    bufWup[i] = random_bfloat16_t();

  DATATYPE *bufOut = bo_out.map<DATATYPE *>();
  std::memset(bufOut, 0, OUT_SIZE);

  bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_wgate.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_wup.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_iterations = vm["iterations"].as<int>();
  unsigned n_warmup_iterations = vm["warmup"].as<int>();
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = std::numeric_limits<float>::max();
  float npu_time_max = 0;

  // FLOPs: matmul = 2*M*K*N (gate) + 2*M*K*N (up), SiLU ~ 8*M*N, mul = M*N
  // Total ~ 4*M*K*N + 9*M*N
  float macs =
      4.0f * float(M) * float(K) * float(N) + 9.0f * float(M) * float(N);

  std::cout << "Fused SwiGLU Benchmark" << std::endl;
  std::cout << "  M=" << M << ", K=" << K << ", N=" << N << std::endl;
  std::cout << "  x: [" << M << "x" << K << "] (" << X_SIZE << " bytes)"
            << std::endl;
  std::cout << "  w_gate: [" << K << "x" << N << "] (" << WGATE_SIZE
            << " bytes)" << std::endl;
  std::cout << "  w_up: [" << K << "x" << N << "] (" << WUP_SIZE << " bytes)"
            << std::endl;
  std::cout << "  output: [" << M << "x" << N << "] (" << OUT_SIZE << " bytes)"
            << std::endl;
  std::cout << "  warmup=" << n_warmup_iterations
            << ", iterations=" << n_iterations << std::endl;

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel (iteration " << iter << ").\n";

    auto start = std::chrono::high_resolution_clock::now();
    // ELF path: use xrt::run with set_arg (4 args: x, w_gate, w_up, out)
    auto run = xrt::run(kernel);
    run.set_arg(0, bo_x);
    run.set_arg(1, bo_wgate);
    run.set_arg(2, bo_wup);
    run.set_arg(3, bo_out);
    run.start();
    run.wait2();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations)
      continue;

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  std::cout << std::endl
            << "Avg NPU fused SwiGLU time: " << npu_time_total / n_iterations
            << "us." << std::endl;
  std::cout << "Avg NPU gflops: "
            << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU fused SwiGLU time: " << npu_time_min << "us."
            << std::endl;
  std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;

  std::cout << std::endl
            << "Max NPU fused SwiGLU time: " << npu_time_max << "us."
            << std::endl;
  std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

  return 0;
}

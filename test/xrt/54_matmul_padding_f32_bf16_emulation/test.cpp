//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Profiling harness for f32 matmul with bf16 emulation (ELF format).
// A is K×M (transposed), B is K×N, C is M×N (all f32).
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Experimental headers for elf format support
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>

using A_DATATYPE = float;
using B_DATATYPE = float;
using C_DATATYPE = float;

void add_default_options(cxxopts::Options &options) {
  options.add_options()("help,h", "produce help message")(
      "elf,e", "the input elf path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name (format: <kernel>:<instance>)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "size_m,M", "Actual matrix M (for GFLOPS)", cxxopts::value<int>())(
      "size_n,N", "Actual matrix N (for GFLOPS)", cxxopts::value<int>())(
      "size_k,K", "Matrix K dimension",
      cxxopts::value<int>())("alloc_m", "M buffer alloc size (default: M)",
                             cxxopts::value<int>()->default_value("0"))(
      "alloc_n", "N buffer alloc size (default: N)",
      cxxopts::value<int>()->default_value("0"));
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Allowed options");
  add_default_options(options);
  auto vm = options.parse(argc, argv);

  if (vm.count("help")) {
    std::cout << options.help() << std::endl;
    return 1;
  }

  if (!vm.count("elf") || !vm.count("kernel") || !vm.count("size_m") ||
      !vm.count("size_n") || !vm.count("size_k")) {
    std::cerr << "Error: Required options missing (-e, -k, -M, -N, -K)\n\n";
    std::cerr << options.help() << std::endl;
    return 1;
  }

  int verbosity = vm["verbosity"].as<int>();
  int M = vm["size_m"].as<int>();
  int K = vm["size_k"].as<int>();
  int N = vm["size_n"].as<int>();
  int M_buf = vm["alloc_m"].as<int>();
  int N_buf = vm["alloc_n"].as<int>();
  if (M_buf <= 0)
    M_buf = M;
  if (N_buf <= 0)
    N_buf = N;

  // A is K×M_buf (transposed layout), B is K×N_buf, C is M_buf×N_buf
  size_t A_VOLUME = (size_t)K * M_buf;
  size_t B_VOLUME = (size_t)K * N_buf;
  size_t C_VOLUME = (size_t)M_buf * N_buf;

  size_t A_SIZE = A_VOLUME * sizeof(A_DATATYPE);
  size_t B_SIZE = B_VOLUME * sizeof(B_DATATYPE);
  size_t C_SIZE = C_VOLUME * sizeof(C_DATATYPE);

  srand(time(NULL));

  // Set up XRT with ELF
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  std::string elfPath = vm["elf"].as<std::string>();
  if (verbosity >= 1)
    std::cout << "Loading elf: " << elfPath << "\n";

  xrt::elf ctx_elf{elfPath};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);

  std::string kernelName = vm["kernel"].as<std::string>();
  if (verbosity >= 1)
    std::cout << "Kernel name: " << kernelName << "\n";

  auto kernel = xrt::ext::kernel(context, kernelName);

  // Create buffer objects using xrt::ext::bo
  xrt::bo bo_a = xrt::ext::bo{device, A_SIZE};
  xrt::bo bo_b = xrt::ext::bo{device, B_SIZE};
  xrt::bo bo_c = xrt::ext::bo{device, C_SIZE};

  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  for (size_t i = 0; i < A_VOLUME; i++)
    bufA[i] = 4.0f * (float)rand() / (float)RAND_MAX;

  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  for (size_t i = 0; i < B_VOLUME; i++)
    bufB[i] = 4.0f * (float)rand() / (float)RAND_MAX;

  C_DATATYPE *bufC = bo_c.map<C_DATATYPE *>();
  memset(bufC, 0, C_SIZE);

  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_iterations = 20;
  unsigned n_warmup_iterations = 10;
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = std::numeric_limits<float>::max();
  float npu_time_max = 0;

  float macs = 2.0f * float(M) * float(K) * float(N);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel.\n";

    auto run = xrt::run(kernel);
    run.set_arg(0, bo_a);
    run.set_arg(1, bo_b);
    run.set_arg(2, bo_c);

    auto start = std::chrono::high_resolution_clock::now();
    run.start();
    run.wait2();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

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
            << "Avg NPU matmul time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  std::cout << "Avg NPU gflops: "
            << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU matmul time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;

  std::cout << std::endl
            << "Max NPU matmul time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

  return 0;
}

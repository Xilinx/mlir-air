//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Profiling harness for f32 matmul with bf16 emulation.
// A is K×M (transposed), B is K×N, C is M×N (all f32).
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using A_DATATYPE = float;
using B_DATATYPE = float;
using C_DATATYPE = float;

void add_default_options(cxxopts::Options &options) {
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i", "path of file containing instructions",
      cxxopts::value<std::string>())("size_m,M", "Actual matrix M (for GFLOPS)",
                                     cxxopts::value<int>())(
      "size_n,N", "Actual matrix N (for GFLOPS)", cxxopts::value<int>())(
      "size_k,K", "Matrix K dimension", cxxopts::value<int>())(
      "alloc_m", "M buffer alloc size (default: M)",
      cxxopts::value<int>()->default_value("0"))(
      "alloc_n", "N buffer alloc size (default: N)",
      cxxopts::value<int>()->default_value("0"));
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Allowed options");
  cxxopts::ParseResult vm;
  add_default_options(options);
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  int M = vm["size_m"].as<int>();
  int K = vm["size_k"].as<int>();
  int N = vm["size_n"].as<int>();
  // Buffer alloc sizes (padded to tile-aligned for hardware).
  // M/N are the actual dimensions used for GFLOPS calculation.
  int M_buf = vm["alloc_m"].as<int>();
  int N_buf = vm["alloc_n"].as<int>();
  if (M_buf <= 0) M_buf = M;
  if (N_buf <= 0) N_buf = N;

  // A is K×M_buf (transposed layout), B is K×N_buf, C is M_buf×N_buf
  int A_VOLUME = K * M_buf;
  int B_VOLUME = K * N_buf;
  int C_VOLUME = M_buf * N_buf;

  int A_SIZE = A_VOLUME * sizeof(A_DATATYPE);
  int B_SIZE = B_VOLUME * sizeof(B_DATATYPE);
  int C_SIZE = C_VOLUME * sizeof(C_DATATYPE);

  srand(time(NULL));

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  std::string Node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1)
                                   std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c =
      xrt::bo(device, C_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  for (int i = 0; i < A_VOLUME; i++)
    bufA[i] = 4.0f * (float)rand() / (float)RAND_MAX;

  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  for (int i = 0; i < B_VOLUME; i++)
    bufB[i] = 4.0f * (float)rand() / (float)RAND_MAX;

  C_DATATYPE *bufC = bo_c.map<C_DATATYPE *>();
  memset(bufC, 0, C_SIZE);

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_iterations = 20;
  unsigned n_warmup_iterations = 10;
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  float macs = 2.0f * float(M) * float(K) * float(N);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel.\n";

    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
    run.wait();
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
            << "Avg NPU matmul time: " << npu_time_total / n_iterations
            << "us." << std::endl;
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

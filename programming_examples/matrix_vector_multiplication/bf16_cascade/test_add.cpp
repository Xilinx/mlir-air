//===- test_add.cpp ---------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// XRT profiling harness for fused GEMV + residual add:
//   D[M] = A[M,K] @ B[K] + R[M]
//

#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using A_DATATYPE = std::bfloat16_t;
using B_DATATYPE = std::bfloat16_t;
using R_DATATYPE = std::bfloat16_t;
using D_DATATYPE = std::bfloat16_t;

void add_default_options(cxxopts::Options &options) {
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions to be sent to the LX6",
      cxxopts::value<std::string>())(
      "size_m,M", "Matrix rows M (output dimension)",
      cxxopts::value<int>()->default_value("2048"))(
      "size_k,K", "Vector size K (reduction dimension)",
      cxxopts::value<int>()->default_value("8192"));
}

static inline std::bfloat16_t random_bfloat16_t() {
  return std::bfloat16_t(4.0 * (float)rand() / (float)(RAND_MAX));
}

int main(int argc, const char *argv[]) {

  cxxopts::Options options("GEMV+Add BF16 Profiling");
  cxxopts::ParseResult vm;
  add_default_options(options);
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  int M = vm["size_m"].as<int>();
  int K = vm["size_k"].as<int>();

  int A_VOLUME = M * K;
  int B_VOLUME = K;
  int R_VOLUME = M;
  int D_VOLUME = M;

  int A_SIZE = (A_VOLUME * sizeof(A_DATATYPE));
  int B_SIZE = (B_VOLUME * sizeof(B_DATATYPE));
  int R_SIZE = (R_VOLUME * sizeof(R_DATATYPE));
  int D_SIZE = (D_VOLUME * sizeof(D_DATATYPE));

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

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel_it = std::find_if(xkernels.begin(), xkernels.end(),
                                 [Node, verbosity](xrt::xclbin::kernel &k) {
                                   auto name = k.get_name();
                                   if (verbosity >= 1) {
                                     std::cout << "Name: " << name << std::endl;
                                   }
                                   return name.rfind(Node, 0) == 0;
                                 });
  if (xkernel_it == xkernels.end()) {
    std::cerr << "Error: kernel '" << Node << "' not found in xclbin '"
              << vm["xclbin"].as<std::string>() << "'. Available kernels:";
    for (auto &k : xkernels)
      std::cerr << "\n  - " << k.get_name();
    std::cerr << std::endl;
    return EXIT_FAILURE;
  }
  auto xkernel = *xkernel_it;
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  // Kernel signature: (opcode, instr, instr_count, A, B, R, D)
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_r =
      xrt::bo(device, R_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_d =
      xrt::bo(device, D_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec(A_VOLUME);
  for (int i = 0; i < A_VOLUME; i++) {
    AVec[i] = random_bfloat16_t();
  }
  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(A_DATATYPE)));

  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec(B_VOLUME);
  for (int i = 0; i < B_VOLUME; i++) {
    BVec[i] = random_bfloat16_t();
  }
  memcpy(bufB, BVec.data(), (BVec.size() * sizeof(B_DATATYPE)));

  R_DATATYPE *bufR = bo_r.map<R_DATATYPE *>();
  std::vector<R_DATATYPE> RVec(R_VOLUME);
  for (int i = 0; i < R_VOLUME; i++) {
    RVec[i] = random_bfloat16_t();
  }
  memcpy(bufR, RVec.data(), (RVec.size() * sizeof(R_DATATYPE)));

  D_DATATYPE *bufD = bo_d.map<D_DATATYPE *>();
  std::vector<D_DATATYPE> DVec(D_VOLUME, 0);
  memcpy(bufD, DVec.data(), (DVec.size() * sizeof(D_DATATYPE)));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_r.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_d.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_iterations = 20;
  unsigned n_warmup_iterations = 10;
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  // GEMV+add: D = A·B + R, MACs = 2·M·K (the +R is M scalar adds, ignored).
  float macs = 2.0 * float(M) * float(K);

  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel.\n";
    }
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_r, bo_d);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_d.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      continue;
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  std::cout << std::endl
            << "GEMV+Add size: M=" << M << ", K=" << K << std::endl;

  std::cout << std::endl
            << "Avg NPU GEMV+Add time: " << npu_time_total / n_iterations
            << "us." << std::endl;
  std::cout << "Avg NPU gflops: "
            << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU GEMV+Add time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;

  std::cout << std::endl
            << "Max NPU GEMV+Add time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

  return 0;
}

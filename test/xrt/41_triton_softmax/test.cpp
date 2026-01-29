//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

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

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using IN_DATATYPE = float;
using OUT_DATATYPE = float;

static inline float random_float() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

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
      "size_m,M", "Matrix size M (parallel dimension)",
      cxxopts::value<int>()->default_value("256"))(
      "size_n,N", "Matrix size N (reduction dimension)",
      cxxopts::value<int>()->default_value("256"));
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  cxxopts::Options options("Triton Softmax Profiling");
  cxxopts::ParseResult vm;
  add_default_options(options);
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  int M = vm["size_m"].as<int>();
  int N = vm["size_n"].as<int>();

  int IN_VOLUME = M * N;
  int OUT_VOLUME = M * N;

  int IN_SIZE = (IN_VOLUME * sizeof(IN_DATATYPE));
  int OUT_SIZE = (OUT_VOLUME * sizeof(OUT_DATATYPE));

  srand(time(NULL));

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in =
      xrt::bo(device, IN_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out =
      xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Initialize input matrix with random float values
  IN_DATATYPE *bufIn = bo_in.map<IN_DATATYPE *>();
  std::vector<IN_DATATYPE> InVec(IN_VOLUME);
  for (int i = 0; i < IN_VOLUME; i++) {
    InVec[i] = random_float();
  }
  memcpy(bufIn, InVec.data(), (InVec.size() * sizeof(IN_DATATYPE)));

  OUT_DATATYPE *bufOut = bo_out.map<OUT_DATATYPE *>();
  std::vector<OUT_DATATYPE> OutVec(OUT_VOLUME, 0);
  memcpy(bufOut, OutVec.data(), (OutVec.size() * sizeof(OUT_DATATYPE)));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_iterations = 20;
  unsigned n_warmup_iterations = 10;
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  // Softmax operations count estimation:
  // For each row of size N:
  // - N reads + 1 max comparison per element = N ops (max reduction)
  // - N subtractions = N ops
  // - N exp operations ~ 10N ops (rough estimate)
  // - N additions = N ops (sum reduction)
  // - N divisions = N ops
  // Total per row: ~14N ops, for M rows: 14*M*N ops
  float ops = 14.0f * float(M) * float(N);

  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel.\n";
    }
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    memcpy(OutVec.data(), bufOut, (OutVec.size() * sizeof(OUT_DATATYPE)));

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  std::cout << std::endl << "Softmax size: " << M << " x " << N << std::endl;
  std::cout << std::endl
            << "Avg NPU softmax time: " << npu_time_total / n_iterations
            << "us." << std::endl;
  std::cout << "Avg NPU Mops: " << ops / (1000 * npu_time_total / n_iterations)
            << std::endl;

  std::cout << std::endl
            << "Min NPU softmax time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU Mops: " << ops / (1000 * npu_time_min) << std::endl;

  std::cout << std::endl
            << "Max NPU softmax time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU Mops: " << ops / (1000 * npu_time_max) << std::endl;

  return 0;
}

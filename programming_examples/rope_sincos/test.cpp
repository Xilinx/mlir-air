//===- test.cpp - RoPE sin/cos profiling harness ----------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// C++ XRT harness for profiling the RoPE sin/cos kernel.
// Times only kernel dispatch + wait (no BO sync in timer).
//
// Usage:
//   ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin
//              -H <num_heads> -D <head_size>
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdfloat>
#include <vector>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using DATATYPE = std::bfloat16_t;

int main(int argc, const char *argv[]) {

  cxxopts::Options options("RoPE sin/cos profiling");
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i", "path of file containing instructions",
      cxxopts::value<std::string>())("heads,H", "Number of heads",
                                     cxxopts::value<int>()->default_value("8"))(
      "dim,D", "Head dimension (head_size)",
      cxxopts::value<int>()->default_value("48"));

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  int num_heads = vm["heads"].as<int>();
  int head_size = vm["dim"].as<int>();
  // Input/output layout: [3 * num_heads * head_size] (Q, K, V concatenated)
  int total = 3 * num_heads * head_size;

  int DATA_SIZE = total * sizeof(DATATYPE);

  srand(time(NULL));

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // XRT setup
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string Node = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  // Allocate BOs: instructions, input, output
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input =
      xrt::bo(device, DATA_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_output =
      xrt::bo(device, DATA_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  // Fill input with random data
  DATATYPE *buf_input = bo_input.map<DATATYPE *>();
  DATATYPE *buf_output = bo_output.map<DATATYPE *>();

  for (int i = 0; i < total; i++) {
    buf_input[i] = DATATYPE(8.0f * (float)rand() / (float)RAND_MAX - 4.0f);
    buf_output[i] = DATATYPE(0.0f);
  }

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Profiling
  unsigned n_warmup = 10;
  unsigned n_iterations = 20;
  unsigned num_iter = n_warmup + n_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  // Total data moved: input + output (same size)
  float data_bytes = 2.0f * total * sizeof(DATATYPE);

  std::cout << "RoPE sin/cos profiling: num_heads=" << num_heads
            << ", head_size=" << head_size << ", total=" << total << std::endl;

  for (unsigned iter = 0; iter < num_iter; iter++) {
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_input, bo_output);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();

    bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup)
      continue;

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  float avg_us = npu_time_total / n_iterations;

  std::cout << std::endl;
  std::cout << "Avg NPU time: " << avg_us << " us" << std::endl;
  std::cout << "Min NPU time: " << npu_time_min << " us" << std::endl;
  std::cout << "Max NPU time: " << npu_time_max << " us" << std::endl;
  std::cout << std::endl;
  std::cout << "Avg bandwidth: " << std::fixed << std::setprecision(2)
            << data_bytes / (avg_us * 1000.0f) << " GB/s" << std::endl;
  std::cout << "Max bandwidth: " << data_bytes / (npu_time_min * 1000.0f)
            << " GB/s" << std::endl;

  return 0;
}

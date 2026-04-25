//===- test_xclbin_npu2.cpp -----------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Flash attention benchmark for NPU2 (xclbin format).
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdfloat>
#include <string>
#include <vector>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using DATATYPE = std::bfloat16_t;

static inline std::bfloat16_t random_bfloat16_t() {
  return std::bfloat16_t(4.0 * (float)rand() / (float)(RAND_MAX));
}

int main(int argc, const char *argv[]) {

  cxxopts::Options options("Allowed options");

  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN",
      cxxopts::value<std::string>())(
      "instr,i", "path of file containing instructions (instr.bin)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "lq", "Query sequence length",
      cxxopts::value<int>()->default_value("512"))(
      "lk", "Key/Value sequence length",
      cxxopts::value<int>()->default_value("512"))(
      "dk", "Key dimension", cxxopts::value<int>()->default_value("64"))(
      "dv", "Value dimension", cxxopts::value<int>()->default_value("64"))(
      "num-heads", "Number of attention heads",
      cxxopts::value<int>()->default_value("12"))(
      "warmup,w", "Number of warmup iterations",
      cxxopts::value<int>()->default_value("10"))(
      "iterations,n", "Number of iterations",
      cxxopts::value<int>()->default_value("20"))(
      "trace-size,t", "Trace buffer size in bytes (0 to disable tracing)",
      cxxopts::value<int>()->default_value("0"));

  auto vm = options.parse(argc, argv);

  if (vm.count("help")) {
    std::cout << options.help() << std::endl;
    return 1;
  }

  if (!vm.count("xclbin") || !vm.count("kernel") || !vm.count("instr")) {
    std::cerr << "Error: Required options --xclbin, --kernel, --instr\n\n";
    std::cerr << "Usage:\n" << options.help() << std::endl;
    return 1;
  }

  int trace_size = vm["trace-size"].as<int>();
  if (trace_size < 0) {
    std::cerr << "Error: --trace-size must be >= 0\n";
    return 1;
  }

  int lq = vm["lq"].as<int>();
  int lk = vm["lk"].as<int>();
  int dk = vm["dk"].as<int>();
  int dv = vm["dv"].as<int>();
  int num_heads = vm["num-heads"].as<int>();
  int verbosity = vm["verbosity"].as<int>();

  unsigned int n_iterations = vm["iterations"].as<int>();
  unsigned int n_warmup_iterations = vm["warmup"].as<int>();
  if (n_iterations == 0) {
    std::cerr << "Error: --iterations must be > 0\n";
    return 1;
  }

  size_t Q_VOLUME = (size_t)num_heads * lq * dk;
  size_t K_VOLUME = (size_t)num_heads * lk * dk;
  size_t V_VOLUME = (size_t)num_heads * lk * dv;
  size_t OUTPUT_VOLUME = (size_t)num_heads * lq * dv;

  size_t Q_SIZE = Q_VOLUME * sizeof(DATATYPE);
  size_t K_SIZE = K_VOLUME * sizeof(DATATYPE);
  size_t V_SIZE = V_VOLUME * sizeof(DATATYPE);
  size_t OUTPUT_SIZE = OUTPUT_VOLUME * sizeof(DATATYPE);

  // Load instruction binary
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Instruction count: " << instr_v.size() << "\n";

  // Get device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load xclbin
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  // Find kernel in xclbin
  std::string Node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto xkernel_it = std::find_if(xkernels.begin(), xkernels.end(),
                                 [Node, verbosity](xrt::xclbin::kernel &k) {
                                   auto name = k.get_name();
                                   if (verbosity >= 1)
                                     std::cout << "Found kernel: " << name
                                               << std::endl;
                                   return name.rfind(Node, 0) == 0;
                                 });
  if (xkernel_it == xkernels.end()) {
    std::cerr << "Error: Kernel '" << Node << "' not found in xclbin\n";
    return 1;
  }
  auto kernelName = xkernel_it->get_name();

  if (verbosity >= 1)
    std::cout << "Using kernel: " << kernelName << "\n";

  // Register xclbin and create context
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  // Allocate buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  auto bo_q =
      xrt::bo(device, Q_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_k =
      xrt::bo(device, K_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_v =
      xrt::bo(device, V_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_out = xrt::bo(device, OUTPUT_SIZE + static_cast<size_t>(trace_size),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  // Fill input data directly into mapped BO buffers
  DATATYPE *bufQ = bo_q.map<DATATYPE *>();
  for (size_t i = 0; i < Q_VOLUME; i++)
    bufQ[i] = random_bfloat16_t();

  DATATYPE *bufK = bo_k.map<DATATYPE *>();
  for (size_t i = 0; i < K_VOLUME; i++)
    bufK[i] = random_bfloat16_t();

  DATATYPE *bufV = bo_v.map<DATATYPE *>();
  for (size_t i = 0; i < V_VOLUME; i++)
    bufV[i] = random_bfloat16_t();

  DATATYPE *bufOut = bo_out.map<DATATYPE *>();
  memset(bufOut, 0, OUTPUT_SIZE + trace_size);

  // Copy instructions
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  // Sync to device
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_q.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_k.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_v.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = std::numeric_limits<float>::max();
  float npu_time_max = 0;

  // FLOPs: Q@K^T (lq*lk*dk*2) + S@V (lq*dv*lk*2) per head
  float macs =
      (float)num_heads * ((float)lq * lk * dk * 2 + (float)lk * lq * dv * 2);

  std::cout << "Flash Attention Benchmark (xclbin format, NPU2)" << std::endl;
  std::cout << "  num_heads=" << num_heads << ", lq=" << lq << ", lk=" << lk
            << ", dk=" << dk << ", dv=" << dv << std::endl;
  std::cout << "  Q: [" << num_heads << "x" << lq << "x" << dk << "] ("
            << Q_SIZE << " bytes)" << std::endl;
  std::cout << "  K: [" << num_heads << "x" << lk << "x" << dk << "] ("
            << K_SIZE << " bytes)" << std::endl;
  std::cout << "  V: [" << num_heads << "x" << lk << "x" << dv << "] ("
            << V_SIZE << " bytes)" << std::endl;
  std::cout << "  Output: [" << num_heads << "x" << lq << "x" << dv << "] ("
            << OUTPUT_SIZE << " bytes)" << std::endl;
  std::cout << "  Warmup: " << n_warmup_iterations
            << ", Iterations: " << n_iterations << std::endl;

  // Opcode 3 = execute NPU instructions from instruction buffer
  unsigned int opcode = 3;

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel (iteration " << iter << ").\n";

    auto start = std::chrono::high_resolution_clock::now();

    auto run =
        kernel(opcode, bo_instr, instr_v.size(), bo_q, bo_k, bo_v, bo_out);
    run.wait();

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

  if (verbosity >= 1)
    std::cout << "Done Running Kernel.\n";

  if (trace_size > 0) {
    test_utils::write_out_trace(((char *)bufOut) + OUTPUT_SIZE, trace_size,
                                "trace.txt");
  }

  std::cout << std::endl
            << "Avg NPU attention time: " << npu_time_total / n_iterations
            << "us." << std::endl;
  std::cout << "Avg NPU gflops: "
            << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU attention time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;

  std::cout << std::endl
            << "Max NPU attention time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

  return 0;
}

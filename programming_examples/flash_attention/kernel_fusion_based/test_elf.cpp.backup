//===- test_elf.cpp ---------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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

using DATATYPE = std::bfloat16_t;

static inline std::bfloat16_t random_bfloat16_t() {
  // Random numbers should NOT be uniformly between 0 and 1, because that
  // would make the matrix product AB always close to 1.
  return std::bfloat16_t(4.0 * (float)rand() / (float)(RAND_MAX));
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  cxxopts::Options options("Allowed options");

  options.add_options()("help,h", "produce help message")(
      "elf,e", "the input elf path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name (format: <kernel_name>:<instance_name>)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "lq", "Query sequence length",
      cxxopts::value<int>()->default_value("512"))(
      "lk", "Key/Value sequence length",
      cxxopts::value<int>()->default_value("12288"))(
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

  // Check required options
  if (!vm.count("elf") || !vm.count("kernel")) {
    std::cerr << "Error: Required options missing\n\n";
    std::cerr << "Usage:\n" << options.help() << std::endl;
    return 1;
  }

  // Get trace size from command line
  int trace_size = vm["trace-size"].as<int>();

  // Get dimensions from command line
  int lq = vm["lq"].as<int>();
  int lk = vm["lk"].as<int>();
  int dk = vm["dk"].as<int>();
  int dv = vm["dv"].as<int>();
  int num_heads = vm["num-heads"].as<int>();

  size_t Q_VOLUME = (size_t)num_heads * lq * dk;
  size_t K_VOLUME = (size_t)num_heads * dk * lk;
  size_t V_VOLUME = (size_t)num_heads * lk * dv;
  size_t M_VOLUME = (size_t)num_heads * lq * lk;
  size_t OUTPUT_VOLUME = (size_t)num_heads * lq * dv;

  size_t Q_SIZE = Q_VOLUME * sizeof(DATATYPE);
  size_t K_SIZE = K_VOLUME * sizeof(DATATYPE);
  size_t V_SIZE = V_VOLUME * sizeof(DATATYPE);
  size_t M_SIZE = M_VOLUME * sizeof(DATATYPE);
  size_t OUTPUT_SIZE = OUTPUT_VOLUME * sizeof(DATATYPE);

  int verbosity = vm["verbosity"].as<int>();

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the elf and create context
  std::string elfPath = vm["elf"].as<std::string>();
  if (verbosity >= 1)
    std::cout << "Loading elf: " << elfPath << "\n";

  xrt::elf ctx_elf{elfPath};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);

  // The name format here is <kernel_name>:<instance_name> from the config.json
  std::string kernelName = vm["kernel"].as<std::string>();
  if (verbosity >= 1)
    std::cout << "Kernel name: " << kernelName << "\n";

  auto kernel = xrt::ext::kernel(context, kernelName);

  // Create buffer objects using xrt::ext::bo (declared as xrt::bo type)
  xrt::bo bo_q = xrt::ext::bo{device, Q_SIZE};
  xrt::bo bo_k = xrt::ext::bo{device, K_SIZE};
  xrt::bo bo_v = xrt::ext::bo{device, V_SIZE};
  xrt::bo bo_m = xrt::ext::bo{device, M_SIZE};
  xrt::bo bo_out =
      xrt::ext::bo{device, OUTPUT_SIZE + static_cast<size_t>(trace_size)};

  unsigned n_iterations = vm["iterations"].as<int>();
  unsigned n_warmup_iterations = vm["warmup"].as<int>();
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = std::numeric_limits<float>::max();
  float npu_time_max = 0;

  // FLOPs for attention: Q@K^T (lq*lk*dk*2) + softmax(~5*lq*lk) + S@V
  // (lq*dv*lk*2) per head, multiplied by num_heads
  float macs =
      (float)num_heads * ((float)lq * lk * dk * 2 + (float)lk * lq * dv * 2);

  std::cout << "Flash Attention Benchmark (ELF format)" << std::endl;
  std::cout << "  num_heads=" << num_heads << ", lq=" << lq << ", lk=" << lk
            << ", dk=" << dk << ", dv=" << dv << std::endl;
  std::cout << "  Q: [" << num_heads << "x" << lq << "x" << dk << "] ("
            << Q_SIZE << " bytes)" << std::endl;
  std::cout << "  K: [" << num_heads << "x" << dk << "x" << lk << "] ("
            << K_SIZE << " bytes)" << std::endl;
  std::cout << "  V: [" << num_heads << "x" << lk << "x" << dv << "] ("
            << V_SIZE << " bytes)" << std::endl;
  std::cout << "  Output: [" << num_heads << "x" << lq << "x" << dv << "] ("
            << OUTPUT_SIZE << " bytes)" << std::endl;

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  DATATYPE *bufQ = bo_q.map<DATATYPE *>();
  std::vector<DATATYPE> QVec;
  for (size_t i = 0; i < Q_VOLUME; i++)
    QVec.push_back(random_bfloat16_t());
  memcpy(bufQ, QVec.data(), (QVec.size() * sizeof(DATATYPE)));

  DATATYPE *bufK = bo_k.map<DATATYPE *>();
  std::vector<DATATYPE> KVec;
  for (size_t i = 0; i < K_VOLUME; i++)
    KVec.push_back(random_bfloat16_t());
  memcpy(bufK, KVec.data(), (KVec.size() * sizeof(DATATYPE)));

  DATATYPE *bufV = bo_v.map<DATATYPE *>();
  std::vector<DATATYPE> VVec;
  for (size_t i = 0; i < V_VOLUME; i++)
    VVec.push_back(random_bfloat16_t());
  memcpy(bufV, VVec.data(), (VVec.size() * sizeof(DATATYPE)));

  DATATYPE *bufM = bo_m.map<DATATYPE *>();
  std::vector<DATATYPE> MVec;
  for (size_t i = 0; i < M_VOLUME; i++)
    MVec.push_back(std::bfloat16_t(0.0f)); // Mask initialized to zero
  memcpy(bufM, MVec.data(), (MVec.size() * sizeof(DATATYPE)));

  DATATYPE *bufOut = bo_out.map<DATATYPE *>();
  memset(bufOut, 0, OUTPUT_SIZE + trace_size);

  bo_q.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_k.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_v.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_m.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel (iteration " << iter << ").\n";

    auto run = xrt::run(kernel);
    run.set_arg(0, bo_q);
    run.set_arg(1, bo_k);
    run.set_arg(2, bo_v);
    run.set_arg(3, bo_m);
    run.set_arg(4, bo_out);

    auto start = std::chrono::high_resolution_clock::now();
    run.start();
    run.wait2();
    auto stop = std::chrono::high_resolution_clock::now();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

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

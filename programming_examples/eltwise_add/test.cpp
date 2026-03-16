//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
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
#include <stdfloat>
#include <vector>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using DATATYPE = std::bfloat16_t;

void add_default_options(cxxopts::Options &options) {
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions to be sent to the LX6",
      cxxopts::value<std::string>())("size,S", "Total number of elements",
                                     cxxopts::value<int>());
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  cxxopts::Options options("Allowed options");
  cxxopts::ParseResult vm;
  add_default_options(options);
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  int SIZE = vm["size"].as<int>();
  int DATA_SIZE = SIZE * sizeof(DATATYPE);

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
  auto bo_a =
      xrt::bo(device, DATA_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, DATA_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c =
      xrt::bo(device, DATA_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Fill input buffers with random data
  DATATYPE *bufA = bo_a.map<DATATYPE *>();
  std::vector<DATATYPE> AVec(SIZE);
  for (int i = 0; i < SIZE; i++) {
    AVec[i] = DATATYPE(4.0f * rand() / RAND_MAX);
  }
  memcpy(bufA, AVec.data(), AVec.size() * sizeof(DATATYPE));

  DATATYPE *bufB = bo_b.map<DATATYPE *>();
  std::vector<DATATYPE> BVec(SIZE);
  for (int i = 0; i < SIZE; i++) {
    BVec[i] = DATATYPE(4.0f * rand() / RAND_MAX);
  }
  memcpy(bufB, BVec.data(), BVec.size() * sizeof(DATATYPE));

  DATATYPE *bufC = bo_c.map<DATATYPE *>();
  std::vector<DATATYPE> CVec(SIZE, 0);
  memcpy(bufC, CVec.data(), CVec.size() * sizeof(DATATYPE));

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

  // Bandwidth: 2 reads + 1 write = 3 * SIZE * sizeof(DATATYPE) bytes
  float total_bytes = 3.0f * SIZE * sizeof(DATATYPE);
  bool verified = false;

  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel.\n";
    }
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    // Verify correctness on first measurement iteration
    if (!verified) {
      memcpy(CVec.data(), bufC, CVec.size() * sizeof(DATATYPE));
      int errors = 0;
      for (int i = 0; i < SIZE; i++) {
        float expected = float(AVec[i]) + float(BVec[i]);
        float got = float(CVec[i]);
        float tol = std::max(0.05f, 0.01f * std::abs(expected));
        if (std::abs(got - expected) > tol) {
          if (errors < 10) {
            std::cout << "Error at index " << i << ": got " << got
                      << ", expected " << expected << std::endl;
          }
          errors++;
        }
      }
      if (errors > 0) {
        std::cout << "FAIL: " << errors << " / " << SIZE
                  << " elements incorrect." << std::endl;
      } else {
        std::cout << "PASS: All " << SIZE << " elements correct." << std::endl;
      }
      verified = true;
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  std::cout << std::endl
            << "Problem size: " << SIZE << " elements ("
            << (SIZE * sizeof(DATATYPE)) / 1024 << " KB per buffer)"
            << std::endl;

  std::cout << std::endl
            << "Avg NPU eltwise_add time: " << npu_time_total / n_iterations
            << "us." << std::endl;
  std::cout << "Avg bandwidth: "
            << total_bytes / (1000.0f * npu_time_total / n_iterations)
            << " GB/s" << std::endl;

  std::cout << std::endl
            << "Min NPU eltwise_add time: " << npu_time_min << "us."
            << std::endl;
  std::cout << "Max bandwidth: " << total_bytes / (1000.0f * npu_time_min)
            << " GB/s" << std::endl;

  std::cout << std::endl
            << "Max NPU eltwise_add time: " << npu_time_max << "us."
            << std::endl;
  std::cout << "Min bandwidth: " << total_bytes / (1000.0f * npu_time_max)
            << " GB/s" << std::endl;

  return 0;
}

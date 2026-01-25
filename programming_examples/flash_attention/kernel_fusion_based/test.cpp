//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

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
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions to be sent to the LX6",
      cxxopts::value<std::string>())(
      "lq", "Query sequence length",
      cxxopts::value<int>()->default_value("128"))(
      "lk", "Key/Value sequence length",
      cxxopts::value<int>()->default_value("12288"))(
      "dk", "Key dimension", cxxopts::value<int>()->default_value("64"))(
      "dv", "Value dimension", cxxopts::value<int>()->default_value("64"))(
      "warmup,w", "Number of warmup iterations",
      cxxopts::value<int>()->default_value("0"))(
      "iterations,n", "Number of iterations",
      cxxopts::value<int>()->default_value("1"))(
      "trace-size,t", "Trace buffer size in bytes (0 to disable tracing)",
      cxxopts::value<int>()->default_value("0"));

  auto vm = options.parse(argc, argv);

  if (vm.count("help")) {
    std::cout << options.help() << std::endl;
    return 1;
  }

  // Check required options
  if (!vm.count("xclbin") || !vm.count("kernel") || !vm.count("instr")) {
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

  int Q_VOLUME = lq * dk;
  int K_VOLUME = dk * lk;
  int V_VOLUME = lk * dv;
  int M_VOLUME = lq * lk;
  int OUTPUT_VOLUME = lq * dv;

  int Q_SIZE = Q_VOLUME * sizeof(DATATYPE);
  int K_SIZE = K_VOLUME * sizeof(DATATYPE);
  int V_SIZE = V_VOLUME * sizeof(DATATYPE);
  int M_SIZE = M_VOLUME * sizeof(DATATYPE);
  int OUTPUT_SIZE = OUTPUT_VOLUME * sizeof(DATATYPE);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
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
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
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
  auto bo_q =
      xrt::bo(device, Q_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_k =
      xrt::bo(device, K_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_v =
      xrt::bo(device, V_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_m =
      xrt::bo(device, M_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  auto bo_out = xrt::bo(device, OUTPUT_SIZE + trace_size,
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  unsigned n_iterations = vm["iterations"].as<int>();
  unsigned n_warmup_iterations = vm["warmup"].as<int>();
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = std::numeric_limits<float>::max();
  float npu_time_max = 0;

  // FLOPs for attention: Q@K^T (lq*lk*dk*2) + softmax(~5*lq*lk) + S@V
  // (lq*dv*lk*2)
  float macs = (float)lq * lk * dk * 2 + (float)lk * lq * dv * 2;

  std::cout << "Flash Attention Benchmark" << std::endl;
  std::cout << "  lq=" << lq << ", lk=" << lk << ", dk=" << dk << ", dv=" << dv
            << std::endl;
  std::cout << "  Q: " << lq << "x" << dk << " (" << Q_SIZE << " bytes)"
            << std::endl;
  std::cout << "  K: " << dk << "x" << lk << " (" << K_SIZE << " bytes)"
            << std::endl;
  std::cout << "  V: " << lk << "x" << dv << " (" << V_SIZE << " bytes)"
            << std::endl;
  std::cout << "  Output: " << lq << "x" << dv << " (" << OUTPUT_SIZE
            << " bytes)" << std::endl;

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";
  DATATYPE *bufQ = bo_q.map<DATATYPE *>();
  std::vector<DATATYPE> QVec;
  for (int i = 0; i < Q_VOLUME; i++)
    QVec.push_back(random_bfloat16_t());
  memcpy(bufQ, QVec.data(), (QVec.size() * sizeof(DATATYPE)));

  DATATYPE *bufK = bo_k.map<DATATYPE *>();
  std::vector<DATATYPE> KVec;
  for (int i = 0; i < K_VOLUME; i++)
    KVec.push_back(random_bfloat16_t());
  memcpy(bufK, KVec.data(), (KVec.size() * sizeof(DATATYPE)));

  DATATYPE *bufV = bo_v.map<DATATYPE *>();
  std::vector<DATATYPE> VVec;
  for (int i = 0; i < V_VOLUME; i++)
    VVec.push_back(random_bfloat16_t());
  memcpy(bufV, VVec.data(), (VVec.size() * sizeof(DATATYPE)));

  DATATYPE *bufM = bo_m.map<DATATYPE *>();
  std::vector<DATATYPE> MVec;
  for (int i = 0; i < M_VOLUME; i++)
    MVec.push_back(std::bfloat16_t(0.0f)); // Mask initialized to zero
  memcpy(bufM, MVec.data(), (MVec.size() * sizeof(DATATYPE)));

  DATATYPE *bufOut = bo_out.map<DATATYPE *>();
  memset(bufOut, 0, OUTPUT_SIZE + trace_size);

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_q.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_k.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_v.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_m.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel (iteration " << iter << ").\n";
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_q, bo_k, bo_v, bo_m,
                      bo_out);
    run.wait();
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

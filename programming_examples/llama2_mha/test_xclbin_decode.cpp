//===- test_xclbin_decode.cpp -----------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Decode flash attention benchmark (xclbin format) for NPU2.
// Mirrors programming_examples/flash_attention/kernel_fusion_based/
// test_xclbin.cpp but for the decode-side mha_gqa_multicol.py kernel
// signature: mha_bf16(xrms[2,k], W[NKV,GEMV_COUNT,k,n],
//                    K_cache[NKV,lk,n], V_cache[NKV,lk,n],
//                    xb[NKV,GROUP_SIZE,n]).
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
  return std::bfloat16_t(4.0f * (float)rand() / (float)(RAND_MAX));
}

int main(int argc, const char *argv[]) {

  cxxopts::Options options("Allowed options");

  options.add_options()
      ("help,h", "produce help message")
      ("xclbin,x", "input xclbin path", cxxopts::value<std::string>())
      ("kernel,k", "kernel name in the XCLBIN", cxxopts::value<std::string>())
      ("instr,i", "instructions binary path", cxxopts::value<std::string>())
      ("verbosity,v", "verbosity",
       cxxopts::value<int>()->default_value("0"))
      ("nkv", "Number of KV heads (cols)",
       cxxopts::value<int>()->default_value("8"))
      ("group-size", "GQA group size (Q heads per KV head)",
       cxxopts::value<int>()->default_value("4"))
      ("n-in", "GEMV input dim (n_in / hidden_size)",
       cxxopts::value<int>()->default_value("64"))
      ("head-dim", "Head dim (dk = dv)",
       cxxopts::value<int>()->default_value("64"))
      ("lk", "K/V cache length (sequence length)",
       cxxopts::value<int>()->default_value("2048"))
      ("warmup,w", "warmup iterations",
       cxxopts::value<int>()->default_value("10"))
      ("iterations,n", "profile iterations",
       cxxopts::value<int>()->default_value("20"))
      ("trace-size,t", "trace buffer bytes (0 disables)",
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

  int nkv = vm["nkv"].as<int>();
  int group_size = vm["group-size"].as<int>();
  int kdim = vm["n-in"].as<int>();   // GEMV input (n_in / hidden_size)
  int hdim = vm["head-dim"].as<int>(); // Head dim (dk = dv)
  int lk = vm["lk"].as<int>();
  int verbosity = vm["verbosity"].as<int>();
  int gemv_count = group_size + 2; // group_size Q heads + 1 K + 1 V

  unsigned int n_iterations = vm["iterations"].as<int>();
  unsigned int n_warmup_iterations = vm["warmup"].as<int>();
  if (n_iterations == 0) {
    std::cerr << "Error: --iterations must be > 0\n";
    return 1;
  }

  size_t XRMS_VOLUME = (size_t)2 * kdim;
  size_t W_VOLUME = (size_t)nkv * gemv_count * kdim * hdim;
  size_t KC_VOLUME = (size_t)nkv * lk * hdim;
  size_t VC_VOLUME = (size_t)nkv * lk * hdim;
  size_t XB_VOLUME = (size_t)nkv * group_size * hdim;

  size_t XRMS_SIZE = XRMS_VOLUME * sizeof(DATATYPE);
  size_t W_SIZE = W_VOLUME * sizeof(DATATYPE);
  size_t KC_SIZE = KC_VOLUME * sizeof(DATATYPE);
  size_t VC_SIZE = VC_VOLUME * sizeof(DATATYPE);
  size_t XB_SIZE = XB_VOLUME * sizeof(DATATYPE);

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

  // Find kernel
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

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  // Allocate buffer objects: instr + 5 kernel args.
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  auto bo_xrms =
      xrt::bo(device, XRMS_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_w =
      xrt::bo(device, W_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_kc =
      xrt::bo(device, KC_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_vc =
      xrt::bo(device, VC_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  auto bo_xb = xrt::bo(device, XB_SIZE + (size_t)trace_size,
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  // Fill input data with random bf16.
  DATATYPE *bufXrms = bo_xrms.map<DATATYPE *>();
  for (size_t i = 0; i < XRMS_VOLUME; i++)
    bufXrms[i] = random_bfloat16_t();
  DATATYPE *bufW = bo_w.map<DATATYPE *>();
  for (size_t i = 0; i < W_VOLUME; i++)
    bufW[i] = random_bfloat16_t();
  DATATYPE *bufKc = bo_kc.map<DATATYPE *>();
  for (size_t i = 0; i < KC_VOLUME; i++)
    bufKc[i] = random_bfloat16_t();
  DATATYPE *bufVc = bo_vc.map<DATATYPE *>();
  for (size_t i = 0; i < VC_VOLUME; i++)
    bufVc[i] = random_bfloat16_t();
  DATATYPE *bufXb = bo_xb.map<DATATYPE *>();
  memset(bufXb, 0, XB_SIZE + trace_size);

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_xrms.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_kc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_vc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_xb.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = std::numeric_limits<float>::max();
  float npu_time_max = 0;

  // FLOPs for one decode token at this config (rough lower bound):
  //   GEMV : nkv * gemv_count * kdim * hdim * 2
  //   Attn : nkv * group_size * lk * hdim * 4   (Q.K + softmax*V)
  //   RMSNorm/RoPE: small, ignored.
  float macs_gemv =
      (float)nkv * (float)gemv_count * (float)kdim * (float)hdim * 2.0f;
  float macs_attn =
      (float)nkv * (float)group_size * (float)lk * (float)hdim * 4.0f;
  float macs = macs_gemv + macs_attn;

  std::cout << "Decode Attention Benchmark (xclbin format)" << std::endl;
  std::cout << "  nkv=" << nkv << " group_size=" << group_size
            << " kdim(n_in)=" << kdim << " hdim(dk=dv)=" << hdim
            << " lk=" << lk << std::endl;
  std::cout << "  xrms: [2x" << kdim << "] (" << XRMS_SIZE << " B)\n"
            << "  W:    [" << nkv << "x" << gemv_count << "x" << kdim << "x"
            << hdim << "] (" << W_SIZE << " B)\n"
            << "  Kc:   [" << nkv << "x" << lk << "x" << hdim << "] ("
            << KC_SIZE << " B)\n"
            << "  Vc:   [" << nkv << "x" << lk << "x" << hdim << "] ("
            << VC_SIZE << " B)\n"
            << "  xb:   [" << nkv << "x" << group_size << "x" << hdim << "] ("
            << XB_SIZE << " B)" << std::endl;
  std::cout << "  Warmup: " << n_warmup_iterations
            << ", Iterations: " << n_iterations << std::endl;

  unsigned int opcode = 3;

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel (iteration " << iter << ").\n";

    auto start = std::chrono::high_resolution_clock::now();

    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_xrms, bo_w, bo_kc,
                      bo_vc, bo_xb);
    run.wait();

    auto stop = std::chrono::high_resolution_clock::now();

    bo_xb.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations)
      continue;

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  if (trace_size > 0) {
    test_utils::write_out_trace(((char *)bufXb) + XB_SIZE, trace_size,
                                "trace.txt");
  }

  std::cout << std::endl
            << "Avg NPU decode time: " << npu_time_total / n_iterations
            << " us." << std::endl;
  std::cout << "Avg NPU gflops: "
            << macs / (1000 * npu_time_total / n_iterations) << std::endl;
  std::cout << std::endl
            << "Min NPU decode time: " << npu_time_min << " us." << std::endl;
  std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;
  std::cout << std::endl
            << "Max NPU decode time: " << npu_time_max << " us." << std::endl;
  std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

  return 0;
}

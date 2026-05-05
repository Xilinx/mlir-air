//===- test_elf_npu2_decode.cpp ---------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Decode flash attention benchmark (ELF format) for NPU2. Mirrors prefill
// flash_attention/kernel_fusion_based/test_elf_npu2.cpp but for the decode
// kernel signature: mha_bf16(xrms[2,k], W[NKV,GEMV_COUNT,k,n],
//                            K_cache[NKV,lk,n], V_cache[NKV,lk,n],
//                            xb[NKV,GROUP_SIZE,n]).
//
// ELF format has the lightweight reset device that resets BD state between
// host invocations — required for the decode design which generates a
// memtile DMA chain with `repeat_count` on the weight forwarding path.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <chrono>
#include <cstdint>
#include <cstring>
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

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>

using DATATYPE = std::bfloat16_t;

static inline std::bfloat16_t random_bfloat16_t() {
  return std::bfloat16_t(4.0f * (float)rand() / (float)(RAND_MAX));
}

int main(int argc, const char *argv[]) {

  cxxopts::Options options("Allowed options");

  options.add_options()
      ("help,h", "produce help message")
      ("elf,e", "input elf path", cxxopts::value<std::string>())
      ("kernel,k", "kernel name (format: <kernel_name>:<instance_name>)",
       cxxopts::value<std::string>())
      ("verbosity,v", "verbosity",
       cxxopts::value<int>()->default_value("0"))
      ("nkv", "Number of KV heads (cols)",
       cxxopts::value<int>()->default_value("8"))
      ("group-size", "GQA group size",
       cxxopts::value<int>()->default_value("4"))
      ("n-in", "GEMV input dim (n_in / hidden_size)",
       cxxopts::value<int>()->default_value("64"))
      ("head-dim", "Head dim (dk = dv)",
       cxxopts::value<int>()->default_value("64"))
      ("lk", "K/V cache length (sequence length)",
       cxxopts::value<int>()->default_value("2048"))
      ("tile-k", "GEMV inner-k tile (xrms padded to [tile_k, tile_n])",
       cxxopts::value<int>()->default_value("128"))
      ("tile-n", "GEMV output tile (xrms padded to [tile_k, tile_n])",
       cxxopts::value<int>()->default_value("64"))
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
  if (!vm.count("elf") || !vm.count("kernel")) {
    std::cerr << "Error: Required options --elf, --kernel\n\n";
    std::cerr << "Usage:\n" << options.help() << std::endl;
    return 1;
  }

  int trace_size = vm["trace-size"].as<int>();
  int nkv = vm["nkv"].as<int>();
  int group_size = vm["group-size"].as<int>();
  int kdim = vm["n-in"].as<int>();
  int hdim = vm["head-dim"].as<int>();
  int lk = vm["lk"].as<int>();
  int tile_k = vm["tile-k"].as<int>();
  int tile_n = vm["tile-n"].as<int>();
  int verbosity = vm["verbosity"].as<int>();
  int gemv_count = group_size + 2;

  // xrms is padded to [tile_k, tile_n] so it shares one BD shape with
  // weight chunks on bL3ToL2 (single self-loop, no repeat_count). The
  // first 2*kdim flat elements hold real x_raw + w_rms; the rest is
  // padding the kernel demux ignores.
  size_t XRMS_VOLUME = (size_t)tile_k * tile_n;
  size_t W_VOLUME = (size_t)nkv * gemv_count * kdim * hdim;
  size_t KC_VOLUME = (size_t)nkv * lk * hdim;
  size_t VC_VOLUME = (size_t)nkv * lk * hdim;
  size_t XB_VOLUME = (size_t)nkv * group_size * hdim;

  size_t XRMS_SIZE = XRMS_VOLUME * sizeof(DATATYPE);
  size_t W_SIZE = W_VOLUME * sizeof(DATATYPE);
  size_t KC_SIZE = KC_VOLUME * sizeof(DATATYPE);
  size_t VC_SIZE = VC_VOLUME * sizeof(DATATYPE);
  size_t XB_SIZE = XB_VOLUME * sizeof(DATATYPE);

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

  xrt::bo bo_xrms = xrt::ext::bo{device, XRMS_SIZE};
  xrt::bo bo_w = xrt::ext::bo{device, W_SIZE};
  xrt::bo bo_kc = xrt::ext::bo{device, KC_SIZE};
  xrt::bo bo_vc = xrt::ext::bo{device, VC_SIZE};
  xrt::bo bo_xb =
      xrt::ext::bo{device, XB_SIZE + static_cast<size_t>(trace_size)};

  unsigned n_iterations = vm["iterations"].as<int>();
  unsigned n_warmup_iterations = vm["warmup"].as<int>();
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = std::numeric_limits<float>::max();
  float npu_time_max = 0;

  // FLOPs lower bound (per token): GEMV + Attn (RMSNorm/RoPE small, ignored).
  float macs_gemv =
      (float)nkv * (float)gemv_count * (float)kdim * (float)hdim * 2.0f;
  float macs_attn =
      (float)nkv * (float)group_size * (float)lk * (float)hdim * 4.0f;
  float macs = macs_gemv + macs_attn;

  std::cout << "Decode Attention Benchmark (ELF format)" << std::endl;
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

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

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

  bo_xrms.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_kc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_vc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_xb.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel (iteration " << iter << ").\n";

    auto run = xrt::run(kernel);
    run.set_arg(0, bo_xrms);
    run.set_arg(1, bo_w);
    run.set_arg(2, bo_kc);
    run.set_arg(3, bo_vc);
    run.set_arg(4, bo_xb);

    auto start = std::chrono::high_resolution_clock::now();
    run.start();
    run.wait2();
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

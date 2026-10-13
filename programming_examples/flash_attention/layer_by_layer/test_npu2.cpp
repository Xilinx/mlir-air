//===- test_npu2.cpp ------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Layer-by-layer attention benchmark: qk -> sm -> pv, three binaries in three
// hardware contexts, S and P in DDR. One timed iteration runs all three.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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
  // Not uniform in [0, 1): that would make every score close to the mean.
  return std::bfloat16_t(4.0 * (float)rand() / (float)(RAND_MAX));
}

namespace {

struct Op {
  bool elf = false;
  xrt::hw_context ctx;
  xrt::kernel kernel;
  xrt::bo instr;
  size_t ninstr = 0;

  void load(xrt::device &device, const std::string &dir, bool use_elf) {
    elf = use_elf;
    if (elf) {
      xrt::elf ctx_elf{dir + "/air.elf"};
      ctx = xrt::hw_context(device, ctx_elf);
      kernel = xrt::ext::kernel(ctx, "main:attention_bf16");
      return;
    }
    auto xclbin = xrt::xclbin(dir + "/air.xclbin");
    device.register_xclbin(xclbin);
    ctx = xrt::hw_context(device, xclbin.get_uuid());
    std::string name;
    for (auto &k : xclbin.get_kernels())
      if (k.get_name().rfind("MLIR_AIE", 0) == 0)
        name = k.get_name();
    kernel = xrt::kernel(ctx, name);
    auto v = test_utils::load_instr_binary(dir + "/air.insts.bin");
    instr = xrt::bo(device, v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
                    kernel.group_id(1));
    memcpy(instr.map<void *>(), v.data(), v.size() * sizeof(uint32_t));
    instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ninstr = v.size();
  }

  void run(xrt::bo &a, xrt::bo &b, xrt::bo &c) {
    if (elf) {
      auto r = xrt::run(kernel);
      r.set_arg(0, a);
      r.set_arg(1, b);
      r.set_arg(2, c);
      r.start();
      r.wait2();
      return;
    }
    kernel(3, instr, ninstr, a, b, c).wait();
  }

  void run(xrt::bo &a, xrt::bo &b) {
    if (elf) {
      auto r = xrt::run(kernel);
      r.set_arg(0, a);
      r.set_arg(1, b);
      r.start();
      r.wait2();
      return;
    }
    kernel(3, instr, ninstr, a, b).wait();
  }
};

} // namespace

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Allowed options");
  options.add_options()("help,h", "produce help message")(
      "dir,d", "directory holding qk/, sm/ and pv/",
      cxxopts::value<std::string>()->default_value("."))(
      "format,f", "xclbin or elf",
      cxxopts::value<std::string>()->default_value("elf"))(
      "lq", "Query sequence length",
      cxxopts::value<int>()->default_value("512"))(
      "lk", "Key/value sequence length",
      cxxopts::value<int>()->default_value("512"))(
      "dk", "Key dimension", cxxopts::value<int>()->default_value("64"))(
      "dv", "Value dimension", cxxopts::value<int>()->default_value("64"))(
      "num-heads", "Number of heads",
      cxxopts::value<int>()->default_value("2"))(
      "warmup", "Number of warmup iterations",
      cxxopts::value<int>()->default_value("10"))(
      "iterations", "Number of timed iterations",
      cxxopts::value<int>()->default_value("20"));
  auto vm = options.parse(argc, argv);
  if (vm.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }

  std::string dir = vm["dir"].as<std::string>();
  std::string format = vm["format"].as<std::string>();
  if (format != "xclbin" && format != "elf") {
    std::cerr << "--format must be xclbin or elf\n";
    return 1;
  }
  bool use_elf = format == "elf";
  size_t lq = vm["lq"].as<int>(), lk = vm["lk"].as<int>();
  size_t dk = vm["dk"].as<int>(), dv = vm["dv"].as<int>();
  size_t num_heads = vm["num-heads"].as<int>();
  int n_warmup = vm["warmup"].as<int>();
  int n_iterations = vm["iterations"].as<int>();

  size_t nqb = lq / 64, nkb = lk / 64;
  size_t q_size = num_heads * lq * dk * sizeof(DATATYPE);
  size_t k_size = num_heads * lk * dk * sizeof(DATATYPE);
  size_t v_size = num_heads * lk * dv * sizeof(DATATYPE);
  size_t o_size = num_heads * lq * dv * sizeof(DATATYPE);
  size_t s_size = num_heads * nkb * nqb * 64 * 64 * sizeof(DATATYPE);

  auto device = xrt::device(0);
  Op qk, sm, pv;
  qk.load(device, dir + "/qk", use_elf);
  sm.load(device, dir + "/sm", use_elf);
  pv.load(device, dir + "/pv", use_elf);

  auto make_bo = [&](size_t size) {
    if (use_elf)
      return xrt::bo(xrt::ext::bo{device, size});
    return xrt::bo(device, size, XRT_BO_FLAGS_HOST_ONLY, qk.kernel.group_id(3));
  };
  xrt::bo bo_q = make_bo(q_size), bo_k = make_bo(k_size),
          bo_v = make_bo(v_size), bo_o = make_bo(o_size),
          bo_s = make_bo(s_size), bo_p = make_bo(s_size);
  for (auto [bo, size] : {std::pair{&bo_q, q_size}, std::pair{&bo_k, k_size},
                          std::pair{&bo_v, v_size}}) {
    DATATYPE *buf = bo->map<DATATYPE *>();
    for (size_t i = 0; i < size / sizeof(DATATYPE); i++)
      buf[i] = random_bfloat16_t();
    bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  std::cout << "Layer-by-layer attention benchmark (" << format << ")\n"
            << "  num_heads=" << num_heads << ", lq=" << lq << ", lk=" << lk
            << ", dk=" << dk << ", dv=" << dv << "\n";

  float macs =
      (float)num_heads * ((float)lq * lk * dk * 2 + (float)lk * lq * dv * 2);
  float t_total = 0, t_min = std::numeric_limits<float>::max(), t_max = 0;
  for (int it = 0; it < n_warmup + n_iterations; it++) {
    auto start = std::chrono::high_resolution_clock::now();
    qk.run(bo_q, bo_k, bo_s);
    sm.run(bo_s, bo_p);
    pv.run(bo_p, bo_v, bo_o);
    auto stop = std::chrono::high_resolution_clock::now();
    if (it < n_warmup)
      continue;
    float t =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    t_total += t;
    t_min = std::min(t_min, t);
    t_max = std::max(t_max, t);
  }
  if (n_iterations <= 0)
    return 0;

  std::cout << "\nAvg NPU attention time: " << t_total / n_iterations << "us.\n"
            << "Avg NPU gflops: " << macs / (1000 * t_total / n_iterations)
            << "\n\nMin NPU attention time: " << t_min << "us.\n"
            << "Max NPU gflops: " << macs / (1000 * t_min)
            << "\n\nMax NPU attention time: " << t_max << "us.\n"
            << "Min NPU gflops: " << macs / (1000 * t_max) << "\n";
  return 0;
}

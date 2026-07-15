//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// C++ profiling harness for the conv2d_14x14 design.
//
// Correctness is verified by the Python flow (conv2d_14x14.py via XRTRunner,
// which uses bit-exact np.array_equal on i8). This harness only measures
// steady-state NPU latency over an xclbin/insts pair that the Python flow
// has already produced.
//
// Works for both the npu2 (8x4 herd) and npu1 (4x4 herd) builds: the
// total IN/WT/OUT buffer sizes are device-independent because the same
// total work (CO=1152, H=W=896) is partitioned across either 8 cols * 9
// oc-groups or 4 cols * 18 oc-groups. The --device flag is purely
// informational and prints the herd layout.

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

using IN_DATATYPE =
    int8_t; // i8 acts (host-side uint8 bytes packed identically)
using WT_DATATYPE = int8_t;  // i8 weights
using OUT_DATATYPE = int8_t; // i8 outputs

// Problem dims (fixed by the design; see conv2d_14x14.py).
constexpr int CI = 4;
constexpr int CO = 1152;
constexpr int H = 896;
constexpr int W = 896;
constexpr int KSZ = 14;

// Total buffer sizes (device-independent: same on npu2 and npu1).
constexpr int IN_VOLUME = CI * H * W;                  // 3,211,264 B
constexpr int WT_VOLUME = CO * CI * KSZ * KSZ;         // 903,168 B
constexpr int OUT_VOLUME = CO * (H / KSZ) * (W / KSZ); // 4,718,592 B

void add_default_options(cxxopts::Options &options) {
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance MLIR_AIE)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions to be sent to the LX6",
      cxxopts::value<std::string>())(
      "iters", "number of timed iterations",
      cxxopts::value<int>()->default_value("20"))(
      "warmup", "number of warmup iterations",
      cxxopts::value<int>()->default_value("10"))(
      "device,d",
      "target device label (npu1 or npu2) - informational only; total BO "
      "sizes are identical on both",
      cxxopts::value<std::string>()->default_value("npu2"));
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Allowed options");
  cxxopts::ParseResult vm;
  add_default_options(options);
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  const std::string device_label = vm["device"].as<std::string>();
  if (device_label != "npu1" && device_label != "npu2") {
    std::cerr << "Unknown --device '" << device_label
              << "' (expected npu1 or npu2)\n";
    return 1;
  }
  const int n_cols = (device_label == "npu1") ? 4 : 8;
  const int num_g = (device_label == "npu1") ? 18 : 9;
  if (verbosity >= 0)
    std::cout << "Device: " << device_label << " (" << n_cols
              << " cols x 4 rows, " << num_g << " oc-groups per col)\n";

  srand(0);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  std::string Node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1)
                                   std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, IN_VOLUME * sizeof(IN_DATATYPE),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_wt = xrt::bo(device, WT_VOLUME * sizeof(WT_DATATYPE),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_VOLUME * sizeof(OUT_DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  IN_DATATYPE *bufIn = bo_in.map<IN_DATATYPE *>();
  for (int i = 0; i < IN_VOLUME; i++)
    bufIn[i] = static_cast<IN_DATATYPE>(i & 0xff);
  WT_DATATYPE *bufWt = bo_wt.map<WT_DATATYPE *>();
  for (int i = 0; i < WT_VOLUME; i++)
    bufWt[i] = static_cast<WT_DATATYPE>((i % 18) + 2);
  OUT_DATATYPE *bufOut = bo_out.map<OUT_DATATYPE *>();
  std::memset(bufOut, 0, OUT_VOLUME * sizeof(OUT_DATATYPE));

  void *bufInstr = bo_instr.map<void *>();
  std::memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_wt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_iterations = vm["iters"].as<int>();
  unsigned n_warmup_iterations = vm["warmup"].as<int>();
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = std::numeric_limits<float>::max();
  float npu_time_max = 0;

  // KSZxKSZ stride-KSZ conv: per output = KSZ*KSZ*CI MACs;
  //   outputs = (H/KSZ) * (W/KSZ) * CO; *2 for mul+add per MAC.
  double macs = 2.0 * double(KSZ) * double(KSZ) * double(CI) * double(H / KSZ) *
                double(W / KSZ) * double(CO);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_wt, bo_out);
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

  std::cout << "\nAvg NPU conv time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  std::cout << "Avg NPU gops: " << macs / (1000 * npu_time_total / n_iterations)
            << std::endl;
  std::cout << "\nMin NPU conv time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU gops: " << macs / (1000 * npu_time_min) << std::endl;
  std::cout << "\nMax NPU conv time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU gops: " << macs / (1000 * npu_time_max) << std::endl;

  return 0;
}

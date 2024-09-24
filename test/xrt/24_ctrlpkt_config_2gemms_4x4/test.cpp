//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "experimental/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "matrix_multiplication.h"

constexpr int M = 512;
constexpr int K1 = 512;
constexpr int K2 = 1024;
constexpr int N = 512;

constexpr int A_VOLUME_1 = M * K1;
constexpr int B_VOLUME_1 = N * K1;
constexpr int C_VOLUME_1 = M * N;
constexpr int A_VOLUME_2 = M * K2;
constexpr int B_VOLUME_2 = N * K2;
constexpr int C_VOLUME_2 = M * N;

using A_DATATYPE = std::bfloat16_t;
using B_DATATYPE = std::bfloat16_t;
using C_DATATYPE = std::bfloat16_t;

constexpr int A_SIZE_1 = (A_VOLUME_1 * sizeof(A_DATATYPE));
constexpr int B_SIZE_1 = (B_VOLUME_1 * sizeof(B_DATATYPE));
constexpr int C_SIZE_1 = (C_VOLUME_1 * sizeof(C_DATATYPE));

constexpr int A_SIZE_2 = (A_VOLUME_2 * sizeof(A_DATATYPE));
constexpr int B_SIZE_2 = (B_VOLUME_2 * sizeof(B_DATATYPE));
constexpr int C_SIZE_2 = (C_VOLUME_2 * sizeof(C_DATATYPE));

constexpr bool VERIFY = true;

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  po::variables_map vm;
  matmul_common::add_default_options(desc);
  matmul_common::parse_options(argc, argv, desc, vm);
  int verbosity = vm["verbosity"].as<int>();

  srand(time(NULL));

  std::vector<uint32_t> instr1_v =
      matmul_common::load_instr_sequence("aie_run_seq.txt");

  std::vector<uint32_t> ctrlpkt_instr1_v =
      matmul_common::load_instr_sequence("ctrlpkt_dma_seq.txt");

  std::vector<uint32_t> ctrlPackets1 =
      matmul_common::load_instr_sequence("ctrlpkt.txt");

  std::vector<uint32_t> instr2_v =
      matmul_common::load_instr_sequence("aie2_run_seq.txt");

  std::vector<uint32_t> ctrlpkt_instr2_v =
      matmul_common::load_instr_sequence("aie2_ctrlpkt_dma_seq.txt");

  std::vector<uint32_t> ctrlPackets2 =
      matmul_common::load_instr_sequence("aie2_ctrlpkt.txt");

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

  auto bo_ctrlpkt_instr1 =
      xrt::bo(device, ctrlpkt_instr1_v.size() * sizeof(int),
              XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_ctrlpkt1 = xrt::bo(device, ctrlPackets1.size() * sizeof(int32_t),
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_instr1 = xrt::bo(device, instr1_v.size() * sizeof(int),
                           XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a1 =
      xrt::bo(device, A_SIZE_1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b1 =
      xrt::bo(device, B_SIZE_1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c1 =
      xrt::bo(device, C_SIZE_1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_ctrlpkt_instr2 =
      xrt::bo(device, ctrlpkt_instr2_v.size() * sizeof(int),
              XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_ctrlpkt2 = xrt::bo(device, ctrlPackets2.size() * sizeof(int32_t),
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_instr2 = xrt::bo(device, instr2_v.size() * sizeof(int),
                           XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a2 =
      xrt::bo(device, A_SIZE_2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b2 =
      xrt::bo(device, B_SIZE_2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c2 =
      xrt::bo(device, C_SIZE_2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  A_DATATYPE *bufA1 = bo_a1.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec1(A_VOLUME_1);
  for (int i = 0; i < A_VOLUME_1; i++) {
    AVec1[i] = matmul_common::random_bfloat16_t();
  }
  memcpy(bufA1, AVec1.data(), (AVec1.size() * sizeof(A_DATATYPE)));
  B_DATATYPE *bufB1 = bo_b1.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec1(B_VOLUME_1);
  for (int i = 0; i < B_VOLUME_1; i++) {
    BVec1[i] = matmul_common::random_bfloat16_t();
  }
  memcpy(bufB1, BVec1.data(), (BVec1.size() * sizeof(B_DATATYPE)));
  C_DATATYPE *bufC1 = bo_c1.map<C_DATATYPE *>();
  std::vector<C_DATATYPE> CVec1(C_VOLUME_1);
  memcpy(bufC1, CVec1.data(), (CVec1.size() * sizeof(C_DATATYPE)));

  void *bufInstr1 = bo_instr1.map<void *>();
  memcpy(bufInstr1, instr1_v.data(), instr1_v.size() * sizeof(int));

  void *bufCtrlpktInstr1 = bo_ctrlpkt_instr1.map<void *>();
  memcpy(bufCtrlpktInstr1, ctrlpkt_instr1_v.data(),
         ctrlpkt_instr1_v.size() * sizeof(int));

  void *bufctrlpkt1 = bo_ctrlpkt1.map<void *>();
  memcpy(bufctrlpkt1, ctrlPackets1.data(), ctrlPackets1.size() * sizeof(int));

  A_DATATYPE *bufA2 = bo_a2.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec2(A_VOLUME_2);
  for (int i = 0; i < A_VOLUME_2; i++) {
    AVec2[i] = matmul_common::random_bfloat16_t();
  }
  memcpy(bufA2, AVec2.data(), (AVec2.size() * sizeof(A_DATATYPE)));
  B_DATATYPE *bufB2 = bo_b2.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec2(B_VOLUME_2);
  for (int i = 0; i < B_VOLUME_2; i++) {
    BVec2[i] = matmul_common::random_bfloat16_t();
  }
  memcpy(bufB2, BVec2.data(), (BVec2.size() * sizeof(B_DATATYPE)));
  C_DATATYPE *bufC2 = bo_c2.map<C_DATATYPE *>();
  std::vector<C_DATATYPE> CVec2(C_VOLUME_2);
  memcpy(bufC2, CVec2.data(), (CVec2.size() * sizeof(C_DATATYPE)));

  void *bufInstr2 = bo_instr2.map<void *>();
  memcpy(bufInstr2, instr2_v.data(), instr2_v.size() * sizeof(int));

  void *bufCtrlpktInstr2 = bo_ctrlpkt_instr2.map<void *>();
  memcpy(bufCtrlpktInstr2, ctrlpkt_instr2_v.data(),
         ctrlpkt_instr2_v.size() * sizeof(int));

  void *bufctrlpkt2 = bo_ctrlpkt2.map<void *>();
  memcpy(bufctrlpkt2, ctrlPackets2.data(), ctrlPackets2.size() * sizeof(int));

  bo_ctrlpkt_instr1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ctrlpkt1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_instr1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  bo_ctrlpkt_instr2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ctrlpkt2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_instr2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = 1;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;

  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel.\n";
    }
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;

    // Creating a runlist to contain two seperate runs
    xrt::runlist runlist = xrt::runlist(context);

    // Run 0: configuration
    auto run0 = xrt::run(kernel);
    run0.set_arg(0, opcode);
    run0.set_arg(1, bo_ctrlpkt_instr1);
    run0.set_arg(2, ctrlpkt_instr1_v.size());
    run0.set_arg(3, bo_ctrlpkt1);
    run0.set_arg(4, 0);
    run0.set_arg(5, 0);
    run0.set_arg(6, 0);
    run0.set_arg(7, 0);
    // Run 1: the design
    auto run1 = xrt::run(kernel);
    run1.set_arg(0, opcode);
    run1.set_arg(1, bo_instr1);
    run1.set_arg(2, instr1_v.size());
    run1.set_arg(3, bo_a1);
    run1.set_arg(4, bo_b1);
    run1.set_arg(5, bo_c1);
    run1.set_arg(6, 0);
    run1.set_arg(7, 0);

    // Run 2: configuration
    auto run2 = xrt::run(kernel);
    run2.set_arg(0, opcode);
    run2.set_arg(1, bo_ctrlpkt_instr2);
    run2.set_arg(2, ctrlpkt_instr2_v.size());
    run2.set_arg(3, bo_ctrlpkt2);
    run2.set_arg(4, 0);
    run2.set_arg(5, 0);
    run2.set_arg(6, 0);
    run2.set_arg(7, 0);
    // Run 3: the design
    auto run3 = xrt::run(kernel);
    run3.set_arg(0, opcode);
    run3.set_arg(1, bo_instr2);
    run3.set_arg(2, instr2_v.size());
    run3.set_arg(3, bo_a2);
    run3.set_arg(4, bo_b2);
    run3.set_arg(5, bo_c2);
    run3.set_arg(6, 0);
    run3.set_arg(7, 0);

    // Executing and waiting on the runlist
    runlist.add(run0);
    runlist.add(run1);
    runlist.add(run2);
    runlist.add(run3);
    runlist.execute();
    runlist.wait();

    auto stop = std::chrono::high_resolution_clock::now();

    bo_c1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_c2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(CVec1.data(), bufC1, (CVec1.size() * sizeof(C_DATATYPE)));
    std::vector<C_DATATYPE> CVecRef1(C_VOLUME_1);
    if (VERIFY) {
      if (verbosity >= 1) {
        std::cout << "Verifying against reference matmul ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();
      matmul_common::matmul(M, N, K1, AVec1, BVec1, CVecRef1);
      errors = matmul_common::verify(M, N, K1, AVec1, BVec1, CVec1);
      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (verbosity >= 1) {
        std::cout << "Verify time: " << vtime << "secs." << std::endl;
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: matmul results not verified." << std::endl;
    }

    memcpy(CVec2.data(), bufC2, (CVec2.size() * sizeof(C_DATATYPE)));
    std::vector<C_DATATYPE> CVecRef2(C_VOLUME_2);
    if (VERIFY) {
      if (verbosity >= 1) {
        std::cout << "Verifying against reference matmul ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();
      matmul_common::matmul(M, N, K2, AVec2, BVec2, CVecRef2);
      errors = matmul_common::verify(M, N, K2, AVec2, BVec2, CVec2);
      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (verbosity >= 1) {
        std::cout << "Verify time: " << vtime << "secs." << std::endl;
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: matmul results not verified." << std::endl;
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  std::cout << std::endl
            << "Avg NPU matmul time: " << npu_time_total / num_iter << "us."
            << std::endl;

  std::cout << std::endl
            << "Min NPU matmul time: " << npu_time_min << "us." << std::endl;

  std::cout << std::endl
            << "Max NPU matmul time: " << npu_time_max << "us." << std::endl;

  if (VERIFY && !errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }
}

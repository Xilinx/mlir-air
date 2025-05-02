//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "experimental/xrt_kernel.h"
#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#define M 128
#define N 128
#define K 256

#define A_VOLUME M *K
#define B_VOLUME N *K
#define C_VOLUME M *N

#define A_DATATYPE int32_t
#define B_DATATYPE int32_t
#define C_DATATYPE int32_t

constexpr int A_SIZE = (A_VOLUME * sizeof(A_DATATYPE));
constexpr int B_SIZE = (B_VOLUME * sizeof(B_DATATYPE));
constexpr int C_SIZE = (C_VOLUME * sizeof(C_DATATYPE));
constexpr int TRACE_SIZE = (0 * sizeof(uint32_t));

template <typename T>
void mm_out(std::vector<T> a, std::vector<T> b, std::vector<T> &r) {
  for (size_t m1 = 0; m1 < M; m1++) {
    for (size_t n1 = 0; n1 < N; n1++) {
      size_t idx = m1 * N + n1;
      r[idx] = (T)(0);
      for (size_t k1 = 0; k1 < K; k1++) {
        T _a = a[k1 + m1 * K];
        T _b = b[n1 + k1 * N];
        r[idx] += _a * _b;
      }
    }
  }
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
      "ctrlpktInstr,c",
      "path of file containing control packet instructions to be sent to the "
      "LX6",
      cxxopts::value<std::string>())("ctrlpkts,p",
                                     "path of control packet raw data",
                                     cxxopts::value<std::string>())(
      "trace_sz,t", "size of trace buffer (in bytes)",
      cxxopts::value<int>()->default_value("0"))(
      "trace_file", "where to store trace output",
      cxxopts::value<std::string>()->default_value("trace.txt"));
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

  int trace_size = vm["trace_sz"].as<int>();

  test_utils::check_arg_file_exists(vm, "ctrlpktInstr");
  test_utils::check_arg_file_exists(vm, "ctrlpkts");

  int verbosity = vm["verbosity"].as<int>();

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  std::vector<uint32_t> ctrlpkt_instr_v =
      test_utils::load_instr_binary(vm["ctrlpktInstr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Control packet sequence instr count: "
              << ctrlpkt_instr_v.size() << "\n";

  std::vector<uint32_t> ctrlPackets =
      test_utils::load_instr_binary(vm["ctrlpkts"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Control packet ui32 raw data count: " << ctrlPackets.size()
              << "\n";

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

  auto bo_ctrlpkt_instr = xrt::bo(device, ctrlpkt_instr_v.size() * sizeof(int),
                                  XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_ctrlpkt = xrt::bo(device, ctrlPackets.size() * sizeof(int32_t),
                            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c = xrt::bo(device, C_SIZE + trace_size, XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";
  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec;
  for (int i = 0; i < A_VOLUME; i++)
    AVec.push_back(rand() % UINT16_MAX);
  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(A_DATATYPE)));
  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec;
  for (int i = 0; i < B_VOLUME; i++)
    BVec.push_back(rand() % UINT16_MAX);
  memcpy(bufB, BVec.data(), (BVec.size() * sizeof(B_DATATYPE)));
  C_DATATYPE *bufC = bo_c.map<C_DATATYPE *>();
  memset(bufC, 0, C_SIZE + trace_size);

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  void *bufCtrlpktInstr = bo_ctrlpkt_instr.map<void *>();
  memcpy(bufCtrlpktInstr, ctrlpkt_instr_v.data(),
         ctrlpkt_instr_v.size() * sizeof(int));

  void *bufctrlpkt = bo_ctrlpkt.map<void *>();
  memcpy(bufctrlpkt, ctrlPackets.data(), ctrlPackets.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ctrlpkt_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ctrlpkt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;

  // Creating a runlist to contain two seperate runs
  xrt::runlist runlist = xrt::runlist(context);

  // Run 0: configuration
  auto run0 = xrt::run(kernel);
  run0.set_arg(0, opcode);
  run0.set_arg(1, bo_ctrlpkt_instr);
  run0.set_arg(2, ctrlpkt_instr_v.size());
  run0.set_arg(3, bo_ctrlpkt);
  run0.set_arg(4, 0);
  run0.set_arg(5, 0);
  run0.set_arg(6, 0);
  run0.set_arg(7, 0);
  // Run 1: the design
  auto run1 = xrt::run(kernel);
  run1.set_arg(0, opcode);
  run1.set_arg(1, bo_instr);
  run1.set_arg(2, instr_v.size());
  run1.set_arg(3, bo_a);
  run1.set_arg(4, bo_b);
  run1.set_arg(5, bo_c);
  run1.set_arg(6, 0);
  run1.set_arg(7, 0);

  // Executing and waiting on the runlist
  runlist.add(run0);
  runlist.add(run1);
  runlist.execute();
  runlist.wait();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  C_DATATYPE *bufOut = bo_c.map<C_DATATYPE *>();

  int errors = 0;
  int max_errors = 100;

  std::vector<C_DATATYPE> output_ref0;
  for (uint32_t i = 0; i < C_VOLUME; i++)
    output_ref0.push_back(0);
  mm_out(AVec, BVec, output_ref0);

  for (uint32_t i = 0; i < C_VOLUME; i++) {
    if (bufOut[i] != output_ref0[i]) {
      errors++;
      if (errors < max_errors) {
        std::cout << "\nerror, id " << i << " expected "
                  << std::to_string(output_ref0[i]) << ", got"
                  << std::to_string(bufOut[i]) << "\n";
      }
    }
  }

  if (trace_size > 0) {
    test_utils::write_out_trace(((char *)bufC) + C_SIZE, trace_size,
                                vm["trace_file"].as<std::string>());
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nerror count: " << errors << "\n\n";
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}

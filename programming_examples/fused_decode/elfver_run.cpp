// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Replay one decode step through the FULL-ELF build, taking the context length
// from the scratchpad instead of from a patched instruction stream.
//
// elfver_dump.py writes the BO contents and the xclbin's logits; this replays
// them and dumps the ELF's logits for comparison.
//
// elfver_run.py is the Python equivalent and is the one to prefer. This C++
// version is for hosts whose XRT predates 2026-05-19 (XRT c14df528, xdna-driver
// 1.7): writing the scratchpad goes through xrt::run::get_ctrl_scratchpad_bo(),
// which was in the C++ API well before pyxrt exposed it.
//
//   elfver_run.exe --elf decode.elf --params params.txt --dir /tmp/elfver [--l
//   L]
//
// The two parameters are the whole affine value the BD/core wants, matching the
// names AIR emits (see AIRRtToNpuPass.cpp): the append offset is (L-1)*REGION_W
// elements, the attention mask threshold is L itself.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include "parameter_scratchpad.h"

namespace {

std::map<std::string, long> readMeta(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  std::map<std::string, long> m;
  std::string k;
  long v;
  while (f >> k >> v)
    m[k] = v;
  return m;
}

// Fill a BO from a raw file, checking the size matches what the kernel expects
// -- a short file would otherwise leave the tail as whatever the BO came with.
void fillFromFile(xrt::bo &bo, const std::string &path, size_t bytes) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  size_t have = static_cast<size_t>(f.tellg());
  if (have != bytes)
    throw std::runtime_error(path + ": have " + std::to_string(have) +
                             " bytes, kernel wants " + std::to_string(bytes));
  f.seekg(0);
  f.read(bo.map<char *>(), static_cast<std::streamsize>(bytes));
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

std::string argOf(int argc, char **argv, const std::string &flag,
                  const std::string &dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (flag == argv[i])
      return argv[i + 1];
  return dflt;
}

} // namespace

int main(int argc, char **argv) {
  std::string dir = argOf(argc, argv, "--dir", "/tmp/elfver");
  std::string elfPath = argOf(argc, argv, "--elf", "decode.elf");
  std::string paramsPath = argOf(argc, argv, "--params", "params.txt");

  auto meta = readMeta(dir + "/meta.txt");
  long L = std::stol(argOf(argc, argv, "--l", std::to_string(meta["L"])));
  long regionW = meta["region_w"];

  xrt::device device(0);
  xrt::elf elf(elfPath);
  xrt::hw_context ctx(device, elf);
  xrt::ext::kernel kernel(ctx, "main:q4nx_decode");

  const size_t k = meta["k"], wElems = meta["w_elems"],
               rmsSize = meta["rms_size"];
  const size_t ny = meta["ny"], kvElems = meta["kv_elems"];

  xrt::ext::bo boX(device, k * 2), boW(device, wElems * 2);
  xrt::ext::bo boR(device, rmsSize * 2), boY(device, ny * 2);
  xrt::ext::bo boKv(device, kvElems * 2);

  fillFromFile(boX, dir + "/x.bin", k * 2);
  fillFromFile(boW, dir + "/W.bin", wElems * 2);
  fillFromFile(boR, dir + "/rms.bin", rmsSize * 2);
  fillFromFile(boKv, dir + "/kv.bin", kvElems * 2);
  std::memset(boY.map<char *>(), 0, ny * 2);
  boY.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  xrt::run run(kernel);
  // Bind through xrt::bo&: set_arg's buffer overloads take the base class, and
  // an xrt::ext::bo& is an exact match for the templated SCALAR overload, which
  // would patch the 16-byte handle as a value ("patch_value() only supports
  // 64-bit values or less").
  auto asBo = [](xrt::ext::bo &b) -> xrt::bo & { return b; };
  run.set_arg(0, asBo(boX));
  run.set_arg(1, asBo(boW));
  run.set_arg(2, asBo(boR));
  run.set_arg(3, asBo(boY));
  run.set_arg(4, asBo(boKv));
  // The sequence still takes L as a scalar operand -- it is what the scratchpad
  // parameters are derived from -- and XRT patches every declared argument.
  run.set_arg(5, static_cast<uint32_t>(L));

  test_utils::ParameterScratchpad params(run, paramsPath);
  params.write("__air_param_argoff_5_x256_m256",
               static_cast<uint32_t>((L - 1) * regionW));
  params.write("__air_param_attn_blk_0", static_cast<uint32_t>(L));
  params.sync();

  // Per-token cost: the scratchpad write + sync is what replaces the xclbin
  // path's instruction-stream patch, so time them together with the dispatch.
  long iters = std::stol(argOf(argc, argv, "--iters", "0"));
  if (iters > 0) {
    for (int w = 0; w < 4; ++w) {
      run.start();
      run.wait2();
    }
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < iters; ++i) {
      params.write("__air_param_argoff_5_x256_m256",
                   static_cast<uint32_t>((L - 1) * regionW));
      params.write("__air_param_attn_blk_0", static_cast<uint32_t>(L));
      params.sync();
      run.start();
      run.wait2();
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::printf("[elf] L=%ld per-token %.3f ms (params+sync+dispatch)\n", L,
                ms);

    auto t2 = std::chrono::steady_clock::now();
    for (long i = 0; i < iters; ++i) {
      run.start();
      run.wait2();
    }
    auto t3 = std::chrono::steady_clock::now();
    double msd =
        std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;
    std::printf("[elf] L=%ld dispatch only %.3f ms -> params cost %.3f ms\n", L,
                msd, ms - msd);
  }

  run.start();
  run.wait2(); // throws on unsuccessful completion
  if (run.state() != ERT_CMD_STATE_COMPLETED) {
    std::fprintf(stderr, "dispatch failed: state=%d\n",
                 static_cast<int>(run.state()));
    return 1;
  }

  boY.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::ofstream out(dir + "/elf_y.bin", std::ios::binary);
  out.write(boY.map<char *>(), static_cast<std::streamsize>(ny * 2));
  std::printf("[elf] L=%ld dispatched, wrote %s/elf_y.bin\n", L, dir.c_str());
  return 0;
}

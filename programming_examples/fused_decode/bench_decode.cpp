// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// C++ decode-latency benchmark for the fused decode, scope-matched to the
// FastFlowLM reference so the two numbers can be compared directly.
//
// WHY THIS EXISTS. The Python driver's tok/s and FLM's published tok/s do not
// measure the same thing:
//
//   FLM   profiler_list[DECODING_TIME] wraps ONLY lm_engine->forward();
//         sampling and detokenization are separate accumulators and are NOT in
//         its tok/s (src/common/AutoModel/modeling_llama3.cpp:214-216).
//   AIR   the driver's tok/s wraps dispatch() AND the 128k-element argmax AND
//         the Python loop.
//
// Measured, that scope mismatch is ~0.14 ms/token and the Python host inside
// dispatch() is a further ~0.10 ms. Both are small, but they are the entire
// C++-vs-Python asymmetry in the comparison, so this harness removes them: it
// runs the same per-token sequence dispatch() runs, in C++, timed with
// std::chrono, ending where FLM's forward() ends (logits synced back, no
// sampling, no bf16->f32 conversion).
//
// WEIGHTS ARE SYNTHETIC. Decode time here is data-independent -- every loop
// trip count is fixed at compile time except the attention block loop, which is
// bounded by the patched RTP-L word and not by data -- so this fills the weight
// / KV / rms buffers with a fixed pattern instead of loading the real 773 MB
// requant. That makes it a LATENCY harness only: it does not and cannot check
// numerics. Correctness stays with the Python driver's greedy id-sequence gate.
//
// The per-L instruction stream is derived exactly as DecodeInstsGen does it:
// two same-ATTN_MAXL builds one L apart give a per-word slope, and
// insts(L) = insts(L_base) + (L - L_base) * slope.
//
// Build:  make bench-decode-exe      Run:  make bench-decode [L=1933]
// [ITERS=64]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {

std::vector<uint32_t> readWords(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  auto n = f.tellg();
  if (n <= 0 || n % 4)
    throw std::runtime_error("bad insts size: " + path);
  f.seekg(0);
  std::vector<uint32_t> v(static_cast<size_t>(n) / 4);
  f.read(reinterpret_cast<char *>(v.data()), n);
  return v;
}

long argLong(int argc, char **argv, const std::string &flag, long dflt) {
  for (int i = 1; i + 1 < argc; i++)
    if (flag == argv[i])
      return std::stol(argv[i + 1]);
  return dflt;
}

std::string argStr(int argc, char **argv, const std::string &flag,
                   const std::string &dflt) {
  for (int i = 1; i + 1 < argc; i++)
    if (flag == argv[i])
      return argv[i + 1];
  return dflt;
}

} // namespace

int main(int argc, char **argv) {
  // ---- geometry (llama-3.2-1B defaults; see bench_params in the Makefile)
  // ----
  const size_t K = argLong(argc, argv, "--k", 2048);
  const size_t W_ELEMS = argLong(argc, argv, "--w-elems", 386662400);
  const size_t RMS_SIZE = argLong(argc, argv, "--rms-size", 67648);
  const size_t NY = argLong(argc, argv, "--ny", 134144);
  const size_t KV_ELEMS = argLong(argc, argv, "--kv-elems", 33554432);
  const size_t DECODE_Y = argLong(argc, argv, "--decode-y", 5120);
  const size_t VOC_N = argLong(argc, argv, "--voc-n", 7 * 18432);
  const size_t RMS_LUT_OFF = argLong(argc, argv, "--rms-lut-off", 65536);

  const long L = argLong(argc, argv, "--l", 1933);
  const long iters = argLong(argc, argv, "--iters", 64);
  const long warmup = argLong(argc, argv, "--warmup", 8);
  const std::string dir = argStr(argc, argv, "--dir", ".");
  const long baseL = argLong(argc, argv, "--base-l", 2048);
  const long refL = argLong(argc, argv, "--ref-l", 2047);

  const std::string xclbinPath =
      argStr(argc, argv, "--xclbin",
             dir + "/decode_L" + std::to_string(baseL) + ".xclbin");

  // ---- per-L instruction stream (the DecodeInstsGen slope, in C++) ----
  auto ibase =
      readWords(dir + "/decode_L" + std::to_string(baseL) + ".insts.bin");
  auto iref =
      readWords(dir + "/decode_L" + std::to_string(refL) + ".insts.bin");
  if (ibase.size() != iref.size())
    throw std::runtime_error("insts size mismatch between the two templates; "
                             "they must be same-ATTN_MAXL builds");
  const long dL = baseL - refL;
  std::vector<int64_t> slope(ibase.size());
  size_t lo = ibase.size(), hi = 0;
  for (size_t i = 0; i < ibase.size(); i++) {
    slope[i] =
        (static_cast<int64_t>(ibase[i]) - static_cast<int64_t>(iref[i])) / dL;
    if (slope[i]) {
      lo = std::min(lo, i);
      hi = std::max(hi, i + 1);
    }
  }
  if (lo > hi)
    throw std::runtime_error(
        "no L-dependent words found; templates identical?");
  std::vector<uint32_t> insts(ibase.size());
  for (size_t i = 0; i < insts.size(); i++)
    insts[i] = static_cast<uint32_t>(static_cast<int64_t>(ibase[i]) +
                                     (L - baseL) * slope[i]);

  std::cout << "xclbin      " << xclbinPath << "\n"
            << "insts       " << insts.size() << " words, L-dependent [" << lo
            << ", " << hi << ")\n"
            << "L           " << L << "   iters " << iters << " (warmup "
            << warmup << ")\n";

  // ---- device / kernel ----
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(xclbinPath);
  device.register_xclbin(xclbin);
  std::string kernelName;
  for (auto &k : xclbin.get_kernels())
    if (k.get_name().rfind("MLIR_AIE", 0) == 0)
      kernelName = k.get_name();
  if (kernelName.empty())
    throw std::runtime_error("no MLIR_AIE kernel in the xclbin");
  xrt::hw_context ctx(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(ctx, kernelName);

  const auto HO = XRT_BO_FLAGS_HOST_ONLY;
  auto bo_x = xrt::bo(device, K * 2, HO, kernel.group_id(3));
  auto bo_w = xrt::bo(device, W_ELEMS * 2, HO, kernel.group_id(4));
  auto bo_r = xrt::bo(device, RMS_SIZE * 2, HO, kernel.group_id(5));
  auto bo_y = xrt::bo(device, NY * 2, HO, kernel.group_id(6));
  auto bo_kv = xrt::bo(device, KV_ELEMS * 2, HO, kernel.group_id(7));
  auto bo_i = xrt::bo(device, insts.size() * 4, XCL_BO_FLAGS_CACHEABLE,
                      kernel.group_id(1));

  // Synthetic contents (see the header note: latency here is data-independent).
  // 0x3C00-ish bf16 lanes rather than zeros, so nothing sits on a denormal
  // path.
  auto fill = [](xrt::bo &bo, size_t bytes, uint16_t pat) {
    auto *p = bo.map<uint16_t *>();
    for (size_t i = 0; i < bytes / 2; i++)
      p[i] = pat;
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  };
  fill(bo_w, W_ELEMS * 2, 0x3C00);
  fill(bo_r, RMS_SIZE * 2, 0x3C00);
  fill(bo_kv, KV_ELEMS * 2, 0x3C00);
  fill(bo_x, K * 2, 0x3C00);
  std::memcpy(bo_i.map<uint32_t *>(), insts.data(), insts.size() * 4);
  bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<uint16_t> lut(64, 0x3C00);

  // ---- timed loop: the same per-token sequence the Python dispatch() runs,
  // ending where FLM's forward() ends (logits back, no sampling) ----
  std::vector<double> ms;
  ms.reserve(iters);
  for (long it = 0; it < warmup + iters; it++) {
    auto t0 = std::chrono::steady_clock::now();

    // per-token instruction patch (only the L-dependent slice is re-synced)
    std::memcpy(bo_i.map<uint32_t *>() + lo, insts.data() + lo, (hi - lo) * 4);
    bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE, (hi - lo) * 4, lo * 4);
    // per-position rope LUT + the token embedding
    std::memcpy(bo_r.map<uint16_t *>() + RMS_LUT_OFF, lut.data(), 64 * 2);
    bo_r.sync(XCL_BO_SYNC_BO_TO_DEVICE, 64 * 2, RMS_LUT_OFF * 2);
    bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(3u, bo_i, static_cast<uint32_t>(insts.size()), bo_x, bo_w,
                      bo_r, bo_y, bo_kv);
    auto st = run.wait(60000);
    if (st != ERT_CMD_STATE_COMPLETED) {
      std::cerr << "dispatch did not complete: state=" << st << "\n";
      return 1;
    }
    bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE, VOC_N * 2, DECODE_Y * 2);

    auto t1 = std::chrono::steady_clock::now();
    if (it >= warmup)
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  double mean = 0;
  for (double v : ms)
    mean += v;
  mean /= ms.size();
  double sd = 0;
  for (double v : ms)
    sd += (v - mean) * (v - mean);
  sd = std::sqrt(sd / ms.size());
  double lo_ms = ms[0], hi_ms = ms[0];
  for (double v : ms) {
    lo_ms = std::min(lo_ms, v);
    hi_ms = std::max(hi_ms, v);
  }
  std::printf("\n[bench] n=%zu  mean %.3f ms  sd %.3f  min %.3f  max %.3f  "
              "(%.2f tok/s)\n",
              ms.size(), mean, sd, lo_ms, hi_ms, 1000.0 / mean);
  return 0;
}

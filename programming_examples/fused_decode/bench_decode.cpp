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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

std::vector<size_t> parseList(const std::string &csv) {
  std::vector<size_t> v;
  for (size_t pos = 0; pos < csv.size();) {
    size_t comma = std::min(csv.find(',', pos), csv.size());
    v.push_back(std::stoull(csv.substr(pos, comma - pos)));
    pos = comma + 1;
  }
  return v;
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

  // --w-parts: per-buffer weight element counts for a DECODE_WGROUP build (a
  // shim BD's byte offset is a uint32, so one buffer only reaches 4 GiB and
  // qwen3-8b's 4.41 GiB of weights must be split). Empty => the single-buffer
  // models, which is every other one. The sum is checked against --w-elems
  // because a wrong split still dispatches -- it just feeds the later layers
  // from the wrong base and reports a plausible number.
  std::vector<size_t> wParts = parseList(argStr(argc, argv, "--w-parts", ""));
  if (wParts.empty())
    wParts.push_back(W_ELEMS);
  size_t wTotal = 0;
  for (size_t n : wParts)
    wTotal += n;
  if (wTotal != W_ELEMS)
    throw std::runtime_error("--w-parts sums to " + std::to_string(wTotal) +
                             ", not --w-elems " + std::to_string(W_ELEMS));

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
            << warmup << ")\n"
            << "weights     " << W_ELEMS << " elems in " << wParts.size()
            << " buffer(s)\n";

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
  // Weight group 0 keeps the original weight arg; the remaining groups and the
  // lm-head slab are appended after kvc, so every pre-existing binding position
  // is unchanged (fused_decode.py `_w_extra` / WARG).
  std::vector<xrt::bo> bo_w;
  for (size_t i = 0; i < wParts.size(); i++)
    bo_w.emplace_back(device, wParts[i] * 2, HO,
                      kernel.group_id(i == 0 ? 4 : 7 + static_cast<int>(i)));
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
  for (size_t i = 0; i < bo_w.size(); i++)
    fill(bo_w[i], wParts[i] * 2, 0x3C00);
  fill(bo_r, RMS_SIZE * 2, 0x3C00);
  fill(bo_kv, KV_ELEMS * 2, 0x3C00);
  fill(bo_x, K * 2, 0x3C00);
  std::memcpy(bo_i.map<uint32_t *>(), insts.data(), insts.size() * 4);
  bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<uint16_t> lut(64, 0x3C00);

  // ---- timed loop: the same per-token sequence the Python dispatch() runs,
  // ending where FLM's forward() ends (logits back, no sampling) ----
  // Phase split. The four host phases bracket the one device phase, so
  // `dispatch` isolates the on-NPU time from the fixed per-dispatch host cost
  // that every measurement (including any UNI_WAVE_LO/HI part-build) pays once.
  // Without that split, timing two wave-subsets and summing double-counts the
  // host cost and inflates the apparent cost of whichever part you attribute it
  // to. The four extra steady_clock reads are ~100 ns total, six orders of
  // magnitude below the signal.
  std::vector<double> ms, msInsts, msFeed, msDisp, msBack;
  ms.reserve(iters);
  for (long it = 0; it < warmup + iters; it++) {
    auto t0 = std::chrono::steady_clock::now();

    // per-token instruction patch (only the L-dependent slice is re-synced)
    std::memcpy(bo_i.map<uint32_t *>() + lo, insts.data() + lo, (hi - lo) * 4);
    bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE, (hi - lo) * 4, lo * 4);
    auto t1 = std::chrono::steady_clock::now();

    // per-position rope LUT + the token embedding
    std::memcpy(bo_r.map<uint16_t *>() + RMS_LUT_OFF, lut.data(), 64 * 2);
    bo_r.sync(XCL_BO_SYNC_BO_TO_DEVICE, 64 * 2, RMS_LUT_OFF * 2);
    bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t2 = std::chrono::steady_clock::now();

    // set_arg rather than kernel(...): the argument count is not fixed, because
    // a split-weight build appends one buffer per extra group. Same sequence
    // xrt::kernel::operator() runs (construct, set, start), so the per-dispatch
    // host cost inside the `dispatch` phase is unchanged.
    xrt::run run(kernel);
    int a = 0;
    run.set_arg(a++, 3u);
    run.set_arg(a++, bo_i);
    run.set_arg(a++, static_cast<uint32_t>(insts.size()));
    run.set_arg(a++, bo_x);
    run.set_arg(a++, bo_w[0]);
    run.set_arg(a++, bo_r);
    run.set_arg(a++, bo_y);
    run.set_arg(a++, bo_kv);
    for (size_t i = 1; i < bo_w.size(); i++)
      run.set_arg(a++, bo_w[i]);
    run.start();
    auto st = run.wait(60000);
    if (st != ERT_CMD_STATE_COMPLETED) {
      std::cerr << "dispatch did not complete: state=" << st << "\n";
      return 1;
    }
    auto t3 = std::chrono::steady_clock::now();

    bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE, VOC_N * 2, DECODE_Y * 2);
    auto t4 = std::chrono::steady_clock::now();

    if (it >= warmup) {
      using d = std::chrono::duration<double, std::milli>;
      ms.push_back(d(t4 - t0).count());
      msInsts.push_back(d(t1 - t0).count());
      msFeed.push_back(d(t2 - t1).count());
      msDisp.push_back(d(t3 - t2).count());
      msBack.push_back(d(t4 - t3).count());
    }
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

  auto avg = [](const std::vector<double> &v) {
    double s = 0;
    for (double x : v)
      s += x;
    return s / v.size();
  };
  const double aI = avg(msInsts), aF = avg(msFeed), aD = avg(msDisp),
               aB = avg(msBack);
  std::printf("[phase] insts %.3f  feed %.3f  dispatch %.3f  logits-back %.3f "
              " (host total %.3f)\n",
              aI, aF, aD, aB, aI + aF + aB);
  return 0;
}

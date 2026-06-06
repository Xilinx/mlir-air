// SPDX-License-Identifier: MIT
// ELF profile harness for o_gemv_ffn_int4_fused. 8 args:
//   PACKED_la, B_la, R_la, PACKED_lgu, gamma, PACKED_ld, D_ld, D_dbg

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <stdfloat>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>

using DT_BF16 = std::bfloat16_t;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("o_gemv_ffn_int4_fused ELF profile");
  options.add_options()("help,h", "")("elf,e", "",
                                      cxxopts::value<std::string>())(
      "kernel,k", "", cxxopts::value<std::string>())(
      "size_m_la,L", "M_LA", cxxopts::value<int>()->default_value("2048"))(
      "size_m_lgu,U", "M_LGU", cxxopts::value<int>()->default_value("16384"))(
      "size_k_ld,D", "K_LD", cxxopts::value<int>()->default_value("8192"))(
      "size_k,K", "K", cxxopts::value<int>()->default_value("2048"))(
      "group_size,G", "GS", cxxopts::value<int>()->default_value("128"))(
      "tile_m,T", "M_TILE", cxxopts::value<int>()->default_value("8"))(
      "k_chunk,C", "K_CHUNK", cxxopts::value<int>()->default_value("2048"))(
      "iterations", "", cxxopts::value<int>()->default_value("200"))(
      "warmup", "", cxxopts::value<int>()->default_value("50"));

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int M_LA = vm["size_m_la"].as<int>();
  int M_LGU = vm["size_m_lgu"].as<int>();
  int K_LD = vm["size_k_ld"].as<int>();
  int K = vm["size_k"].as<int>(), GS = vm["group_size"].as<int>();
  int M_TILE = vm["tile_m"].as<int>(), K_CHUNK = vm["k_chunk"].as<int>();
  int n_gpc = K_CHUNK / GS;

  size_t Q_B = (size_t)M_TILE * (K_CHUNK / 2);
  size_t S_B = (size_t)n_gpc * M_TILE * sizeof(DT_BF16);
  size_t Z_B = (size_t)n_gpc * M_TILE;
  size_t TILE_B = Q_B + S_B + Z_B;
  size_t n_tiles_la = (size_t)M_LA / M_TILE;
  size_t PACKED_LA = n_tiles_la * TILE_B;
  size_t n_tiles_lgu = (size_t)M_LGU / M_TILE;
  size_t PACKED_LGU = n_tiles_lgu * TILE_B;
  size_t n_tiles_ld = (size_t)M_LA / M_TILE * (K_LD / K_CHUNK);
  size_t PACKED_LD = n_tiles_ld * TILE_B;
  size_t B_SIZE = (size_t)K * sizeof(DT_BF16);
  size_t R_SIZE = (size_t)M_LA * sizeof(DT_BF16);
  size_t GAMMA_SIZE = (size_t)K * sizeof(DT_BF16);
  size_t D_SIZE = (size_t)M_LA * sizeof(DT_BF16);

  srand(time(NULL));
  auto device = xrt::device(0);
  xrt::elf ctx_elf{vm["elf"].as<std::string>()};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, vm["kernel"].as<std::string>());

  xrt::bo bo_p_la = xrt::ext::bo{device, PACKED_LA};
  xrt::bo bo_b = xrt::ext::bo{device, B_SIZE};
  xrt::bo bo_r = xrt::ext::bo{device, R_SIZE};
  xrt::bo bo_p_lgu = xrt::ext::bo{device, PACKED_LGU};
  xrt::bo bo_g = xrt::ext::bo{device, GAMMA_SIZE};
  xrt::bo bo_p_ld = xrt::ext::bo{device, PACKED_LD};
  xrt::bo bo_d = xrt::ext::bo{device, D_SIZE};
  xrt::bo bo_dbg = xrt::ext::bo{device, D_SIZE};

  auto fill_packed = [&](xrt::bo &bo, size_t n_tiles) {
    uint8_t *p = bo.map<uint8_t *>();
    for (size_t t = 0; t < n_tiles; t++) {
      uint8_t *tp = p + t * TILE_B;
      for (size_t i = 0; i < Q_B; i++)
        tp[i] = (uint8_t)(rand() & 0xFF);
      DT_BF16 *s = reinterpret_cast<DT_BF16 *>(tp + Q_B);
      for (size_t i = 0; i < S_B / sizeof(DT_BF16); i++)
        s[i] = DT_BF16(0.005f + 0.015f * ((float)rand() / RAND_MAX));
      uint8_t *z = tp + Q_B + S_B;
      for (size_t i = 0; i < Z_B; i++)
        z[i] = (uint8_t)(7 + (rand() & 1));
    }
  };
  fill_packed(bo_p_la, n_tiles_la);
  fill_packed(bo_p_lgu, n_tiles_lgu);
  fill_packed(bo_p_ld, n_tiles_ld);
  DT_BF16 *bp = bo_b.map<DT_BF16 *>();
  for (int i = 0; i < K; i++)
    bp[i] = DT_BF16(2.0f * (float)rand() / RAND_MAX - 1.0f);
  DT_BF16 *rp = bo_r.map<DT_BF16 *>();
  for (int i = 0; i < M_LA; i++)
    rp[i] = DT_BF16(2.0f * (float)rand() / RAND_MAX - 1.0f);
  DT_BF16 *gp = bo_g.map<DT_BF16 *>();
  for (int i = 0; i < K; i++)
    gp[i] = DT_BF16(0.2f * (float)rand() / RAND_MAX + 0.9f);
  memset(bo_d.map<DT_BF16 *>(), 0, D_SIZE);
  memset(bo_dbg.map<DT_BF16 *>(), 0, D_SIZE);

  bo_p_la.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_r.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_p_lgu.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_g.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_p_ld.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_d.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_dbg.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_iter = vm["iterations"].as<int>(), n_warm = vm["warmup"].as<int>();
  unsigned total = n_iter + n_warm;
  float t_total = 0, t_min = std::numeric_limits<float>::max(), t_max = 0;
  // LA: 2 * M_LA * K. LGU: 2 * M_LGU * K. LD: 2 * M_LA * K_LD.
  float macs = 2.0f * ((float)M_LA * K + (float)M_LGU * K + (float)M_LA * K_LD);

  std::cout << "o_gemv_ffn_int4_fused Benchmark (ELF)\n  M_LA=" << M_LA
            << " M_LGU=" << M_LGU << " K=" << K << " K_LD=" << K_LD
            << " PACKED_la=" << PACKED_LA << " PACKED_lgu=" << PACKED_LGU
            << " PACKED_ld=" << PACKED_LD << "\n";
  for (unsigned i = 0; i < total; i++) {
    auto run = xrt::run(kernel);
    run.set_arg(0, bo_p_la);
    run.set_arg(1, bo_b);
    run.set_arg(2, bo_r);
    run.set_arg(3, bo_p_lgu);
    run.set_arg(4, bo_g);
    run.set_arg(5, bo_p_ld);
    run.set_arg(6, bo_d);
    run.set_arg(7, bo_dbg);
    auto t0 = std::chrono::high_resolution_clock::now();
    run.start();
    run.wait2();
    auto t1 = std::chrono::high_resolution_clock::now();
    bo_d.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (i < n_warm)
      continue;
    float dt =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    t_total += dt;
    t_min = std::min(t_min, dt);
    t_max = std::max(t_max, dt);
  }
  std::cout << "\nAvg NPU time: " << t_total / n_iter << " us\n";
  std::cout << "Avg gflops:   " << macs / (1000.0f * t_total / n_iter)
            << "  (LA + LGU + LD)\n";
  std::cout << "Min NPU time: " << t_min << " us  Max: " << t_max << " us\n";
  return 0;
}

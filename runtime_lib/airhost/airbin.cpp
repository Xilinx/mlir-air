//===- airbin.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "airbin.h"
#include "air_host.h"
#include "air_queue.h"
#include <algorithm> // minmax_element, sort
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility> // pair
#include <vector>

// Note that the file header we store ignores the magic bytes.
struct Air64_Fhdr {
  uint16_t f_type;
  uint16_t arch;
  uint16_t f_ver;
  uint16_t num_ch;
  uint32_t chcfg;
};

static Air64_Fhdr read_file_header(std::ifstream &infile) {
  Air64_Fhdr result;
  unsigned char longnum[8] = {0};
  infile.seekg(2 * sizeof(longnum));

  infile.read(reinterpret_cast<char *>(longnum), 2);
  result.f_type = std::stoi(reinterpret_cast<char *>(longnum), NULL, 16);

  infile.read(reinterpret_cast<char *>(longnum), 2);
  result.arch = std::stoi(reinterpret_cast<char *>(longnum), NULL, 16);

  infile.read(reinterpret_cast<char *>(longnum), 2);
  result.f_ver = std::stoi(reinterpret_cast<char *>(longnum), NULL, 16);

  infile.read(reinterpret_cast<char *>(longnum), 2);
  result.num_ch = std::stoi(reinterpret_cast<char *>(longnum), NULL, 16);

  infile.read(reinterpret_cast<char *>(longnum), 8);
  result.chcfg = std::stoi(reinterpret_cast<char *>(longnum), NULL, 16);

  return result;
}

struct Air64_Chdr {
  uint16_t ch_tile;
  uint8_t ch_name;
  uint32_t ch_type;
  uint64_t ch_addr;
  uint64_t ch_offset;
  uint64_t ch_size;
};

static std::string to_air_cfg_name(uint32_t in) {
  switch (in) {
  case 1:
    return ".ssmast";
    break;
  case 2:
    return ".ssslve";
    break;
  case 3:
    return ".sspckt";
    break;
  case 4:
    return ".sdma.bd";
    break;
  case 5:
    return ".shmmux";
    break;
  case 6:
    return ".sdma.ctl";
    break;
  case 7:
    return ".prgm.mem";
    break;
  case 8:
    return ".tdma.bd";
    break;
  case 9:
    return ".tdma.ctl";
    break;
  case 10:
    return ".data.stk";
    break;
  case 11:
    return ".data.mem";
    break;
  default:
    return "";
    break;
  }
  return "";
}

static Air64_Chdr read_section_header(std::ifstream &infile,
                                      uint8_t column_offset) {
  Air64_Chdr config_header;
  uint32_t location_and_name;

  auto start = infile.tellg();
  infile >> std::hex >> location_and_name;
  infile >> std::hex >> config_header.ch_type;
  infile >> std::hex >> config_header.ch_addr;
  infile >> std::hex >> config_header.ch_addr;
  infile >> std::hex >> config_header.ch_offset;
  infile >> std::hex >> config_header.ch_offset;
  infile >> std::hex >> config_header.ch_size;
  infile >> std::hex >> config_header.ch_size;

  // Each byte needs 2 hex digits to represent.
  // Each 4-byte word (except the last) also needs a newline.
  static constexpr auto section_header_size =
      sizeof(config_header) * 2 + sizeof(config_header) / 4 - 1;
  // TODO: Better error messages
  assert(infile.tellg() - start == section_header_size);

  config_header.ch_name = location_and_name & 0xFFu;
  config_header.ch_tile = (location_and_name >> 8u) & 0xFFFFu;
  config_header.ch_tile += static_cast<uint16_t>(column_offset) << 5u;

  assert(config_header.ch_tile != 0);
  assert(config_header.ch_size % 4 == 0);

  return config_header;
}

airbin_size readairbinsize(std::ifstream &infile, uint8_t column_offset) {
  std::vector<std::pair<uint8_t, uint8_t>> tiles;
  uint32_t next_chcfg_idx = 0;
  Air64_Fhdr file_header = read_file_header(infile);
  while (next_chcfg_idx < file_header.num_ch) {
    infile.seekg(file_header.chcfg + 1 + next_chcfg_idx * 9 * 8);
    Air64_Chdr config_header = read_section_header(infile, column_offset);
    if (config_header.ch_type) {
      auto row_num = config_header.ch_tile & 0x1FU;
      auto col_num = (config_header.ch_tile >> 5u) & 0x3FU;

      tiles.emplace_back(col_num, row_num);
    }
    next_chcfg_idx++;
  }

  std::sort(tiles.begin(), tiles.end());

  airbin_size result;
  result.start_col = tiles.front().first;
  result.num_cols = (tiles.back().first - result.start_col) + 1u;

  // Min = first, max = second
  auto minmax_rows =
      std::minmax_element(tiles.begin(), tiles.end(), [](auto lhs, auto rhs) {
        return lhs.second < rhs.second;
      });

  result.start_row = minmax_rows.first->second;
  result.num_rows = (minmax_rows.second->second - result.start_row) + 1u;

  return result;
}

void readairbin(std::ifstream &infile) {
  unsigned char longnum[8] = {0};
  uint16_t f_type;
  uint16_t arch;
  uint16_t f_ver;
  uint16_t num_ch;
  uint32_t chcfg;
  uint32_t next_chcfg_idx = 0;
  infile.seekg(2 * sizeof(longnum));
  infile.read(reinterpret_cast<char *>(longnum), 2);
  f_type = std::stoi(reinterpret_cast<char *>(longnum), NULL, 16);
  infile.read(reinterpret_cast<char *>(longnum), 2);
  arch = std::stoi(reinterpret_cast<char *>(longnum), NULL, 16);
  infile.read(reinterpret_cast<char *>(longnum), 2);
  f_ver = std::stoi(reinterpret_cast<char *>(longnum), NULL, 16);
  infile.read(reinterpret_cast<char *>(longnum), 2);
  num_ch = std::stoi(reinterpret_cast<char *>(longnum), NULL, 16);
  infile.read(reinterpret_cast<char *>(longnum), 8);
  chcfg = std::stoi(reinterpret_cast<char *>(longnum), NULL, 16);
  std::cout << "Configuration Headers:" << std::endl;
  std::cout << std::right << std::setw(6) << std::setfill(' ') << std::dec
            << "[Nr]" << ' ';
  std::cout << std::left << std::hex << std::setw(12) << std::setfill(' ')
            << "Name" << ' ';
  std::cout << std::hex << std::setw(12) << std::setfill(' ') << "Type" << ' ';
  std::cout << std::hex << std::setw(8) << std::setfill(' ') << "Addr" << ' ';
  std::cout << std::hex << std::setw(6) << std::setfill(' ') << "Offset" << ' ';
  std::cout << std::hex << std::setw(6) << std::setfill(' ') << "Size"
            << std::endl;
  while (next_chcfg_idx < num_ch) {
    infile.seekg(chcfg + 1 + next_chcfg_idx * 9 * 8);
    Air64_Chdr config_header;
    infile >> std::hex >> config_header.ch_tile;
    infile >> std::hex >> config_header.ch_name;
    infile >> std::hex >> config_header.ch_type;
    infile >> std::hex >> config_header.ch_addr;
    infile >> std::hex >> config_header.ch_addr;
    infile >> std::hex >> config_header.ch_offset;
    infile >> std::hex >> config_header.ch_offset;
    infile >> std::hex >> config_header.ch_size;
    infile >> std::hex >> config_header.ch_size;
    std::cout << std::right << std::dec << "  [" << std::setw(2)
              << std::setfill(' ') << next_chcfg_idx + 1 << "]" << ' '
              << std::left;
    if (config_header.ch_type) {
      std::cout << std::hex << std::setw(12) << std::setfill(' ')
                << to_air_cfg_name(config_header.ch_name) << ' ';
      std::cout << std::hex << std::setw(12) << std::setfill(' ') << "PROGBITS"
                << ' ';
      std::cout << std::right << std::hex << std::setw(8) << std::setfill('0')
                << config_header.ch_addr << ' ';
      std::cout << std::hex << std::setw(6) << std::setfill('0')
                << config_header.ch_offset << ' ';
      std::cout << std::hex << std::setw(6) << std::setfill('0')
                << config_header.ch_size << std::endl;
    } else {
      std::cout << std::hex << std::setw(12) << std::setfill(' ') << "" << ' ';
      std::cout << std::hex << std::setw(12) << std::setfill(' ') << "NULL"
                << ' ';
      std::cout << std::right << std::hex << std::setw(8) << std::setfill('0')
                << config_header.ch_addr << ' ';
      std::cout << std::hex << std::setw(6) << std::setfill('0')
                << config_header.ch_offset << ' ';
      std::cout << std::hex << std::setw(6) << std::setfill('0')
                << config_header.ch_size << std::endl;
    }
    next_chcfg_idx++;
  }
}

uint64_t airbin2mem(std::ifstream &infile, volatile uint32_t *tds_va,
                    uint32_t *tds_pa, volatile uint32_t *data_va,
                    uint32_t *data_pa, uint8_t col) {
  uint64_t last_td = 0;

  uint32_t next_chcfg_idx = 0;
  Air64_Fhdr file_header = read_file_header(infile);

  int p = 0;
  int bd = 0;
  uint64_t next_ptr = uint64_t(tds_pa) + 64 * (bd + 1);
  uint64_t sa = uint64_t(data_pa);

  while (next_chcfg_idx < file_header.num_ch) {
    infile.seekg(file_header.chcfg + 1 + next_chcfg_idx * 9 * 8);
    Air64_Chdr config_header = read_section_header(infile, col);
    infile.seekg(config_header.ch_offset);
    if (config_header.ch_type) {
      if (config_header.ch_name != 11) {
        std::string line_desc;
        std::getline(infile, line_desc);
      }
      for (auto i = 0ul; i < config_header.ch_size / sizeof(uint32_t); i++) {
        uint32_t val;
        infile >> std::hex >> val;
        data_va[p++] = val;
      }
      if (config_header.ch_name != 11) {
        std::string line_desc;
        std::getline(infile, line_desc);
      }

      static constexpr uint64_t array_offset = static_cast<uint64_t>(0x800u)
                                               << (18u + 12u);
      auto core =
          array_offset | (static_cast<uint64_t>(config_header.ch_tile) << 18u);
      uint64_t da = uint64_t(core + config_header.ch_addr);
      assert((da & core) == core);
      tds_va[16 * bd + 0] = next_ptr & 0xffffffffu; // NXTDESC_PNTR
      tds_va[16 * bd + 1] = (next_ptr >> 32u);      // NXTDESC_PNTR_MSB
      tds_va[16 * bd + 2] = sa & 0xffffffffu;       // SA
      tds_va[16 * bd + 3] = (sa >> 32u);            // SA_MSB
      tds_va[16 * bd + 4] = da & 0xffffffffu;       // DA
      tds_va[16 * bd + 5] = (da >> 32u);            // DA_MSB
      tds_va[16 * bd + 6] = config_header.ch_size;  // CONTROL (BTT)
      tds_va[16 * bd + 7] = 0;                      // STATUS
      bd++;
      if (next_chcfg_idx + 2 ==
          file_header.num_ch) { // FIXME save rather than "last"
        last_td = next_ptr;
        next_ptr = 0;
      } else {
        next_ptr = uint64_t(tds_pa) + 64ul * (bd + 1);
      }
      sa += config_header.ch_size;
    }
    next_chcfg_idx++;
  }

  return last_td;
}

int air_load_airbin(queue_t *q, const char *filename, uint8_t column,
                    uint32_t device_id) {

  // Initializing the device memory allocator
  if (air_init_dev_mem_allocator(0x2000000 /* dev_mem_size */,
                                 device_id /* device_id (optional)*/)) {
    std::cout << "Error creating device memory allocator" << std::endl;
    return -1;
  }

  auto *data_p = air_dev_mem_alloc(2 * 65536);
  volatile uint32_t *bram_ptr = reinterpret_cast<volatile uint32_t *>(data_p);
  auto *paddr = reinterpret_cast<uint32_t *>(air_dev_mem_get_pa(data_p));

  auto *bd_p = air_dev_mem_alloc(0x8000);
  volatile uint32_t *bd_ptr = reinterpret_cast<volatile uint32_t *>(bd_p);
  auto bd_paddr = uint64_t(air_dev_mem_get_pa(bd_p));

  std::ifstream infile{filename};

  auto size = readairbinsize(infile, column);

  // AIRBIN from file to memory
  uint64_t last_td = airbin2mem(infile, bd_ptr, (uint32_t *)bd_paddr, bram_ptr,
                                paddr, size.start_col);

  // Send configuration packet
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt =
      reinterpret_cast<dispatch_packet_t *>(q->base_address_vaddr) + packet_id;
  air_packet_cdma_configure(pkt, last_td, uint64_t(bd_paddr), 0xffffffff,
                            &size);

  // struct timespec ts_start;
  // struct timespec ts_end;
  // clock_gettime(CLOCK_BOOTTIME, &ts_start);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);
  // clock_gettime(CLOCK_BOOTTIME, &ts_end);

  // auto time_spec_diff = [](struct timespec &start, struct timespec &end) {
  //  return (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
  //};

  // printf("config time: %0.8f sec\n", time_spec_diff(ts_start, ts_end));
  return 0;
}

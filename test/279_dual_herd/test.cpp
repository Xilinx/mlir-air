//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include <xaiengine.h>

#include "acdc_queue.h"
#include "air_host.h"
#include "hsa_defs.h"

#include <ctime>
#include <sys/time.h>

float time_diff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

// #define EI_NIDENT       16
//
// typedef struct {
//         unsigned char   e_ident[EI_NIDENT];
//         Elf32_Half      e_type;
//         Elf32_Half      e_machine;
//         Elf32_Word      e_version;
//         Elf32_Addr      e_entry;
//         Elf32_Off       e_phoff;
//         Elf32_Off       e_shoff;
//         Elf32_Word      e_flags;
//         Elf32_Half      e_ehsize;
//         Elf32_Half      e_phentsize;
//         Elf32_Half      e_phnum;
//         Elf32_Half      e_shentsize;
//         Elf32_Half      e_shnum;
//         Elf32_Half      e_shstrndx;
// } Elf32_Ehdr;
//
// typedef struct {
//         Elf32_Word      p_type;
//         Elf32_Off       p_offset;
//         Elf32_Addr      p_vaddr;
//         Elf32_Addr      p_paddr;
//         Elf32_Word      p_filesz;
//         Elf32_Word      p_memsz;
//         Elf32_Word      p_flags;
//         Elf32_Word      p_align;
// } Elf32_Phdr;

#define XAIE_ADDR_ARRAY_OFF 0x800

typedef enum { AIE = 1, AIEAIEML = 2 } aie_arch_t;

typedef enum { AIR_COL = 1, AIR_HRD = 2 } airbin_file_t;

typedef struct {
  uint32_t ch_name;
  uint32_t ch_type;
  uint64_t ch_addr;
  uint64_t ch_offset;
  uint64_t ch_size;
} Air64_Chdr;

uint32_t swapped(unsigned char num[4]) {
  uint32_t val = 0;
  val = ((uint32_t(num[0])) & 0xff) | ((uint32_t(num[2]) << 16) & 0xff0000) |
        ((uint32_t(num[1]) << 8) & 0xff00) |
        ((uint32_t(num[3]) << 24) & 0xff000000);
  return val;
}

uint32_t to32(unsigned char num[4]) {
  uint32_t val = 0;
  val = ((uint32_t(num[3])) & 0xff) | ((uint32_t(num[1]) << 16) & 0xff0000) |
        ((uint32_t(num[2]) << 8) & 0xff00) |
        ((uint32_t(num[0]) << 24) & 0xff000000);
  return val;
}

u64 getTileAddr(u16 ColIdx, u16 RowIdx) {
  u64 TileAddr = 0;
  u64 ArrOffset = XAIE_ADDR_ARRAY_OFF;

#ifdef XAIE_BASE_ARRAY_ADDR_OFFSET
  ArrOffset = XAIE_BASE_ARRAY_ADDR_OFFSET;
#endif

  /*
   * Tile address format:
   * --------------------------------------------
   * |                7 bits  5 bits   18 bits  |
   * --------------------------------------------
   * | Array offset | Column | Row | Tile addr  |
   * --------------------------------------------
   */
  TileAddr = (u64)((ArrOffset << XAIEGBL_TILE_ADDR_ARR_SHIFT) |
                   (ColIdx << XAIEGBL_TILE_ADDR_COL_SHIFT) |
                   (RowIdx << XAIEGBL_TILE_ADDR_ROW_SHIFT));

  return TileAddr;
}

const int ping_in_ofst = 4096;
int32_t mlir_aie_read_buffer_ping_in(aie_libxaie_ctx_t *ctx, int col, int row,
                                     int index) {
  int32_t value = XAieTile_DmReadWord(&(ctx->TileInst[col][row]),
                                      ping_in_ofst + (index * 4));
  return value;
}
void mlir_aie_write_buffer_ping_in(aie_libxaie_ctx_t *ctx, int col, int row,
                                   int index, int32_t value) {
  int32_t int_value = value;
  return XAieTile_DmWriteWord(&(ctx->TileInst[col][row]),
                              ping_in_ofst + (index * 4), int_value);
}
const int ping_out_ofst = 4128;
int32_t mlir_aie_read_buffer_ping_out(aie_libxaie_ctx_t *ctx, int col, int row,
                                      int index) {
  int32_t value = XAieTile_DmReadWord(&(ctx->TileInst[col][row]),
                                      ping_out_ofst + (index * 4));
  return value;
}
void mlir_aie_write_buffer_ping_out(aie_libxaie_ctx_t *ctx, int col, int row,
                                    int index, int32_t value) {
  int32_t int_value = value;
  return XAieTile_DmWriteWord(&(ctx->TileInst[col][row]),
                              ping_out_ofst + (index * 4), int_value);
}
const int pong_in_ofst = 4160;
int32_t mlir_aie_read_buffer_pong_in(aie_libxaie_ctx_t *ctx, int col, int row,
                                     int index) {
  int32_t value = XAieTile_DmReadWord(&(ctx->TileInst[col][row]),
                                      pong_in_ofst + (index * 4));
  return value;
}
void mlir_aie_write_buffer_pong_in(aie_libxaie_ctx_t *ctx, int col, int row,
                                   int index, int32_t value) {
  int32_t int_value = value;
  return XAieTile_DmWriteWord(&(ctx->TileInst[col][row]),
                              pong_in_ofst + (index * 4), int_value);
}
const int pong_out_ofst = 4192;
int32_t mlir_aie_read_buffer_pong_out(aie_libxaie_ctx_t *ctx, int col, int row,
                                      int index) {
  int32_t value = XAieTile_DmReadWord(&(ctx->TileInst[col][row]),
                                      pong_out_ofst + (index * 4));
  return value;
}
void mlir_aie_write_buffer_pong_out(aie_libxaie_ctx_t *ctx, int col, int row,
                                    int index, int32_t value) {
  int32_t int_value = value;
  return XAieTile_DmWriteWord(&(ctx->TileInst[col][row]),
                              pong_out_ofst + (index * 4), int_value);
}
const int ping_a_offset = 4096;
int32_t mlir_aie_read_buffer_ping_a(aie_libxaie_ctx_t *ctx, int col, int row,
                                    int index) {
  int32_t value = XAieTile_DmReadWord(&(ctx->TileInst[col][row]),
                                      ping_a_offset + (index * 4));
  return value;
}
void mlir_aie_write_buffer_ping_a(aie_libxaie_ctx_t *ctx, int col, int row,
                                  int index, int32_t value) {
  int32_t int_value = value;
  return XAieTile_DmWriteWord(&(ctx->TileInst[col][row]),
                              ping_a_offset + (index * 4), int_value);
}
const int ping_b_offset = 8192;
int32_t mlir_aie_read_buffer_ping_b(aie_libxaie_ctx_t *ctx, int col, int row,
                                    int index) {
  int32_t value = XAieTile_DmReadWord(&(ctx->TileInst[col][row]),
                                      ping_b_offset + (index * 4));
  return value;
}
void mlir_aie_write_buffer_ping_b(aie_libxaie_ctx_t *ctx, int col, int row,
                                  int index, int32_t value) {
  int32_t int_value = value;
  return XAieTile_DmWriteWord(&(ctx->TileInst[col][row]),
                              ping_b_offset + (index * 4), int_value);
}
const int ping_c_offset = 12288;
int32_t mlir_aie_read_buffer_ping_c(aie_libxaie_ctx_t *ctx, int col, int row,
                                    int index) {
  int32_t value = XAieTile_DmReadWord(&(ctx->TileInst[col][row]),
                                      ping_c_offset + (index * 4));
  return value;
}
void mlir_aie_write_buffer_ping_c(aie_libxaie_ctx_t *ctx, int col, int row,
                                  int index, int32_t value) {
  int32_t int_value = value;
  return XAieTile_DmWriteWord(&(ctx->TileInst[col][row]),
                              ping_c_offset + (index * 4), int_value);
}
const int pong_a_offset = 16384;
int32_t mlir_aie_read_buffer_pong_a(aie_libxaie_ctx_t *ctx, int col, int row,
                                    int index) {
  int32_t value = XAieTile_DmReadWord(&(ctx->TileInst[col][row]),
                                      pong_a_offset + (index * 4));
  return value;
}
void mlir_aie_write_buffer_pong_a(aie_libxaie_ctx_t *ctx, int col, int row,
                                  int index, int32_t value) {
  int32_t int_value = value;
  return XAieTile_DmWriteWord(&(ctx->TileInst[col][row]),
                              pong_a_offset + (index * 4), int_value);
}
const int pong_b_offset = 20480;
int32_t mlir_aie_read_buffer_pong_b(aie_libxaie_ctx_t *ctx, int col, int row,
                                    int index) {
  int32_t value = XAieTile_DmReadWord(&(ctx->TileInst[col][row]),
                                      pong_b_offset + (index * 4));
  return value;
}
void mlir_aie_write_buffer_pong_b(aie_libxaie_ctx_t *ctx, int col, int row,
                                  int index, int32_t value) {
  int32_t int_value = value;
  return XAieTile_DmWriteWord(&(ctx->TileInst[col][row]),
                              pong_b_offset + (index * 4), int_value);
}
const int pong_c_offset = 24576;
int32_t mlir_aie_read_buffer_pong_c(aie_libxaie_ctx_t *ctx, int col, int row,
                                    int index) {
  int32_t value = XAieTile_DmReadWord(&(ctx->TileInst[col][row]),
                                      pong_c_offset + (index * 4));
  return value;
}
void mlir_aie_write_buffer_pong_c(aie_libxaie_ctx_t *ctx, int col, int row,
                                  int index, int32_t value) {
  int32_t int_value = value;
  return XAieTile_DmWriteWord(&(ctx->TileInst[col][row]),
                              pong_c_offset + (index * 4), int_value);
}

std::string to_air_cfg_name(uint32_t in) {
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

void elf2airbin(std::ifstream &infile, std::ofstream &myfile) {
  int start = 0;
  int stop = 0;
  int total = 4096;

  unsigned char num[4] = {0};
  uint32_t phstart = 0;
  infile.seekg(7 * sizeof(num));
  infile.read(reinterpret_cast<char *>(&num), sizeof(num));
  phstart = swapped(num);

  infile.seekg(phstart + sizeof(num));
  infile.read(reinterpret_cast<char *>(&num), sizeof(num));
  start = swapped(num);
  infile.seekg(phstart + 4 * sizeof(num));
  infile.read(reinterpret_cast<char *>(&num), sizeof(num));
  stop = swapped(num) / sizeof(uint32_t);

  // myfile << "Program Memory:" << std::endl;
  for (int i = 0; i < stop; i++) {
    infile.seekg(start + i * sizeof(num));
    infile.read(reinterpret_cast<char *>(&num), sizeof(num));
    myfile << std::hex << std::setw(8) << std::setfill('0') << swapped(num)
           << std::endl;
  }
  for (int i = stop; i < total; i++) {
    myfile << std::hex << std::setw(8) << std::setfill('0') << uint32_t(0)
           << std::endl;
  }
}

void make_airbin_header(std::ofstream &myfile, uint16_t f_type, uint16_t arch,
                        uint16_t f_ver, uint16_t num_ch, uint32_t chcfg) {
  unsigned char num[4] = {0};
  num[0] = '~' + 1;
  num[1] = 'A';
  num[2] = 'I';
  num[3] = 'R';
  myfile << std::hex << num[0] << num[1] << num[2] << num[3];
  num[0] = 2;
  num[1] = 2;
  num[2] = 1;
  num[3] = 0;
  myfile << std::hex << num[0] << num[1] << num[2] << num[3];
  myfile << std::hex << std::setw(8) << std::setfill('\0') << char(0);
  myfile << std::hex << std::setw(2) << std::setfill('0') << f_type;
  myfile << std::hex << std::setw(2) << std::setfill('0') << arch;
  myfile << std::hex << std::setw(2) << std::setfill('0') << f_ver;
  myfile << std::hex << std::setw(2) << std::setfill('0') << num_ch;
  myfile << std::hex << std::setw(8) << std::setfill('0') << chcfg;
}

void print_airbin_header(std::ifstream &infile) {
  unsigned char num[4] = {0};
  infile.seekg(0);
  infile >> std::hex >> num[0] >> num[1] >> num[2] >> num[3];
  std::cout << num[1] << " " << num[2] << " " << num[3] << std::endl;
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
                    uint32_t *data_pa, int col) {
  uint64_t last_td = 0;

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

  int row = 0;
  int p = 0;
  int bd = 0;
  uint64_t core = getTileAddr(col, row);
  uint64_t next_ptr = uint64_t(tds_pa) + 64 * (bd + 1);
  uint64_t sa = uint64_t(data_pa);

  while (next_chcfg_idx < num_ch) {
    infile.seekg(chcfg + 1 + next_chcfg_idx * 9 * 8);
    Air64_Chdr config_header;
    infile >> std::hex >> config_header.ch_name;
    infile >> std::hex >> config_header.ch_type;
    infile >> std::hex >> config_header.ch_addr;
    infile >> std::hex >> config_header.ch_addr;
    infile >> std::hex >> config_header.ch_offset;
    infile >> std::hex >> config_header.ch_offset;
    infile >> std::hex >> config_header.ch_size;
    infile >> std::hex >> config_header.ch_size;
    infile.seekg(config_header.ch_offset);
    if (config_header.ch_type) {
      if (config_header.ch_name != 11) {
        std::string line_desc;
        std::getline(infile, line_desc);
        // std::cout << std::right << line_desc << " 0x" << std::hex << sa <<
        // std::endl;
      }
      for (int i = 0; i < (config_header.ch_size / sizeof(uint32_t)); i++) {
        uint32_t val;
        infile >> std::hex >> val;
        data_va[p++] = val;
        // std::cout << std::hex << std::setw(8) << std::setfill('0') << val <<
        // std::endl;
      }
      if (config_header.ch_name != 11) {
        std::string line_desc;
        std::getline(infile, line_desc);
      }
      uint64_t da = uint64_t(core + config_header.ch_addr);
      tds_va[16 * bd + 0] = next_ptr & 0xffffffff; // NXTDESC_PNTR
      tds_va[16 * bd + 1] = (next_ptr >> 32);      // NXTDESC_PNTR_MSB
      tds_va[16 * bd + 2] = sa & 0xffffffff;       // SA
      tds_va[16 * bd + 3] = (sa >> 32);            // SA_MSB
      tds_va[16 * bd + 4] = da & 0xffffffff;       // DA
      tds_va[16 * bd + 5] = (da >> 32);            // DA_MSB
      tds_va[16 * bd + 6] = config_header.ch_size; // CONTROL (BTT)
      tds_va[16 * bd + 7] = 0;                     // STATUS
      // std::cout << "Next: " << next_ptr << " SA: " << sa << " DA: " << da <<
      // " BTT: " << config_header.ch_size << std::endl;
      bd++;
      if (next_chcfg_idx + 2 == num_ch) { // FIXME save rather than "last"
        last_td = next_ptr;
        next_ptr = 0;
      } else {
        next_ptr = uint64_t(tds_pa) + 64 * (bd + 1);
      }
      sa += config_header.ch_size;
    }
    next_chcfg_idx++;
    if ((row == 0) && (next_chcfg_idx == 6)) {
      row++;
      core = getTileAddr(col, row);
    } else if ((next_chcfg_idx - 6) % 8 == 0) {
      row++;
      core = getTileAddr(col, row);
    }
  }
  return last_td;
}

int addone_driver(aie_libxaie_ctx_t *xaie, queue_t *q, int col, int row) {
  /////////////////////////////////////////////////////////////////////////////
  //////////////////////// Run Add One Application ////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

#define DMA_COUNT 16

  XAieLib_MemInst *mem2 =
      XAieLib_MemAllocate(2 * DMA_COUNT * sizeof(uint32_t), 0);
  uint32_t *bb_ptr = (uint32_t *)XAieLib_MemGetVaddr(mem2);
  uint32_t *bb_paddr = (uint32_t *)XAieLib_MemGetPaddr(mem2);

  XAieLib_MemSyncForCPU(mem2);
  if (mem2) {
    for (int i = 0; i < DMA_COUNT; i++) {
      bb_ptr[i] = i + 1;
      bb_ptr[DMA_COUNT + i] = 0xdeface;
    }
  } else {
    printf("ERROR: could not allocate memory!\n");
    return 1;
  }
  XAieLib_MemSyncForDev(mem2);

  for (int i = 0; i < 8; i++) {
    mlir_aie_write_buffer_ping_in(xaie, col, row, i, 0xabbaba00 + i);
    mlir_aie_write_buffer_pong_in(xaie, col, row, i, 0xdeeded00 + i);
    mlir_aie_write_buffer_ping_out(xaie, col, row, i, 0x12345670 + i);
    mlir_aie_write_buffer_pong_out(xaie, col, row, i, 0x76543210 + i);
  }

  //// setup the herd
  // uint64_t wr_idx = queue_add_write_index(q, 1);
  // uint64_t packet_id = wr_idx % q->size;
  // dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) +
  // packet_id; air_packet_herd_init(herd_pkt, 0, col, 1, row, 3);
  // air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  // wr_idx = queue_add_write_index(q, 1);
  // packet_id = wr_idx % q->size;
  // dispatch_packet_t *shim_pkt = (dispatch_packet_t*)(q->base_address_vaddr) +
  // packet_id; air_packet_device_init(shim_pkt,XAIE_NUM_COLS);
  // air_queue_dispatch_and_wait(q, wr_idx, shim_pkt);

  mlir_aie_print_tile_status(xaie, col, row);
  mlir_aie_print_dma_status(xaie, col, row);

  //
  // send the data
  //

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt1 =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt1, 0, col, 1, 0, 4, 2, (uint64_t)bb_paddr,
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);

  //
  // read the data
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt2 =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt2, 0, col, 0, 0, 4, 2,
                       (uint64_t)bb_paddr + (DMA_COUNT * sizeof(float)),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, pkt2);

  mlir_aie_print_tile_status(xaie, col, row);
  mlir_aie_print_dma_status(xaie, col, row);

  int errors = 0;

  for (int i = 0; i < 8; i++) {
    uint32_t d0 = mlir_aie_read_buffer_ping_in(xaie, col, row, i);
    uint32_t d1 = mlir_aie_read_buffer_pong_in(xaie, col, row, i);
    uint32_t d2 = mlir_aie_read_buffer_ping_out(xaie, col, row, i);
    uint32_t d3 = mlir_aie_read_buffer_pong_out(xaie, col, row, i);
    if (d0 + 1 != d2) {
      printf("mismatch ping %x != %x\n", d0, d2);
      errors++;
    }
    if (d1 + 1 != d3) {
      printf("mismatch pong %x != %x\n", d1, d3);
      errors++;
    }
  }

  XAieLib_MemSyncForCPU(mem2);
  for (int i = 0; i < DMA_COUNT; i++) {
    uint32_t d = bb_ptr[DMA_COUNT + i];
    if (d != (i + 2)) {
      errors++;
      printf("mismatch %x != 2 + %x\n", d, i);
    }
  }

  XAieLib_MemFree(mem2);

  if (!errors) {
    return 0;
  } else {
    return 1;
  }
}

int matadd_driver(aie_libxaie_ctx_t *xaie, queue_t *q, int col, int row) {
// test configuration
#define IMAGE_WIDTH 192
#define IMAGE_HEIGHT 192
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT)

#define NUM_3D (IMAGE_WIDTH / TILE_WIDTH)
#define NUM_4D (IMAGE_HEIGHT / TILE_HEIGHT)

  XAieLib_MemInst *mem2 =
      XAieLib_MemAllocate(3 * IMAGE_SIZE * sizeof(uint32_t), 0);
  uint32_t *dram_ptr = (uint32_t *)XAieLib_MemGetVaddr(mem2);
  uint32_t *dram_paddr = (uint32_t *)XAieLib_MemGetPaddr(mem2);

  XAieLib_MemSyncForCPU(mem2);
  if (mem2) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
      dram_ptr[i] = i + 1;
      dram_ptr[IMAGE_SIZE + i] = i + 2;
      dram_ptr[2 * IMAGE_SIZE + i] = 0xdeface;
    }
  } else {
    printf("ERROR: could not allocate memory!\n");
    return 1;
  }
  XAieLib_MemSyncForDev(mem2);

  // stamp over the aie tiles
  for (int i = 0; i < TILE_SIZE; i++) {
    mlir_aie_write_buffer_ping_a(xaie, col, row, i, 0xabba0000 + i);
    mlir_aie_write_buffer_pong_a(xaie, col, row, i, 0xdeeded00 + i);
    mlir_aie_write_buffer_ping_b(xaie, col, row, i, 0xcafe0000 + i);
    mlir_aie_write_buffer_pong_b(xaie, col, row, i, 0xfabcab00 + i);
    mlir_aie_write_buffer_ping_c(xaie, col, row, i, 0x12345670 + i);
    mlir_aie_write_buffer_pong_c(xaie, col, row, i, 0x76543210 + i);
  }

  //// setup the herd
  // uint64_t wr_idx = queue_add_write_index(q, 1);
  // uint64_t packet_id = wr_idx % q->size;
  // dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) +
  // packet_id; air_packet_herd_init(herd_pkt, 0, col, 1, row, 3);
  // air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  // wr_idx = queue_add_write_index(q, 1);
  // packet_id = wr_idx % q->size;
  // dispatch_packet_t *shim_pkt = (dispatch_packet_t*)(q->base_address_vaddr) +
  // packet_id; air_packet_device_init(shim_pkt,XAIE_NUM_COLS);
  // air_queue_dispatch_and_wait(q, wr_idx, shim_pkt);

  mlir_aie_print_tile_status(xaie, col, row);
  mlir_aie_print_dma_status(xaie, col, row);

  //
  // packet to read the output matrix
  //

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_c =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_c, 0, col, 0, 0, 4, 2,
                       uint64_t(dram_paddr) + (2 * IMAGE_SIZE * sizeof(float)),
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  //
  // packet to send the input matrices
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_a =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_a, 0, col, 1, 0, 4, 2, uint64_t(dram_paddr),
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_b =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_b, 0, col, 1, 1, 4, 2,
                       uint64_t(dram_paddr) + (IMAGE_SIZE * sizeof(float)),
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  //
  // dispatch the packets to the MB
  //

  air_queue_dispatch_and_wait(q, wr_idx, pkt_c);

  mlir_aie_print_tile_status(xaie, col, row);
  mlir_aie_print_dma_status(xaie, col, row);

  int errors = 0;
  // check the aie tiles
  for (int i = 0; i < TILE_SIZE; i++) {
    uint32_t d0 = mlir_aie_read_buffer_ping_a(xaie, col, row, i);
    uint32_t d1 = mlir_aie_read_buffer_pong_a(xaie, col, row, i);
    uint32_t d4 = mlir_aie_read_buffer_ping_b(xaie, col, row, i);
    uint32_t d5 = mlir_aie_read_buffer_pong_b(xaie, col, row, i);
    uint32_t d2 = mlir_aie_read_buffer_ping_c(xaie, col, row, i);
    uint32_t d3 = mlir_aie_read_buffer_pong_c(xaie, col, row, i);
    if (d0 + d4 != d2) {
      printf("mismatch [%d] ping %x+%x != %x\n", i, d0, d4, d2);
      errors++;
    }
    if (d1 + d5 != d3) {
      printf("mismatch [%d] pong %x+%x != %x\n", i, d1, d5, d3);
      errors++;
    }
  }

  // check the output image
  for (int i = 0; i < IMAGE_SIZE; i++) {
    uint32_t d = dram_ptr[2 * IMAGE_SIZE + i];
    if (d != ((i + 1) + (i + 2))) {
      errors++;
      printf("mismatch %x != %x\n", d, 2 * (i + 1));
    }
  }

  XAieLib_MemFree(mem2);

  if (!errors) {
    return 0;
  } else {
    return 1;
  }
}

void configure_herd(queue_t *q, int col, int row, std::string airbin_name) {
  uint8_t start_col = col;
  uint8_t num_cols = 1;
  uint8_t start_row = 1;
  uint8_t num_rows = row;

  XAieLib_MemInst *mem = XAieLib_MemAllocate(2 * 65536, XAIELIB_MEM_ATTR_CACHE);
  volatile uint32_t *bram_ptr = (volatile uint32_t *)XAieLib_MemGetVaddr(mem);
  uint32_t *paddr = (uint32_t *)XAieLib_MemGetPaddr(mem);
  XAieLib_MemInst *mem2 = XAieLib_MemAllocate(0x4000, XAIELIB_MEM_ATTR_CACHE);
  volatile uint32_t *bd_ptr = (volatile uint32_t *)XAieLib_MemGetVaddr(mem2);
  uint64_t bd_paddr = uint64_t(XAieLib_MemGetPaddr(mem2));

  std::ifstream infile;
  infile.open(airbin_name);

  std::cout << std::endl
            << "Configuring herd in col " << col << "..." << std::endl;

  XAieLib_MemSyncForCPU(mem);
  XAieLib_MemSyncForCPU(mem2);
  uint64_t last_td =
      airbin2mem(infile, bd_ptr, (uint32_t *)bd_paddr, bram_ptr, paddr, col);
  XAieLib_MemSyncForDev(mem);
  XAieLib_MemSyncForDev(mem2);
  // Send configuration packet to MicroBlaze
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_cdma_memcpy(pkt, last_td, uint64_t(bd_paddr), 0xffffffff);
  pkt->type = 0x31;
  pkt->arg[3] = 0;
  pkt->arg[3] |= ((uint64_t)num_cols) << 24;
  pkt->arg[3] |= ((uint64_t)start_col) << 16;
  pkt->arg[3] |= ((uint64_t)num_rows) << 8;
  pkt->arg[3] |= ((uint64_t)start_row);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  std::cout << std::endl << "Done configuring!" << std::endl << std::endl;

  XAieLib_MemFree(mem);
  XAieLib_MemFree(mem2);

  infile.close();
}

int main(int argc, char **argv) {
  std::string airbin_name_1 = "addone.airbin";
  std::string airbin_name_2 = "matadd.airbin";

  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(xaie);

  std::vector<air_agent_t> agents;
  auto ret = air_get_agents(&agents);
  assert(ret == 0 && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "fail." << std::endl;
    return -1;
  }

  std::vector<queue_t *> queues;
  for (auto agent : agents) {
    // create the queue
    queue_t *q = nullptr;
    ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                           agent.handle);
    assert(ret == 0 && "failed to create queue!");
    queues.push_back(q);
  }

  configure_herd(queues[0], 6, 2, airbin_name_1);
  configure_herd(queues[1], 7, 2, airbin_name_2);

  int errors = 0;
  errors += addone_driver(xaie, queues[0], 6, 2);
  errors += matadd_driver(xaie, queues[1], 7, 2);

  if (!errors) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << "fail." << std::endl;
    return -1;
  }
}

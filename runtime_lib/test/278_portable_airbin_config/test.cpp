// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

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

// #include <xaiengine.h>

#include "air_host.h"
#include "air_queue.h"
#include "hsa_defs.h"

#include <ctime>
#include <sys/time.h>

float time_spec_diff(struct timespec *start, struct timespec *end) {
  return (end->tv_sec - start->tv_sec) + 1e-9 * (end->tv_nsec - start->tv_nsec);
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

#define TD_ADDR 0x800001000
#define DATA_ADDR 0x800002000

#define BAR_PF0_DEV_FILE_DDR "/sys/bus/pci/devices/0000:21:00.0/resource0"
#define BAR_PF0_DEV_FILE_AIE "/sys/bus/pci/devices/0000:21:00.0/resource2"
#define BAR_PF0_DEV_FILE_BRAM "/sys/bus/pci/devices/0000:21:00.0/resource4"

#define u16 uint16_t
#define u32 uint32_t
#define u64 uint64_t

#define AIE_NUM_ROWS 8
#define AIE_NUM_COLS 50
#define AIE_ADDR_ARRAY_OFF 0x800

#define AIEGBL_TILE_ADDR_ARR_SHIFT 30U
#define AIEGBL_TILE_ADDR_ROW_SHIFT 18U
#define AIEGBL_TILE_ADDR_COL_SHIFT 23U

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

uint32_t swapped2(uint32_t num) {
  uint32_t val = 0;
  val = num;
  // val = ((num>>24)&0xff) |      // move byte 3 to byte 0
  //       ((num<<8)&0xff0000) |   // move byte 1 to byte 2
  //       ((num>>8)&0xff00) |     // move byte 2 to byte 1
  //       ((num<<24)&0xff000000); // byte 0 to byte 3
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
  u64 ArrOffset = AIE_ADDR_ARRAY_OFF;

#ifdef AIE_BASE_ARRAY_ADDR_OFFSET
  ArrOffset = AIE_BASE_ARRAY_ADDR_OFFSET;
#endif

  /*
   * Tile address format:
   * --------------------------------------------
   * |                7 bits  5 bits   18 bits  |
   * --------------------------------------------
   * | Array offset | Column | Row | Tile addr  |
   * --------------------------------------------
   */
  TileAddr = (u64)((ArrOffset << AIEGBL_TILE_ADDR_ARR_SHIFT) |
                   (ColIdx << AIEGBL_TILE_ADDR_COL_SHIFT) |
                   (RowIdx << AIEGBL_TILE_ADDR_ROW_SHIFT));

  return TileAddr;
}

u64 getTileAddr_BAR(u16 ColIdx, u16 RowIdx) {
  u64 TileAddr = 0;
  u64 ArrOffset = 0;

  /*
   * Tile address format:
   * --------------------------------------------
   * |                7 bits  5 bits   18 bits  |
   * --------------------------------------------
   * | Array offset | Column | Row | Tile addr  |
   * --------------------------------------------
   */
  TileAddr = (u64)((ArrOffset << AIEGBL_TILE_ADDR_ARR_SHIFT) |
                   (ColIdx << AIEGBL_TILE_ADDR_COL_SHIFT) |
                   (RowIdx << AIEGBL_TILE_ADDR_ROW_SHIFT));

  return TileAddr;
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

int addone_driver(queue_t *q, int col, int row) {
  /////////////////////////////////////////////////////////////////////////////
  //////////////////////// Run Add One Application ////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

#define DMA_COUNT 16

  char bar_dev_file[100];
  strcpy(bar_dev_file, BAR_PF0_DEV_FILE_DDR);

  // Opening BAR
  int fd;
  if ((fd = open(bar_dev_file, O_RDWR | O_SYNC)) == -1) {
    printf("[ERROR] Failed to open device file\n");
    return 1;
  }

  // Map the memory region into userspace
  void *map_axib_base = mmap(NULL,                   // virtual address
                             0x2000000,              // length
                             PROT_READ | PROT_WRITE, // prot
                             MAP_SHARED,             // flags
                             fd,                     // device fd
                             0);                     // offset

  volatile uint32_t *bb_ptr = (uint32_t *)(map_axib_base);
  uint64_t paddr = 0x800000000;

  for (int i = 0; i < DMA_COUNT; i++) {
    bb_ptr[i] = i + 1;
    bb_ptr[DMA_COUNT + i] = 0xdeface;
  }

  //
  // send the data
  //

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt1 =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt1, 0, col, 1, 0, 4, 2, paddr,
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);

  //
  // read the data
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt2 =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt2, 0, col, 0, 0, 4, 2,
                       paddr + (DMA_COUNT * sizeof(float)),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, pkt2);

  int errors = 0;

  for (int i = 0; i < DMA_COUNT; i++) {
    volatile uint32_t d = bb_ptr[DMA_COUNT + i];
    if (d != (i + 2)) {
      errors++;
      printf("mismatch %x != 2 + %x\n", d, i);
    }
  }

  munmap(map_axib_base, 0x2000000);

  if (!errors) {
    return 0;
  } else {
    return 1;
  }
}

int matadd_driver(queue_t *q, int col, int row) {
// test configuration
#define IMAGE_WIDTH 192
#define IMAGE_HEIGHT 192
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT)

#define NUM_3D (IMAGE_WIDTH / TILE_WIDTH)
#define NUM_4D (IMAGE_HEIGHT / TILE_HEIGHT)

  char bar_dev_file[100];
  strcpy(bar_dev_file, BAR_PF0_DEV_FILE_DDR);

  // Opening BAR
  int fd;
  if ((fd = open(bar_dev_file, O_RDWR | O_SYNC)) == -1) {
    printf("[ERROR] Failed to open device file\n");
    return 1;
  }

  // Map the memory region into userspace
  void *map_axib_base = mmap(NULL,                   // virtual address
                             0x2000000,              // length
                             PROT_READ | PROT_WRITE, // prot
                             MAP_SHARED,             // flags
                             fd,                     // device fd
                             0);                     // offset

  volatile uint32_t *dram_ptr = (uint32_t *)(map_axib_base);
  uint64_t dram_paddr = 0x800000000;

  if (dram_ptr) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
      dram_ptr[i] = i + 1;
      dram_ptr[IMAGE_SIZE + i] = i + 2;
      dram_ptr[2 * IMAGE_SIZE + i] = 0xdeface;
    }
  } else {
    printf("ERROR: could not allocate memory!\n");
    return 1;
  }

  //
  // packet to read the output matrix
  //

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_c =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_c, 0, col, 0, 0, 4, 2, dram_paddr + (2 * IMAGE_SIZE * sizeof(float)),
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  //
  // packet to send the input matrices
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_a =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_a, 0, col, 1, 0, 4, 2, dram_paddr,
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_b =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_b, 0, col, 1, 1, 4, 2, dram_paddr + (IMAGE_SIZE * sizeof(float)),
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  //
  // dispatch the packets to the MB
  //

  air_queue_dispatch_and_wait(q, wr_idx, pkt_b);

  int errors = 0;

  // check the output image
  for (int i = 0; i < IMAGE_SIZE; i++) {
    uint32_t d = dram_ptr[2 * IMAGE_SIZE + i];
    if (d != ((i + 1) + (i + 2))) {
      errors++;
      printf("mismatch %x != %x\n", d, 2 * (i + 1));
    }
  }

  munmap(map_axib_base, 0x2000000);

  if (!errors) {
    return 0;
  } else {
    return 1;
  }
}

int main(int argc, char **argv) {
  int col = 7;
  int row = 2;
  char *airbin_name;

  if (argc == 3) {
    col = atoi(argv[2]);
    airbin_name = argv[1];
  } else if (argc == 2) {
    col = 7;
    airbin_name = argv[1];
  } else {
    col = 7;
    airbin_name = strdup("addone.airbin");
  }

  uint8_t start_col = col;
  uint8_t num_cols = 1;
  uint8_t start_row = 1;
  uint8_t num_rows = 3;

  // Opening the correct bar
  char bar_dev_file[100];
  strcpy(bar_dev_file, BAR_PF0_DEV_FILE_DDR);
  printf("Opening %s to access MMIO DDR\n", bar_dev_file);

  // Opening BAR
  int fd;
  if ((fd = open(bar_dev_file, O_RDWR | O_SYNC)) == -1) {
    printf("[ERROR] Failed to open device file\n");
    return 1;
  }
  printf("device file opened\n");

  // Map the memory region into userspace
  volatile void *map_axib_base = mmap(NULL,                   // virtual address
                                      0x2000000,              // length
                                      PROT_READ | PROT_WRITE, // prot
                                      MAP_SHARED,             // flags
                                      fd,                     // device fd
                                      0x1000);                // offset
  printf("memory mapped into userspace\n");

  // Assigning virt_addr to the device memory
  volatile uint64_t *virt_addr = (volatile uint64_t *)(map_axib_base);
  printf("virtual address = 0x%lx\n", (unsigned long)virt_addr);

  volatile uint32_t *bram_ptr =
      (volatile uint32_t *)((u64)virt_addr + 0x1000); // Config data ptr
  uint32_t *paddr = (uint32_t *)(DATA_ADDR);          // Config data paddr
  volatile uint32_t *bd_ptr = (volatile uint32_t *)(virt_addr); // TD data ptr
  uint64_t bd_paddr = uint64_t(TD_ADDR);                        // TD paddr

  std::ifstream infile;
  infile.open(airbin_name);

  std::cout << std::endl
            << "Configuring herd in col " << col << "..." << std::endl;

  // Copy airbin data to DDR
  // print_airbin_header(infile);
  readairbin(infile);

  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  // struct timespec ts_start;
  // struct timespec ts_end;

  uint64_t last_td =
      airbin2mem(infile, bd_ptr, (uint32_t *)bd_paddr, bram_ptr, paddr, col);
  std::cout << std::endl << "Done writing config data to memory!" << std::endl;
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
  // clock_gettime(CLOCK_BOOTTIME, &ts_start);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);
  // clock_gettime(CLOCK_BOOTTIME, &ts_end);

  // printf("config time: %0.8f sec\n",
  //           time_spec_diff(&ts_start, &ts_end));

  std::cout << std::endl << "Done configuring!" << std::endl << std::endl;

  int errors = 0;

  if (strcmp(airbin_name, "addone.airbin") == 0) {
    errors = addone_driver(q, col, row);
  } else if (strcmp(airbin_name, "matadd.airbin") == 0) {
    errors = matadd_driver(q, col, row);
  } else
    errors = 1;

  munmap(map_axib_base, 0x2000000);

  infile.close();

  if (!errors) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << "fail." << std::endl;
    return -1;
  }
}

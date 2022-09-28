// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include <assert.h>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <sys/mman.h>
#include <string>
#include <vector>
//#include <xaiengine.h>

#include "../../airhost/include/acdc_queue.h"
#include "../../airhost/include/air_host.h"
#include "../../airhost/include/hsa_defs.h"

#define BRAM_ADDR 0x20100006000

#define BAR_PF0_DEV_FILE_DDR    "/sys/bus/pci/devices/0000:21:00.0/resource2" // Change to resource4 for testing as that memory is just backed by DDR
#define BAR_PF0_DEV_FILE_BRAM   "/sys/bus/pci/devices/0000:21:00.0/resource4" // Change to resource4 for testing as that memory is just backed by BRAM

#define XAIE_ADDR_ARRAY_OFF     0x800
#define XAIE_NUM_COLS           50
#define XAIE_NUM_ROWS            8

#define XAIEGBL_TILE_ADDR_ARR_SHIFT         30U
#define XAIEGBL_TILE_ADDR_ROW_SHIFT         18U
#define XAIEGBL_TILE_ADDR_COL_SHIFT         23U

#define u16 uint16_t
#define u32 uint32_t
#define u64 uint64_t

u64 getTileAddr(u16 ColIdx, u16 RowIdx) 
{
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
  TileAddr = (u64)((ArrOffset <<
      XAIEGBL_TILE_ADDR_ARR_SHIFT) |
    (ColIdx << XAIEGBL_TILE_ADDR_COL_SHIFT) |
    (RowIdx << XAIEGBL_TILE_ADDR_ROW_SHIFT));

  return TileAddr;
}

u64 getTileAddr_BAR(u16 ColIdx, u16 RowIdx)
{
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
  TileAddr = (u64)((ArrOffset <<
      XAIEGBL_TILE_ADDR_ARR_SHIFT) |
    (ColIdx << XAIEGBL_TILE_ADDR_COL_SHIFT) |
    (RowIdx << XAIEGBL_TILE_ADDR_ROW_SHIFT));

  return TileAddr;
}

uint32_t swapped2(uint32_t num) {
  uint32_t val = 0;
  return num;
  //val = ((num>>24)&0xff) |      // move byte 3 to byte 0
  //      ((num<<8)&0xff0000) |   // move byte 1 to byte 2
  //      ((num>>8)&0xff00) |     // move byte 2 to byte 1
  //      ((num<<24)&0xff000000); // byte 0 to byte 3
  //return val;
}

int main(int argc, char *argv[]) {
  int col = 7;
  int col2 = 19;

  if (argc == 2) { 
    col2 = atoi(argv[1]);
  } else {
    col2 = 7;
  }

  //aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  //mlir_aie_init_device(xaie);

  //int fd = open("/dev/mem", O_RDWR | O_SYNC);
  //if (fd == -1) {
  //  printf("failed to open /dev/mem\n");
  //  return -1;
  //}

  //volatile uint32_t *bram_ptr = (volatile uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);

  //std::vector<air_agent_t> agents;
  //auto ret = air_get_agents(&agents);
  //assert(ret == 0 && "failed to get agents!");

  //if (agents.empty()) {
  //  std::cout << "fail." << std::endl;
  //  return -1;
  //}

  //std::vector<queue_t *> queues;
  //for (auto agent : agents) {
  //  // create the queue
  //  queue_t *q = nullptr;
  //  ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
  //                         agent.handle);
  //  assert(ret == 0 && "failed to create queue!");
  //  queues.push_back(q);
  //}

  //auto q = queues[0];
  char bar_dev_file[100];
  strcpy(bar_dev_file, BAR_PF0_DEV_FILE_DDR);
  printf("Opening %s to access MMIO DDR\n", bar_dev_file);

  // Opening BAR2
  int fd2;
  if((fd2 = open(bar_dev_file, O_RDWR | O_SYNC)) == -1) {
      printf("[ERROR] Failed to open device file\n");
      return 1;
  }
  printf("device file opened\n");

  // Map the memory region into userspace
  volatile void *map_aie_base = mmap(NULL,             // virtual address
                      0x10000000,             // length
                      PROT_READ | PROT_WRITE, // prot
                      MAP_SHARED,             // flags
                      fd2,                    // device fd
                      0);                     // offset
  printf("memory mapped into userspace\n");

  //for (int r=1; r<3; r++) {
  //  volatile uint32_t *tile_ctrl_addr = (volatile uint32_t *)(map_aie_base+getTileAddr_BAR(col2,r)+0x32000);
  //  printf("Core %d,%d status reg: %x\n",col2,r,tile_ctrl_addr[1]);
  //  *tile_ctrl_addr = 0x2; 
  //  printf("Core %d,%d status reg: %x\n",col2,r,tile_ctrl_addr[1]);
  //} 

  //volatile uint32_t *shim_addr = (volatile uint32_t *)(map_aie_base+getTileAddr_BAR(col2,0)+0x36048);
  //*shim_addr = 1;
  //*shim_addr = 0;

  //for (int r=1; r<3; r++) {
  //  volatile uint32_t *v_addr = (volatile uint32_t *)(map_aie_base+getTileAddr_BAR(col2,r)+0x3f000);
  //  for (int i=0; i<24; i++) {
  //    v_addr[i] = 0xfeedcafe;
  //  }
  //}

  //for (int r=1; r<3; r++) {
  //  volatile uint32_t *v_addr = (volatile uint32_t *)(map_aie_base+getTileAddr_BAR(col2,r)+0x3f000);
  //  for (int i=0; i<24; i++) {
  //    printf("Data for %d,%d @ %lx : %d = %x\n",col2,r,getTileAddr_BAR(col2,r)+0x3f000,i,v_addr[i]);
  //  }
  //}

  char bar_dev_file_BRAM[100];
  strcpy(bar_dev_file_BRAM, BAR_PF0_DEV_FILE_BRAM);
  printf("Opening %s to access MMIO BRAM\n", bar_dev_file_BRAM);
 
  int fd;
  if((fd = open(bar_dev_file_BRAM, O_RDWR | O_SYNC)) == -1) {
    printf("[ERROR] Failed to open device file\n");
    return 1;
  }

  void *map_axib_base = mmap(NULL,                 // virtual address
  0x20000,                // length
  PROT_READ | PROT_WRITE, // prot
  MAP_SHARED,             // flags
  fd, 0);

  //XAieLib_MemInst *mem = XAieLib_MemAllocate(65536, 0);
  //volatile uint32_t *bram_ptr = (volatile uint32_t *)XAieLib_MemGetVaddr(mem);
  volatile uint32_t *bram_ptr = (volatile uint32_t *)(map_axib_base + 0x6000);
  //uint32_t *paddr = (uint32_t *)XAieLib_MemGetPaddr(mem);
  for (int i = 0; i < 0x1000; i++) {
    bram_ptr[i] = 0xabab;
  }
  //XAieLib_MemSyncForDev(mem);

  //std::vector<queue_t *> queues;
  //for (auto agent : agents) {
    // create the queue
    queue_t *q = nullptr;
    auto ret = air5000_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                                    0, bar_dev_file_BRAM);
    assert(ret == 0 && "failed to create queue!");

  for (int row = 0; row < 1; row++) {
    printf("\nWriting AIE shim %d,%d...\n",col2,row);
    uint64_t core = getTileAddr(col2,row);

    std::ifstream myfile;
    myfile.open ("core_"+std::to_string(col)+"_"+std::to_string(row)+"_fat_binary.txt");

    for (int i = 0; i < 0x1000; i++) {
      bram_ptr[i] = 0x0;
    }

    int p = 0;

    std::string line_desc;
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    for (int i = 0; i < 23; i++) {
      uint32_t val;
      myfile >> std::hex >> val;
      bram_ptr[p++] = swapped2(val);
      //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    }
    std::getline(myfile, line_desc);
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    for (int i = 0; i < 23; i++) {
      uint32_t val;
      myfile >> std::hex >> val;
      bram_ptr[p++] = swapped2(val);
      //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    }
    std::getline(myfile, line_desc);
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    for (int i = 0; i < 92; i++) {
      uint32_t val;
      myfile >> std::hex >> val;
      bram_ptr[p++] = swapped2(val);
      //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    }
    std::getline(myfile, line_desc);
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    for (int b = 0; b < 16*5; b++) {
      uint32_t val;
      myfile >> std::hex >> val;
      bram_ptr[p++] = swapped2(val);
      //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    }
    std::getline(myfile, line_desc);
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    uint32_t val;
    myfile >> std::hex >> val;
    bram_ptr[p++] = swapped2(val);
    //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    myfile >> std::hex >> val;
    bram_ptr[p++] = swapped2(val);
    //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    std::getline(myfile, line_desc);
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    for (int i = 0; i < 8; i++) {
      uint32_t val;
      myfile >> std::hex >> val;
      bram_ptr[p++] = swapped2(val);
      //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    }
    myfile.close();

    uint64_t wr_idx = queue_add_write_index(q, 1);
    uint64_t packet_id = wr_idx % q->size;
    dispatch_packet_t* pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x1)  << 32;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt, uint64_t(core+0x0003F000), uint64_t(BRAM_ADDR), 92);

    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt, uint64_t(core+0x0003F100), uint64_t(BRAM_ADDR+92), 92);

    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt, uint64_t(core+0x0003F200), uint64_t(BRAM_ADDR+184), 368);

    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt, uint64_t(core+0x0001D000), uint64_t(BRAM_ADDR+552), 320);

    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt,  uint64_t(core+0x0001F000), uint64_t(BRAM_ADDR+872), 8);

    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt, uint64_t(core+0x0001D140), uint64_t(BRAM_ADDR+880), 32);

    air_queue_dispatch_and_wait(q, wr_idx, pkt);

  }
  for (int row = 1; row < 3; row++) {
    printf("\nWriting AIE core %d,%d...\n",col2,row);
    uint64_t core = getTileAddr(col2,row);

    std::ifstream myfile;
    myfile.open ("core_"+std::to_string(col)+"_"+std::to_string(row)+"_fat_binary.txt");

    int p = 0;
    for (int i = 0; i < 0x1000; i++) {
      bram_ptr[i] = 0x0;
    }

    //XAieTile_CoreControl(&(xaie->TileInst[col][row]), XAIE_DISABLE, XAIE_ENABLE);
    std::string line_desc;
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    for (int i = 0; i < 0x1000; i++) {
      uint32_t val;
      myfile >> std::hex >> val;
      bram_ptr[p++] = swapped2(val);
      //std::cout << std::hex << std::setw(8) << std::setfill('0') << swapped2(val) << std::endl;
    }

    // FIXME
    //for (int l=0; l<16; l++)
    //  XAieTile_LockRelease(&(xaie->TileInst[col][row]), l, 0x0, 0);

    uint64_t wr_idx = queue_add_write_index(q, 1);
    uint64_t packet_id = wr_idx % q->size;
    dispatch_packet_t* pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x2)  << 32;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt, uint64_t(core+0x20000), uint64_t(BRAM_ADDR), 0x4000);

    air_queue_dispatch_and_wait(q, wr_idx, pkt);

    //if (core == 7 && row == 2) {
    //  int ret = XAieGbl_LoadElf(&(xaie->TileInst[7][2]), (u8*)"core_7_2.elf", XAIE_ENABLE);
    //  if (ret == XAIELIB_FAILURE)
    //  printf("Failed to load elf for Core[%d,%d], ret is %d", 7, 2, ret);
    //  assert(ret != XAIELIB_FAILURE);
    //}

    //XAieTile_CoreControl(&(xaie->TileInst[col][row]), XAIE_ENABLE, XAIE_DISABLE);

    std::getline(myfile, line_desc);
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    p = 0;
    for (int i = 0; i < 25; i++) {
      uint32_t val;
      myfile >> std::hex >> val;
      bram_ptr[p++] = swapped2(val);
      //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    }
    std::getline(myfile, line_desc);
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    for (int i = 0; i < 27; i++) {
      uint32_t val;
      myfile >> std::hex >> val;
      bram_ptr[p++] = swapped2(val);
      //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    }
    std::getline(myfile, line_desc);
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    for (int i = 0; i < 100; i++) {
      uint32_t val;
      myfile >> std::hex >> val;
      bram_ptr[p++] = swapped2(val);
      //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    }
    std::getline(myfile, line_desc);
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    for (int b = 0; b < 16*8; b++) {
      uint32_t val;
      myfile >> std::hex >> val;
      bram_ptr[p++] = swapped2(val);
      //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    }
    std::getline(myfile, line_desc);
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    for (int i = 0; i < 8; i++) {
      uint32_t val;
      myfile >> std::hex >> val;
      bram_ptr[p++] = swapped2(val);
      //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    }

    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt, uint64_t(core+0x0003F000), uint64_t(BRAM_ADDR), 100);

    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt, uint64_t(core+0x0003F100), uint64_t(BRAM_ADDR+100), 108);

    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt, uint64_t(core+0x0003F200), uint64_t(BRAM_ADDR+208), 400);

    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt, uint64_t(core+0x0001D000), uint64_t(BRAM_ADDR+608), 512);

    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt, uint64_t(core+0x0001DE00), uint64_t(BRAM_ADDR+1120), 32);

    air_queue_dispatch_and_wait(q, wr_idx, pkt);

    std::getline(myfile, line_desc);
    std::getline(myfile, line_desc);
    std::cout << line_desc << std::endl;
    p = 0;
    for (int i = 0; i < 0x1000; i++) {
      uint32_t val;
      myfile >> std::hex >> val;
      bram_ptr[p++] = swapped2(val);
      //std::cout << std::hex << std::setw(8) << std::setfill('0') << val << std::endl;
    }

    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    pkt->arg[3]  = 0;
    pkt->arg[3] |= ((uint64_t)0x1)  << 24;
    pkt->arg[3] |= ((uint64_t)col2)  << 16;
    pkt->arg[3] |= ((uint64_t)0x1)  << 8;
    pkt->arg[3] |= ((uint64_t)row);
    air_packet_cdma_memcpy(pkt, uint64_t(core+0x0), uint64_t(BRAM_ADDR), 0x4000);

    air_queue_dispatch_and_wait(q, wr_idx, pkt);

    myfile.close();
  }

  uint64_t sec = 0x1de00;
  int size = 0x20 / sizeof(uint32_t);
  volatile uint32_t *addr = (volatile uint32_t *)(map_aie_base+getTileAddr_BAR(col2,2)+sec);
  for (int r=1; r<3; r++) {
    volatile uint32_t *v_addr = (volatile uint32_t *)(map_aie_base+getTileAddr_BAR(col2,r)+sec);
    for (int i=0; i<size; i++) {
      printf("Data for %d,%d @ %lx : %d = %x\n",col2,r,getTileAddr_BAR(col2,r)+sec+i*4,i,v_addr[i]);
    }
  }
  for (int r=1; r<3; r++) {
  //  for (int l=0; l<16; l++) {
  //    volatile uint32_t *tile_lock_addr = (volatile uint32_t *)(map_aie_base+getTileAddr_BAR(col2,r)+0x1E020+0x80*l);
  //    uint32_t Count = 10;
  //    while (Count > 0U) {
  //      if ((*(tile_lock_addr) & 0x1) == 0x1) {
  //        break;
  //      }
  //      Count--;
  //    }
  //  }
    volatile uint32_t *tile_ctrl_addr = (volatile uint32_t *)(map_aie_base+getTileAddr_BAR(col2,r)+0x32000);
  //  printf("Core %d,%d status reg: %x\n",col2,r,tile_ctrl_addr[1]);
  //  *tile_ctrl_addr = 0x1; 
    printf("Core %d,%d status reg: %x\n",col2,r,tile_ctrl_addr[1]);
  } 

  std::cout << std::endl << "PASS!" << std::endl;
  return 0;
}

//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <xaiengine.h>
#include "test_library.h"

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

#define MLIR_STACK_OFFSET 4096

#define LOCK_TIMEOUT 1000

//define some constants, CAREFUL NEED TO MATCH MLIR
#define LINE_WIDTH 16
#define HEIGHT 10

#define MAP_SIZE 16UL
#define MAP_MASK (MAP_SIZE - 1)

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

#include "aie_inc.cpp"

}

void devmemRW32(uint32_t address, uint32_t value, bool write){  
    int fd;  uint32_t *map_base;  
    uint32_t read_result;  
    uint32_t offset = address - 0xF70A0000;
    
    if ((fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1)    
        printf("ERROR!!!! open(devmem)\n");  
    printf("\n/dev/mem opened.\n");  
    fflush(stdout);
    
    map_base = (uint32_t *)mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0xF70A0000);  
    if (map_base == (void *)-1)    
        printf("ERROR!!!! map_base\n");  printf("Memory mapped at address %p.\n", map_base);  fflush(stdout);
    read_result = map_base[uint32_t(offset / 4)];  
    printf("Value at address 0x%X: 0x%X\n", address, read_result);  
    fflush(stdout);
    if (write)  {    
        map_base[uint32_t(offset / 4)] = value;    
        //msync(map_base, MAP_SIZE, MS_SYNC);    
        read_result = map_base[uint32_t(offset / 4)];    
        printf("Written 0x%X; readback 0x%X\n", value, read_result);    
        fflush(stdout);  
    }
    //msync(map_base, MAP_SIZE, MS_SYNC);  
  
    if (munmap(map_base, MAP_SIZE) == -1)    
        printf("ERROR!!!! unmap_base\n");  
    printf("/dev/mem closed.\n");  
    fflush(stdout);  
    close(fd);
}

int main(int argc, char *argv[])
{
    printf("test start.\n");

    devmemRW32(0xF70A000C, 0xF9E8D7C6, true);
    devmemRW32(0xF70A0000, 0x04000000, true);
    devmemRW32(0xF70A0004, 0x040381B1, true);
    devmemRW32(0xF70A0000, 0x04000000, true);
    devmemRW32(0xF70A0004, 0x000381B1, true);
    devmemRW32(0xF70A000C, 0x12341234, true);

    size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
    XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
    XAieGbl_HwInit(&AieConfig);
    AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
    XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

    ACDC_clear_tile_memory(TileInst[1][2]);
    ACDC_clear_tile_memory(TileInst[1][3]);

    mlir_configure_cores();
    mlir_configure_switchboxes();
    mlir_configure_dmas();
    mlir_initialize_locks();

    ACDC_print_tile_status(TileInst[1][2]);
    ACDC_print_tile_status(TileInst[1][3]);

    printf("\nStarting cores.\n");

    mlir_start_cores();

    int errors = 0;


    printf("Waiting to acquire output lock for read ...\n");
    if(!XAieTile_LockAcquire(&(TileInst[1][3]), 0, 1, LOCK_TIMEOUT)) {
        printf("ERROR: timeout hit!\n");
    }

    for (int i=0; i < HEIGHT; i++){
        for(int j=0; j < LINE_WIDTH; j++)
            printf("%d ",mlir_read_buffer_out(i*LINE_WIDTH+j));
        printf("\n");       
    }
    
    for (int i=0; i < HEIGHT; i++){
        for(int j=0; j < LINE_WIDTH; j++)
            ACDC_check("AFTER", mlir_read_buffer_out(i*LINE_WIDTH+j), i, errors);        
    }


    if (!errors) {
        printf("PASS!\n"); return 0;
    } else {
        printf("Fail!\n"); return -1;
    }
    printf("test done.\n");
}

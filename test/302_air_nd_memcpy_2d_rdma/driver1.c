//===- driver1.c ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Got all includes from pcimem, not sure if they are all necessary
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <ctype.h>
#include <termios.h>
#include <sys/types.h>
#include <sys/mman.h>

#include "pcie-ernic.h"
#include "pcie-bdf.h"

// Controlling what we print in the application
//#define PRINT_QP_STATE
#define PRINT_DEV_STATE
//#define PRINT_SQ
//#define PRINT_CQ
//#define PRINT_BUFF

// Function Defines
#define ERNIC_ID 		0

// ERNIC Source addresses
#define IPV4_ADDR        0xf38590ba
#define MAC_ADDR_LSB     0x17dc5e9a
#define MAC_ADDR_MSB     0x00002f76

// Address Defines
#define DEST_IP          0x610c6007 
#define DEST_MAC_LSB     0x50560f2e 
#define DEST_MAC_MSB     0x000016c4 

// Queue Depths
// [15:0]   SQ and CQ Depth
// [41:16]  RQ Depth
#define QDEPTH           0x01000100

#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 16
#define IMAGE_SIZE  (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 16
#define TILE_HEIGHT 8
#define TILE_SIZE  (TILE_WIDTH * TILE_HEIGHT)

int main(int argc, char *argv[]) {

    if(argc != 2) {
        printf("USAGE: ./driver1 {mode}\n");
        printf("\t0: Hosting remote src buffer\n");
        printf("\t1: Hosting remote dst buffer\n");
        printf("\t2: Hosting remote src and dst buffer\n");
        return 1;
    }
 
    int driver_mode = atoi(argv[1]); 
    if(driver_mode > 2) {
        printf("[ERROR] No driver_mode %d supported\n", driver_mode);
        return 1;
    }
	
    /* Create the pcie_ernic_device. This will map the two BARs into the
    structure, program the BDF, and write to the global CSRs of the ERNIC. */
    struct pcie_ernic_dev *dev =
      pcie_ernic_open_dev(BAR4_DEV_FILE,      // axil_bar_filename
                          BAR4_SIZE,          // axil_bar_size
                          0x000C0000,         // axil_bar_offset
                          BAR0_DEV_FILE,      // dev_mem_bar_filename
                          BAR0_SIZE,          // dev_mem_bar_size
                          0x0000000800000000, // dev_mem_global_offset
                          0x00040000,         // dev_mem_partition_offset
                          0x00100000,         // mrmac_reset_offset
                          0x00110000,         // mac_0_csr_offset
                          0x00120000,         // mac_1_csr_offset
                          ERNIC_ID,           // func_num
                          IPV4_ADDR,          // ipv4 addr
                          MAC_ADDR_LSB,       // mac_addr_lsb
                          MAC_ADDR_MSB,       // mac_addr_msb
                          true,               // configure_cmacs
                          false,              // configure_bdf
                          true,               // is_versal
                          true);              // dual_reset


    if(dev == NULL) {
        printf("[ERROR] Failed to create pcie_ernic_dev structure\n");
        return 1;
    }

    // Allocate a pd to associate to queues and memory regions
    struct pcie_ernic_pd *pd = pcie_ernic_alloc_pd(dev, 0);

    struct pcie_ernic_qp *qp2 = pcie_ernic_create_qp(dev,           // pcie_ernic_dev
                                                    pd, 	    // pcie_ernic_pd
                                                    2, 		    // qpid
                                                    2, 		    // dest qpid
                                                    QDEPTH,         // queue depth
                                                    DEST_IP,        // dest ip
                                                    DEST_MAC_MSB,   // dest mac 
                                                    DEST_MAC_LSB,   // dest mac 
                                                    false,          // enable cq
                                                    true);          // on_device

    if(qp2 == NULL) {
        printf("[ERROR] Failed to create QP2\n");
        return 1;
    }

    if(driver_mode == 0) {
        printf("[INFO] Allocating and registering src buffer\n");    

        // Creating a local buffer that we can register
        struct pcie_ernic_buff *reg_mem = pcie_ernic_malloc(dev, sizeof(uint32_t)*IMAGE_WIDTH*IMAGE_HEIGHT /* size */, true /* on_device */);

  #ifdef PRINT_BUFF
        print_buff(reg_mem);
        printf("\n");
  #endif

        // Writing some noticable data into the buffer
        for(int i = 0; i < IMAGE_SIZE; i++) {
          *(uint32_t *)(reg_mem->buff + 4*i) = i+0x1000;
        }
       
        // Registering and advertising the buffer to the destination of the QP 
        pcie_ernic_reg_adv(dev, reg_mem, qp2, 0x00000010, 0x00200000, PD_READ_WRITE);

        // Waiting for the other process to be done
        printf("Src buffer written to and advertised.\nPress [Enter] to continue.....");
        char enter = 0;
        while(enter != '\r' && enter != '\n') {
            enter = getchar();
        }
        printf("\n");

        // Free the local buffer we allocated
        pcie_ernic_free_buff(reg_mem);
    }
    else if(driver_mode == 1) {
        printf("[INFO] Allocating and registering dst buffer\n");    

        // Creating a local buffer that we can register
        struct pcie_ernic_buff *reg_mem = pcie_ernic_malloc(dev, sizeof(uint32_t)*IMAGE_WIDTH*IMAGE_HEIGHT /* size */, true /* on_device */);

#ifdef PRINT_BUFF
        print_buff(reg_mem);
        printf("\n");
#endif

        // Writing some noticable data into the buffer
        for(int i = 0; i < IMAGE_SIZE; i++) {
          *(uint32_t *)(reg_mem->buff + 4*i) = 0x0defaced;
        }
       
        // Registering and advertising the buffer to the destination of the QP 
        pcie_ernic_reg_adv(dev, reg_mem, qp2, 0x00000010, 0x00200000, PD_READ_WRITE);

        // Waiting for the other process to be done
        printf("Destination buffer written to and advertised.\nPress [Enter] to continue.....");
        char enter = 0;
        while(enter != '\r' && enter != '\n') {
            enter = getchar();
        }
        printf("\n");

        int errors = 0;

        // Now look at the image, should have the bottom left filled in
        for (int i=0;i<IMAGE_SIZE;i++) {
          uint32_t rb = *(uint32_t *)(reg_mem->buff + 4*i);

          uint32_t row = i / IMAGE_WIDTH;
          uint32_t col = i % IMAGE_WIDTH;

          if ((row >= TILE_HEIGHT) && (col < TILE_WIDTH)) {
            if (!(rb == 0x1000+i)) {
              printf("IM %d [%d, %d] should be %08X, is %08X\n", i, col, row, i+0x1000, rb);
              errors++;
            }
          }
          else {
            if (rb != 0x00defaced) {
              printf("IM %d [%d, %d] should be 0xdefaced, is %08X\n", i, col, row, rb);
              errors++;
            }
          }
        }

        if (!errors) {
          printf("PASS!\n");
        }
        else {
          printf("fail %d/%d.\n", (TILE_SIZE+IMAGE_SIZE-errors), TILE_SIZE+IMAGE_SIZE);
        }
      
        // Free the local buffer we allocated
        pcie_ernic_free_buff(reg_mem);

    }
    else if(driver_mode == 2) {
        printf("[INFO] Allocating and registering src and dst buffer\n");    

        // Creating a local buffer that we can register
        struct pcie_ernic_buff *src_reg_mem = pcie_ernic_malloc(dev, sizeof(uint32_t)*IMAGE_WIDTH*IMAGE_HEIGHT /* size */, true /* on_device */);
        struct pcie_ernic_buff *dst_reg_mem = pcie_ernic_malloc(dev, sizeof(uint32_t)*IMAGE_WIDTH*IMAGE_HEIGHT /* size */, true /* on_device */);

#ifdef PRINT_BUFF
        print_buff(reg_mem);
        printf("\n");
#endif

        // Writing some noticable data into the buffer
        for(int i = 0; i < IMAGE_SIZE; i++) {
          *(uint32_t *)(src_reg_mem->buff + 4*i) = i+0x1000;
          *(uint32_t *)(dst_reg_mem->buff + 4*i) = 0x0defaced;
        }
       
        // Registering and advertising the buffer to the destination of the QP 
        pcie_ernic_reg_adv(dev, src_reg_mem, qp2, 0x00000010 /* key */, src_reg_mem->size/*0x00200000*/, PD_READ_WRITE);
        pcie_ernic_reg_adv(dev, dst_reg_mem, qp2, 0x00000010 /* key */, dst_reg_mem->size/*0x00200000*/, PD_READ_WRITE);

        // Waiting for the other process to be done
        printf("Destination buffer written to and advertised.\nPress [Enter] to continue.....");
        char enter = 0;
        while(enter != '\r' && enter != '\n') {
            enter = getchar();
        }
        printf("\n");

        int errors = 0;

        // Now look at the image, should have the bottom left filled in
        for (int i=0;i<IMAGE_SIZE;i++) {
          uint32_t rb = *(uint32_t *)(dst_reg_mem->buff + 4*i);

          uint32_t row = i / IMAGE_WIDTH;
          uint32_t col = i % IMAGE_WIDTH;

          if ((row >= TILE_HEIGHT) && (col < TILE_WIDTH)) {
            if (!(rb == 0x1000+i)) {
              printf("IM %d [%d, %d] should be %08X, is %08X\n", i, col, row, i+0x1000, rb);
              errors++;
            }
          }
          else {
            if (rb != 0x00defaced) {
              printf("IM %d [%d, %d] should be 0xdefaced, is %08X\n", i, col, row, rb);
              errors++;
            }
          }
        }

        if (!errors) {
          printf("PASS!\n");
        }
        else {
          printf("fail %d/%d.\n", (TILE_SIZE+IMAGE_SIZE-errors), TILE_SIZE+IMAGE_SIZE);
        }
      
        // Free the local buffer we allocated
        pcie_ernic_free_buff(src_reg_mem);
        pcie_ernic_free_buff(dst_reg_mem);

    }


#ifdef PRINT_QP_STATE
    printf("[INFO] QP2 State:\n");
    print_qp_state(qp2);
    printf("\n");
#endif

    // Read from the ERNIC status registers
#ifdef PRINT_DEV_STATE
    printf("ERNIC 0 State:\n");
    print_dev_state(dev);
    printf("\n");
#endif
    

    
    // This will free all of the resources assocaited with the device
    pcie_ernic_free_dev(dev);

    /* As the device isn't guranteed to have a reference to each PD
    free them on their own */
    free(pd);

    return 0;
}

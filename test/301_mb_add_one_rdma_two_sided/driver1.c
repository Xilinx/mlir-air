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

int main() {
	
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

    // Creating a local buffer that we can register
    struct pcie_ernic_buff *src = pcie_ernic_malloc(dev, 512 /* size */, true /* on_device */);
    struct pcie_ernic_buff *dst = pcie_ernic_malloc(dev, 512 /* size */, true /* on_device */);

#ifdef PRINT_BUFF
    print_buff(src);
    printf("\n");
#endif

    // Writing some noticable data into the buffer
    for(int i = 0; i < 256/4; i++) {
      *(uint32_t *)(src->buff + 4*i) = i;
      *(uint32_t *)(dst->buff + 4*i) = 0x0defaced;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Simulating air_send()

    printf("Posting data SEND to AIR instance\n");
    int ret_val = pcie_ernic_post_wqe(
                                dev,                                    // dev
                                qp2,                                    // pcie_ernic_qp
                                0xe0a6,                                 // wrid
                                src->pa,                                // laddr 
                                0x00000100,                             // length
                                OP_SEND,                                // op
                                0x0000000000000000,                     // offset
                                0x00000000,                             // rtag
                                0,                                      // send_data_hi
                                0,                                      // send_data_lo
                                0,                                      // immdt_data
                                true);                                  // poll on completion
    if(!ret_val) {
        printf("[ERROR] Failed to send SEND packet\n");
        return 1;
    }

    printf("Polling on synchronizing SEND\n");
    void *rqe_ptr = pcie_ernic_post_recv(dev, qp2);
    if(rqe_ptr == NULL) {
        printf("[ERROR] pcie_ernic_post_recv returned NULL\n");
        return 1;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Simulating air_recv()

    printf("Polling on data SEND\n");
    rqe_ptr = pcie_ernic_post_recv(dev, qp2);
    if(rqe_ptr == NULL) {
        printf("[ERROR] pcie_ernic_post_recv returned NULL\n");
        return 1;
    }

    // Copying from the RQE to the buffer
    for(int i = 0; i < 256/4; i++) {
      *(uint32_t *)(dst->buff + 4*i) = *(uint32_t *)(rqe_ptr + 4*i);
    }

    printf("Posting synchronizing SEND to AIR instance\n");
    ret_val = pcie_ernic_post_wqe(
                                dev,                                    // dev
                                qp2,                                    // pcie_ernic_qp
                                0xe0a6,                                 // wrid
                                src->pa,                                // laddr 
                                0x00000100,                             // length
                                OP_SEND,                                // op
                                0x0000000000000000,                     // offset
                                0x00000000,                             // rtag
                                0,                                      // send_data_hi
                                0,                                      // send_data_lo
                                0,                                      // immdt_data
                                true);                                  // poll on completion
    if(!ret_val) {
        printf("[ERROR] Failed to send SEND packet\n");
        return 1;
    }

    for(int i = 0; i < 256/4; i++) {
      printf("dst[%d] = 0x%x\n", i, *(uint32_t *)(dst->buff + 4*i));
    }
   
#ifdef PRINT_QP_STATE
    printf("[INFO] QP2 State:\n");
    print_qp_state(qp2);
    printf("\n");
#endif
#ifdef PRINT_DEV_STATE
    printf("ERNIC 0 State:\n");
    print_dev_state(dev);
    printf("\n");
#endif

    int errors = 0;
    for(int i = 0; i < 16; i++) {
      uint32_t d = *(uint32_t *)(dst->buff + 4*i);
      uint32_t s = *(uint32_t *)(src->buff + 4*i);

      printf("src[%d] = 0x%x\n", i, s);
      printf("dst[%d] = 0x%x\n", i, d);
      if(d != s + 1) {
        errors++;
        printf("mismatch %x != 1 + %x\n", d, s);
      }
    }
    
    if(!errors) {
      printf("PASS!\n");     
    }
    else {
      printf("fail %d/16\n", errors);
    }
 
    // Free the local buffer we allocated
    pcie_ernic_free_buff(src);
    pcie_ernic_free_buff(dst);
    
    // This will free all of the resources assocaited with the device
    pcie_ernic_free_dev(dev);

    /* As the device isn't guranteed to have a reference to each PD
    free them on their own */
    free(pd);

    return 0;
}

//===- driver2.c -------------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

#include "pcie-bdf.h"
#include "pcie-ernic.h"

// Controlling what we print in the application
#define PRINT_QP_STATE
#define PRINT_DEV_STATE
// #define PRINT_SQ
// #define PRINT_CQ
#define PRINT_BUFF

// Function Defines
#define ERNIC_ID 1

// ERNIC Source addresses
#define IPV4_ADDR 0x610c6007
#define MAC_ADDR_LSB 0x50560f2e
#define MAC_ADDR_MSB 0x000016c4

// ERNIC Source addresses
#define DEST_IP 0xf38590ba
#define DEST_MAC_LSB 0x17dc5e9a
#define DEST_MAC_MSB 0x00002f76

// Queue Depths
#define QP2_QDEPTH 0x01000100

int main() {

  // Creating ERNIC 1
  /* struct pcie_ernic_dev *dev = pcie_ernic_open_dev(BAR4_DEV_FILE,         //
     axil_bar_filename BAR4_SIZE,	    	    // axil_bar_size 0x00080000,
     // axil_bar_offset 0x00020000,             // device memory offset
                                                   ERNIC_ID,	            //
     func_num IPV4_ADDR,              // ipv4 addr MAC_ADDR_LSB,           //
     mac_addr_lsb MAC_ADDR_MSB,           // mac_addr_msb true, //
     configure_cmac false,                  // configure_bdf true); //
     is_versal*/

  // Creating ERNIC
  struct pcie_ernic_dev *dev =
      pcie_ernic_open_dev(BAR4_DEV_FILE,      // axil_bar_filename
                          BAR4_SIZE,          // axil_bar_size
                          0x00080000,         // axil_bar_offset
                          BAR0_DEV_FILE,      // dev_mem_bar_filename
                          BAR0_SIZE,          // dev_mem_bar_size
                          0x0000000800000000, // dev_mem_global_offset
                          0x00020000,         // dev_mem_segment_offset
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

  if (dev == NULL) {
    printf("[ERROR] Failed to create pcie_ernic_dev structure\n");
    return 1;
  }

  // Allocate a pd to associate to queues and memory regions
  struct pcie_ernic_pd *pd = pcie_ernic_alloc_pd(dev, 0);

  struct pcie_ernic_qp *qp2 = pcie_ernic_create_qp(dev,        // pcie_ernic_dev
                                                   pd,         // pcie_ernic_pd
                                                   2,          // qpid
                                                   2,          // dest qpid
                                                   QP2_QDEPTH, // queue depth
                                                   DEST_IP,    // dest ip
                                                   DEST_MAC_MSB, // dest mac
                                                   DEST_MAC_LSB, // dest mac
                                                   false,        // enable cq
                                                   true);        // on_device

  if (qp2 == NULL) {
    printf("[ERROR] Failed to create QP2\n");
    return 1;
  }

  // Creating a local buffer that we can register
  struct pcie_ernic_buff *reg_mem = pcie_ernic_malloc(dev, 4096, true);

#ifdef PRINT_BUFF
  print_buff(reg_mem);
  printf("\n");
#endif

  // Clearing the parts of the buffer we will read into
  *(uint32_t *)(reg_mem->buff) = 0;
  *(uint32_t *)(reg_mem->buff + 4) = 0;
  printf("Memory before remote read: 0x%x\n", *(uint32_t *)(reg_mem->buff));

  // Writing some registered memory to PD 0
  struct pcie_ernic_mr *mr = pcie_ernic_reg_mr(dev, pd, reg_mem, 0x00000010,
                                               0x00200000, PD_READ_WRITE);

  // Printing some information about the QP
#ifdef PRINT_QP_STATE
  printf("[INFO] QP2 State:\n");
  print_qp_state(qp2);
  printf("\n");
#endif

  printf("Polling on first SEND...\n");
  pcie_ernic_flush_cache();
  void *rqe_ptr = pcie_ernic_post_recv(dev, qp2);
  if (rqe_ptr == NULL) {
    printf("[ERROR] pcie_ernic_post_recv returned NULL\n");
    return 1;
  }

  // Copying it so it isn't in the RQ
  uint32_t ernic1_rkey = *(uint32_t *)(rqe_ptr);
  printf("rkey read from ERNIC 0: 0x%x\n", ernic1_rkey);

  printf("Polling on second SEND...\n");
  rqe_ptr = pcie_ernic_post_recv(dev, qp2);
  if (rqe_ptr == NULL) {
    printf("[ERROR] pcie_ernic_post_recv returned NULL\n");
    return 1;
  }

  // Copying it so it isn't in the RQ
  uint64_t buff_vaddr = *(uint64_t *)(rqe_ptr);
  printf("vaddr from ERNIC 0: 0x%lx\n", buff_vaddr);

  // Now, want ERNIC 1 to send a READ REQUEST to ERNIC 0 to read from
  // ernic0_reg_mem
  int ret_val =
      pcie_ernic_post_wqe(dev, qp2, 0xe0a6, reg_mem->pa, 0x00000100, OP_READ,
                          buff_vaddr, ernic1_rkey, 0, 0, 0, true);

  if (!ret_val) {
    printf("[ERROR] Failed to send READ packet\n");
    return 1;
  }

  // Reading the data in the READ RESPONSE
  pcie_ernic_flush_cache();
  printf("Memory read from remote: 0x%x\n", *(uint32_t *)(reg_mem->buff));
  printf("\tnext word: 0x%x\n", *(uint32_t *)(reg_mem->buff + 4));

  // Now ERNIC1 will write what it read to another part of the buffer in ERNIC 0
  ret_val =
      pcie_ernic_post_wqe(dev, qp2, 0xe0a6, reg_mem->pa, 0x00000100, OP_WRITE,
                          buff_vaddr + 0x00000100, ernic1_rkey, 0, 0, 0, true);

  if (!ret_val) {
    printf("[ERROR] Failed to send READ packet\n");
    return 1;
  }

  // Read from the ERNIC status registers
#ifdef PRINT_DEV_STATE
  printf("MRMAC States:\n");
  print_both_mrmac_stats(dev);
  printf("ERNIC 1 State:\n");
  print_dev_state(dev);
  printf("\n");
#endif

#ifdef PRINT_SQ
  printf("Printing WQE information:\n");
  print_wqe(qp2, 0);
  print_wqe(qp2, 1);
  printf("\n");
#endif

#ifdef PRINT_CQ
  printf("Printing CQ information:\n");
  print_cqe(qp2, 0);
  print_cqe(qp2, 1);
  printf("\n");
#endif

  // Checking if it is equal to the hex string that is in the other one
  if (*(uint32_t *)(reg_mem->buff) == 0xEDEDEDED) {
    printf("PASSED\n");
  }
  // Just have this while we fix Versal
  else if (*(uint32_t *)(reg_mem->buff + 4) == 0xEDEDED0D) {
    printf("PASSED WITH KNOWN VERSAL BUG\n");
  } else {
    printf("FAILED\n");
  }

  // Free the local buffer we allocated
  pcie_ernic_free_buff(reg_mem);

  // This will free all of the resources assocaited with the device
  pcie_ernic_free_dev(dev);

  /* As the device isn't guranteed to have a reference to each PD
  free them on their own */
  free(pd);

  return 0;
}

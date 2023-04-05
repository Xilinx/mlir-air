//===- driver1.c ------------------------------------------------*- C++ -*-===//
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
#define ERNIC_ID 0

// ERNIC Source addresses
#define IPV4_ADDR 0xf38590ba
#define MAC_ADDR_LSB 0x17dc5e9a
#define MAC_ADDR_MSB 0x00002f76

// Address Defines
#define DEST_IP 0x610c6007
#define DEST_MAC_LSB 0x50560f2e
#define DEST_MAC_MSB 0x000016c4

// Queue Depths
// [15:0]   SQ and CQ Depth
// [41:16]  RQ Depth
#define QDEPTH 0x01000100

int main() {

  // TODO: Use AIR functionality to find BARs of the PCIe device

  /* Create the pcie_ernic_device. This will map the two BARs into the
  structure, program the BDF, and write to the global CSRs of the ERNIC. */
  struct pcie_ernic_dev *dev =
      pcie_ernic_open_dev(BAR4_DEV_FILE,      // axil_bar_filename
                          BAR4_SIZE,          // axil_bar_size
                          0x000C0000,         // axil_bar_offset
                          BAR0_DEV_FILE,      // dev_mem_bar_filename
                          BAR0_SIZE,          // dev_mem_bar_size
                          0x0000000800000000, // dev_mem_global_offset
                          0x00040000,         // dev_mem_segment_offset
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

  struct pcie_ernic_qp *qp2 = pcie_ernic_create_qp(dev,     // pcie_ernic_dev
                                                   pd,      // pcie_ernic_pd
                                                   2,       // qpid
                                                   2,       // dest qpid
                                                   QDEPTH,  // queue depth
                                                   DEST_IP, // dest ip
                                                   DEST_MAC_MSB, // dest mac
                                                   DEST_MAC_LSB, // dest mac
                                                   false,        // enable cq
                                                   true);        // on_device

  if (qp2 == NULL) {
    printf("[ERROR] Failed to create QP2\n");
    return 1;
  }

  // Creating a local buffer that we can register
  struct pcie_ernic_buff *reg_mem =
      pcie_ernic_malloc(dev, 512 /* size */, true /* on_device */);

#ifdef PRINT_BUFF
  print_buff(reg_mem);
  printf("\n");
#endif

  // Writing some registered memory to PD 0
  struct pcie_ernic_mr *mr = pcie_ernic_reg_mr(dev, pd, reg_mem, 0x00000010,
                                               0x00200000, PD_READ_WRITE);

  // Writing some noticable data into the buffer
  *(uint32_t *)(reg_mem->buff) = 0xEDEDEDED;
  *(uint32_t *)(reg_mem->buff + 4) = 0xEDEDEDED;

  printf("Memory before remote write : 0x%x\n",
         *(uint32_t *)(reg_mem->buff + 0x00000100));

#ifdef PRINT_QP_STATE
  printf("[INFO] QP2 State:\n");
  print_qp_state(qp2);
  printf("\n");
#endif

  // Creating the buffer to store the key
  struct pcie_ernic_buff *rkey_buff =
      pcie_ernic_malloc(dev, 4096 /* size */, true /* on_device */);
  *(uint32_t *)(rkey_buff->buff) = 0x00000010;
  printf("rkey_buff:\n");
  print_buff(rkey_buff);

  // Creating the buffer to store the virtual address
  struct pcie_ernic_buff *vaddr_buff =
      pcie_ernic_malloc(dev, 4096 /* size */, true /* on_device */);
  *(uint64_t *)(vaddr_buff->buff) = (uint64_t)(reg_mem->buff);
  printf("vaddr_buff:\n");
  print_buff(vaddr_buff);

  // Waiting for the other process to be done
  printf("Queue created and memory registered\nPress [Enter] to continue.....");
  char enter = 0;
  while (enter != '\r' && enter != '\n') {
    enter = getchar();
  }
  printf("\n");

  // Writing a SEND with the r_key
  printf("Sending SEND Packet to ERNIC 0\n");
  int ret_val = pcie_ernic_post_wqe(dev,                // dev
                                    qp2,                // pcie_ernic_qp
                                    0xe0a6,             // wrid
                                    rkey_buff->pa,      // laddr
                                    0x00000100,         // length
                                    OP_SEND,            // op
                                    0x0000000000000000, // offset
                                    0x00000000,         // rtag
                                    0,                  // send_data_hi
                                    0,                  // send_data_lo
                                    0,                  // immdt_data
                                    true);              // poll on completion
  if (!ret_val) {
    printf("[ERROR] Failed to send SEND packet\n");
    return 1;
  }

  // Writing a SEND with the r_key
  printf("Sending SEND Packet to ERNIC 0\n");
  ret_val = pcie_ernic_post_wqe(dev,                // dev
                                qp2,                // pcie_ernic_qp
                                0xe0a6,             // wrid
                                vaddr_buff->pa,     // laddr
                                0x00000100,         // length
                                OP_SEND,            // op
                                0x0000000000000000, // offset
                                0x00000000,         // rtag
                                0,                  // send_data_hi
                                0,                  // send_data_lo
                                0,                  // immdt_data
                                true);              // poll on completion
  if (!ret_val) {
    printf("[ERROR] Failed to send SEND packet\n");
    return 1;
  }

  // Waiting for the other process to be done
  printf("Two SENDS complete\nPress [Enter] to continue.....");
  enter = 0;
  while (enter != '\r' && enter != '\n') {
    enter = getchar();
  }
  printf("\n");

  // Reading the data that was just written
  pcie_ernic_flush_cache();
  printf("Memory written to by other ERNIC : 0x%x\n",
         *(uint32_t *)(reg_mem->buff + 0x00000100));

  // Read from the ERNIC status registers
#ifdef PRINT_DEV_STATE
  printf("MRMAC Statistics:\n");
  print_both_mrmac_stats(dev);
  printf("ERNIC 0 State:\n");
  print_dev_state(dev);
  printf("\n");
#endif

  // Print information about the WQEs
#ifdef PRINT_SQ
  printf("Printing WQE information:\n");
  print_wqe(qp2, 0);
  print_wqe(qp2, 1);
  printf("\n");
#endif

  // Print information about the CQEs
#ifdef PRINT_CQ
  printf("Printing CQ information:\n");
  print_cqe(qp2, 0);
  print_cqe(qp2, 1);
  printf("\n");
#endif

  // Checking if the remote memory copy worked
  if (*(uint32_t *)(reg_mem->buff + 0x00000100) ==
      *(uint32_t *)(reg_mem->buff)) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }

  // Free the local buffer we allocated
  pcie_ernic_free_buff(rkey_buff);
  pcie_ernic_free_buff(vaddr_buff);
  pcie_ernic_free_buff(reg_mem);

  // This will free all of the resources assocaited with the device
  pcie_ernic_free_dev(dev);

  /* As the device isn't guranteed to have a reference to each PD
  free them on their own */
  free(pd);

  return 0;
}

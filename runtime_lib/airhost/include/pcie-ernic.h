//===- pcie-ernic.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

#ifndef PCIE_ERNIC_H
#define PCIE_ERNIC_H

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

// Local includes
//#include "pcie-bdf.h"
#include "pcie-ernic-dev-mem-allocator.h"

// Control what debug information is
// printed when using the print_* and
// init functions
#define DEBUG_MEMORY
#define DEBUG_NETWORK
//#define DEBUG_DB
#define DEBUG_PCIE

// If this is defined, we will print in the functions
// corresponding to the DEFINE_* which are defined above
#define VERBOSE_DEBUG

// General page size definitions
#define PAGE_SHIFT 12      // 4KB
#define HUGE_PAGE_SHIFT 21 // 2MB
#define PAGEMAP_LENGTH 8   // Number of bytes in a pagemap entry

// QDMA host memory definitions
#define BDF_LOW_ADDR 0x00082420
#define BDF_HIGH_ADDR 0x00082424
#define BDF_PASID 0x00082428
#define BDF_FUNC_NUM 0x0008242C
#define BDF_MISC 0x00082430
#define BDF_RESERVED 0x00082434
#define USP_BDF_MISC_WRITE_DATA                                                \
  0xC4000000 // Creating a 256GB window for the S_BRIDGE, this is the value used
             // for US+ devices
#define VERSAL_BDF_MISC_WRITE_DATA                                             \
  0xC1000000 // Creating a 256GB window for the S_BRIDGE, this is the value used
             // for Versal devices

// Parameters that control how we flush the cache
#define FLUSH_CACHE_BUFF_SIZE                                                  \
  33554432 // The size of the buffer we write to to clear the cache

// General ERNIC Definitions
#define NUM_QPS 16

// Error Buffer
#define ERROR_BUFF_BASE 0x00110000
#define ERROR_BUFF_SIZE 0x01000040

// Retry buffer - This is located on the
// device in BRAM, so we can't change these
// parameters without also changing the
// hardware
#define RETRY_BUFF_BASE 0x00000000
#define RETRY_BUFF_SIZE 0x10000010

// Addressing
#define SRC_MAC_LSB 0x1C69B8ED
#define SRC_MAC_MSB 0x0000EA1E
#define UDP_SRC_PORT 0xE348
#define SRC_IP 0xD7AC977E

// ERNIC Global CSR Definitions
#define XRNICCONF 0x00020000
#define XRNICADCONF 0x00020004
#define MACXADDLSB 0x00020010
#define MACXADDMSB 0x00020014
#define IPv6XADD1 0x00020020
#define IPv6XADD2 0x00020024
#define IPv6XADD3 0x00020028
#define IPv6XADD4 0x0002002C
#define IPv4XADD 0x00020070
#define ERRBUFBA 0x00020060
#define ERRBUFBAMSB 0x00020064
#define ERRBUFSZ 0x00020068
#define IPKTERRQBA 0x00020088
#define IPKTERRQBAMSB 0x0002008C
#define IPKTERRQSZ 0x00020090
#define DATBUFBA 0x000200A0
#define DATABUFBAMSB 0x000200A4
#define DATABUFSZ 0x000200A8
#define INSRRPKTCNT 0x00020100
#define INAMPKTCNT 0x00020104
#define OUTIOPKTCNT 0x00020108
#define INTEN 0x00020180
#define OUTAMPKTCNT 0x0002010C
#define OUTNAKPKTCNT 0x00020138
#define OUTRDRSPPKTCNT 0x0002017C
#define RESPERRPKTBA 0x000200B0
#define RESPERRPKTBAMSB 0x000200B4
#define RESPERRSZ 0x000200B8
#define STATCURSQPTRi 0x0002028C
#define STATMSN 0x00020284

// MACRO to generate correct address into PD table
#define ERNIC_PD_ADDR(pd, addr) ((addr + 0x0100 * pd) >> 2)

// ERNIC PD Table Definitions
#define PDPDNUM 0x00000000
#define VIRTADDRLSB 0x00000004
#define VIRTADDRMSB 0x00000008
#define BUFBASEADDRLSB 0x0000000C
#define BUFBASEADDRMSB 0x00000010
#define BUFRKEY 0x00000014
#define WRRDBUFLEN 0x00000018
#define ACCESSDESC 0x0000001C

// MACRO to generate correct address into per-queue CSRs
#define ERNIC_QP_ADDR(qpid, addr) ((addr + 0x0100 * (qpid - 1)) >> 2)

// ERNIC Per-Queue CSR Definitions
#define QPCONFi 0x00020200
#define QPADVCONFi 0x00020204
#define RQBAi 0x00020208
#define RQBAMSBi 0x000202C0
#define SQBAi 0x00020210
#define SQBAMSBi 0x000202C8
#define CQBAi 0x00020218
#define CQBAMSBi 0x000202D0
#define RQWPTRDBADDi 0x00020220
#define RQWPTRDBADDMSBi 0x00020224
#define CQDBADDi 0x00020228
#define CQDBADDMSBi 0x0002022C
#define CQHEADi 0x00020230
#define RQCIi 0x00020234
#define SQPIi 0x00020238
#define QDEPTHi 0x0002023C
#define SQPSNi 0x00020240
#define LSTRQREQi 0x00020244
#define DESTQPCONFi 0x00020248
#define MACDESADDLSBi 0x00020250
#define MACDESADDMSBi 0x00020254
#define IPDESADDR1i 0x00020260
#define IPDESADDR2i 0x00020264
#define IPDESADDR3i 0x00020268
#define IPDESADDR4i 0x0002026C
#define STATMSNi 0x00020284
#define STATQPi 0x00020288
#define STATRQPIDBi 0x0002029C
#define PDi 0x000202B0
#define TIMEOUTCONFi 0x0002024C

// SHIM Definitions
#define RQ_PIDB_ADDR 0x00030000
#define RQ_CIDB_ADDR 0x00010000
#define SQ_CIDB_ADDR 0x00020000
#define SQ_PIDB_ADDR 0x00000000

// WQE Op codes
#define OP_WRITE 0
#define OP_WRITE_IMMDT 1
#define OP_SEND 2
#define OP_SEND_IMMDT 3
#define OP_READ 4
#define OP_SEND_INV 5

#define RQE_SIZE 256 // Size in Bytes

////MR-MAC AXI_lite register offsets

// Defines the number of times we check RX_ALIGN until
// we just say the cable is not connected
#define MRMAC_RX_ALIGN_TIMEOUT 20

//// Port 0

#define CONFIGURATION_REVISION_REG_OFFSET 0x00000000
////MR-MAC AXI_lite register offsets
//// Port 0
#define RESET_REG_0_OFFSET 0x00000004
#define MODE_REG_0_OFFSET 0x00000008
#define CONFIGURATION_TX_REG1_0_OFFSET 0x0000000C
#define CONFIGURATION_RX_REG1_0_OFFSET 0x00000010
#define TICK_REG_0_OFFSET 0x0000002C
#define FEC_CONFIGURATION_REG1_0_OFFSET 0x000000D0
#define STAT_RX_STATUS_REG1_0_OFFSET 0x00000744
#define STAT_RX_RT_STATUS_REG1_0_OFFSET 0x0000074C
#define STAT_STATISTICS_READY_0_OFFSET 0x000007D8
#define STAT_TX_TOTAL_PACKETS_0_LSB_OFFSET 0x00000818
#define STAT_TX_TOTAL_PACKETS_0_MSB_OFFSET 0x0000081C
#define STAT_TX_TOTAL_GOOD_PACKETS_0_LSB_OFFSET 0x00000820
#define STAT_TX_TOTAL_GOOD_PACKETS_0_MSB_OFFSET 0x00000824
#define STAT_TX_TOTAL_BYTES_0_LSB_OFFSET 0x00000828
#define STAT_TX_TOTAL_BYTES_0_MSB_OFFSET 0x0000082C
#define STAT_TX_TOTAL_GOOD_BYTES_0_LSB_OFFSET 0x00000830
#define STAT_TX_TOTAL_GOOD_BYTES_0_MSB_OFFSET 0x00000834
#define STAT_RX_TOTAL_PACKETS_0_LSB_OFFSET 0x00000E30
#define STAT_RX_TOTAL_PACKETS_0_MSB_OFFSET 0x00000E34
#define STAT_RX_TOTAL_GOOD_PACKETS_0_LSB_OFFSET 0x00000E38
#define STAT_RX_TOTAL_GOOD_PACKETS_0_MSB_OFFSET 0x00000E3C
#define STAT_RX_TOTAL_BYTES_0_LSB_OFFSET 0x00000E40
#define STAT_RX_TOTAL_BYTES_0_MSB_OFFSET 0x00000E44
#define STAT_RX_TOTAL_GOOD_BYTES_0_LSB_OFFSET 0x00000E48
#define STAT_RX_TOTAL_GOOD_BYTES_0_MSB_OFFSET 0x00000E4C
#define FEC_CONFIGURATION_REG1_1_OFFSET 0x000010D0
#define FEC_CONFIGURATION_REG1_2_OFFSET 0x000020D0
#define FEC_CONFIGURATION_REG1_3_OFFSET 0x000030D0

enum pd_access_flags { PD_READ_ONLY, PD_WRITE_ONLY, PD_READ_WRITE };

// Have a cyclic relationship so have to define it above
struct pcie_ernic_pd;

struct pcie_ernic_mr {
  struct pcie_ernic_pd *pd;
  struct pcie_ernic_buff *buff;
  uint8_t key;
  uint64_t length;
  enum pd_access_flags flags;
};

struct pcie_ernic_pd {
  struct pcie_ernic_qp *qp;
  struct pcie_ernic_mr *mr;
  uint32_t pd_num;
};

/* This contains a 32 bit CQE in
the format that the ERNIC will output */
struct pcie_ernic_cqe {
  uint16_t wrid;
  uint8_t op;
  uint8_t err_flags;
};

/* This contains a 512 bit WQE in the format
that the ERNIC accepts. One annoying thing is that
All reserved bits should be set to zero. In the case there
there is no immdt data, that is treated as immediate and
should also be set to 0. */
struct pcie_ernic_wqe {
  uint32_t wrid;
  uint32_t laddr_lo;
  uint32_t laddr_hi;
  uint32_t length;
  uint32_t op;
  uint32_t offset_lo;
  uint32_t offset_hi;
  uint32_t rtag;
  uint32_t send_data_dw_0;
  uint32_t send_data_dw_1;
  uint32_t send_data_dw_2;
  uint32_t send_data_dw_3;
  uint32_t immdt_data;
  uint32_t reserved_1;
  uint32_t reserved_2;
  uint32_t reserved_3;
};

struct pcie_ernic_qp {
  struct pcie_ernic_buff *sq;
  uint32_t sq_pidb;
  uint32_t sq_cidb;
  struct pcie_ernic_buff *cq;
  uint32_t cq_pidb;
  uint32_t cq_cidb;
  struct pcie_ernic_buff *rq;
  uint32_t rq_pidb;
  uint32_t rq_cidb;
  struct pcie_ernic_pd *pd;
  struct pcie_ernic_buff *dev_written_dbs; // Doorbells written by the device
  uint32_t qpid;
  uint32_t dest_qpid;
  uint32_t dest_ip;
  uint32_t dest_mac_msb;
  uint32_t dest_mac_lsb;
  uint32_t qdepth;
  bool enable_cq; // If we enabled cqe writes
};

/* Everytime we allocate memory we want to note what the
PA is,  because we don't want to have to look it up
everytime. This shouldn't be necessary for queues as
we will just program the PA once at the beginning, but
for buffers that could be referenced a lot it is better
to do this*/
struct pcie_ernic_buff {
  void *buff;
  uint64_t pa;
  uint64_t size;
  bool on_device;
};

/* This contains the address mappings of the MMIO
of one particular function as well as an array of all
the qps associated with this ERNIC. */
struct pcie_ernic_dev {
  struct pcie_ernic_qp **qps;
  struct pcie_ernic_dev_mem_allocator *allocator;
  uint32_t *axil_bar;
  uint32_t axil_bar_size;
  uint64_t *axib_bar;
  uint64_t mac_0_csr_offset;
  uint64_t mac_1_csr_offset;
  uint32_t ernic_id;
  uint32_t axil_bar_offset;
  struct pcie_ernic_buff *err_buff;
  struct pcie_ernic_buff *inc_packet_err_q;
  struct pcie_ernic_buff *resp_err_buff;
};

// Function declarations
void pcie_ernic_flush_cache();
void do_configure_cmac(struct pcie_ernic_dev *dev, uint32_t offset);
void print_both_mrmac_stats(struct pcie_ernic_dev *dev);
int read_db_axil(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp,
                 uint32_t db_addr);
int read_rq_pidb_db(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp,
                    bool poll);
int read_sq_cidb_db(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp,
                    bool poll);
void write_db_axil(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp,
                   uint32_t db_val, uint32_t db_addr);
void write_rq_cidb_db(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp,
                      uint32_t db_val);
void write_sq_pidb_db(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp,
                      uint32_t db_val);
void *pcie_ernic_post_recv(struct pcie_ernic_dev *dev,
                           struct pcie_ernic_qp *qp);
void pcie_ernic_init_bdf(struct pcie_ernic_dev *dev, bool is_versal);
unsigned long get_page_frame_number_of_address(void *addr);
uint64_t get_buff_pa(void *buff);
struct pcie_ernic_buff *pcie_ernic_malloc(struct pcie_ernic_dev *dev,
                                          uint32_t size, bool on_device);
void pcie_ernic_free_buff(struct pcie_ernic_buff *buff);
struct pcie_ernic_dev *pcie_ernic_open_dev(
    const char *axil_bar_filename, uint32_t axil_bar_size, uint32_t axil_bar_offset,
    const char *dev_mem_bar_filename, uint32_t dev_mem_bar_size,
    uint64_t dev_mem_global_offset, uint64_t dev_mem_partition_offset,
    uint64_t mrmac_reset_offset, uint64_t mrmac_0_csr_offset,
    uint64_t mrmac_1_csr_offset, uint32_t ernic_id, uint32_t ipv4_addr,
    uint32_t mac_addr_lsb, uint32_t mac_addr_msb, bool configure_cmac,
    bool configure_bdf, bool is_versal, bool dual_reset);
struct pcie_ernic_pd *pcie_ernic_alloc_pd(struct pcie_ernic_dev *dev,
                                          uint32_t pd_num);
struct pcie_ernic_qp *
pcie_ernic_create_qp(struct pcie_ernic_dev *dev, struct pcie_ernic_pd *pd,
                     uint32_t qpid, uint32_t dest_qpid, uint32_t qdepth,
                     uint32_t dest_ip, uint32_t dest_mac_msb,
                     uint32_t dest_mac_lsb, bool enable_cq, bool on_device);
void print_buff(struct pcie_ernic_buff *buff);
void print_qp_state(struct pcie_ernic_qp *qp);
void print_dev_state(struct pcie_ernic_dev *dev);
void print_op(int op);
void print_wqe(struct pcie_ernic_qp *qp, uint32_t index);
void print_cqe(struct pcie_ernic_qp *qp, uint32_t index);
struct pcie_ernic_mr *pcie_ernic_reg_mr(struct pcie_ernic_dev *dev,
                                        struct pcie_ernic_pd *pd,
                                        struct pcie_ernic_buff *buff,
                                        uint8_t key, uint32_t length,
                                        enum pd_access_flags flags);
int write_wqe_to_sq(struct pcie_ernic_buff *sq, uint32_t index, uint32_t wrid,
                    uint64_t laddr, uint32_t length, uint32_t op,
                    uint64_t offset, uint32_t rtag, uint64_t send_data_hi,
                    uint64_t send_data_lo, uint32_t immdt_data);
int pcie_ernic_post_wqe(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp,
                        uint32_t wrid, uint64_t laddr, uint32_t length,
                        uint32_t op, uint64_t offset, uint32_t rtag,
                        uint64_t send_data_hi, uint64_t send_data_lo,
                        uint32_t immdt_data, bool poll);
void pcie_ernic_free_qp(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp);
void pcie_ernic_free_dev(struct pcie_ernic_dev *dev);
void pcie_ernic_reg_adv(struct pcie_ernic_dev *dev,
                        struct pcie_ernic_buff *buff, struct pcie_ernic_qp *qp,
                        uint8_t key, uint32_t length,
                        enum pd_access_flags flags);
void pcie_ernic_recv_buff(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp,
                          uint32_t *rkey, uint64_t *vaddr);

#endif

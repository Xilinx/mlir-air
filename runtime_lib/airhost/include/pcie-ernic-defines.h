//===- pcie-ernic-defines.h ---------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// This is used by the ARM to be able to interact with the ERNIC

#ifndef PCIE_ERNIC_DEFINES_H
#define PCIE_ERNIC_DEFINSS_H

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

enum pd_access_flags { PD_READ_ONLY, PD_WRITE_ONLY, PD_READ_WRITE };

/* This contains a 512 bit WQE in the format
 * that the ERNIC accepts. One annoying thing is that
 * All reserved bits should be set to zero. In the case there
 * there is no immdt data, that is treated as immediate and
 * should also be set to 0. */
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

/* This contains a 32 bit CQE in
 * the format that the ERNIC will output */
struct pcie_ernic_cqe {
  uint16_t wrid;
  uint8_t op;
  uint8_t err_flags;
};

#endif

//===- pcie_ernic.cpp -------------------------------------------*- C++ -*-===//
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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <ctype.h>
#include <termios.h>
#include <sys/types.h>
#include <sys/mman.h>

#include "include/pcie-ernic.h"

/* This function clears the caches just by writing to a really large
array. */
void pcie_ernic_flush_cache() {
    char *cache_flush_array = (char *)malloc(FLUSH_CACHE_BUFF_SIZE);
    int i = 0;
    
    for(i = 0; i < FLUSH_CACHE_BUFF_SIZE; i++) {
        cache_flush_array[i] = 0xED; // TODO: Make sure it is not optimizing this out
    }
	free(cache_flush_array);
}

void do_configure_cmac(struct pcie_ernic_dev *dev, uint32_t offset) {

#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
    printf("[INFO] Configuring CMAC\n");
    printf("\tConfiguring CMAC at offset %x\n", offset);
#endif

    if(dev == NULL) {
        printf("[ERROR] do_configure_cmac passed NULL dev\n");
        return;
    }

    // Enabling RX
    dev->axil_bar[(offset + 0x14) >> 2] = 0x00000001; 

    // Set ctl_tx_send_lfi and ctl_tx_send_rfi
    dev->axil_bar[(offset + 0x0C) >> 2] = 0x00000018;
    
    // Poll on RX Aligned
    uint32_t rx_aligned_reg = dev->axil_bar[(offset + 0x204) >> 2];
#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
    printf("\tRX aligned buffer: %x\n", rx_aligned_reg);
#endif
    while(!(rx_aligned_reg & 0x00000001) || !(rx_aligned_reg & 0x00000002)) {
#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
        printf("\tRead %x from RX_ALIGNED with offset %x\n", rx_aligned_reg, offset);
#endif
        rx_aligned_reg = dev->axil_bar[(offset + 0x204) >> 2];
        sleep(1);
    }

    // Enable TX
    dev->axil_bar[(offset + 0x0C) >> 2] = 0x00000001;

    // Flow control
    dev->axil_bar[(offset + 0x84) >> 2] = 0x00003DFF;
    dev->axil_bar[(offset + 0x88) >> 2] = 0x0001C631;
    dev->axil_bar[(offset + 0x48) >> 2] = 0xFFFFFFFF;
    dev->axil_bar[(offset + 0x4C) >> 2] = 0xFFFFFFFF;
    dev->axil_bar[(offset + 0x50) >> 2] = 0xFFFFFFFF;
    dev->axil_bar[(offset + 0x54) >> 2] = 0xFFFFFFFF;
    dev->axil_bar[(offset + 0x58) >> 2] = 0x0000FFFF;
    dev->axil_bar[(offset + 0x34) >> 2] = 0xFFFFFFFF;
    dev->axil_bar[(offset + 0x38) >> 2] = 0xFFFFFFFF;
    dev->axil_bar[(offset + 0x3C) >> 2] = 0xFFFFFFFF;
    dev->axil_bar[(offset + 0x40) >> 2] = 0xFFFFFFFF;
    dev->axil_bar[(offset + 0x44) >> 2] = 0x0000FFFF;
    dev->axil_bar[(offset + 0x30) >> 2] = 0x000001FF;


}

void do_configure_mrmac(struct pcie_ernic_dev *dev, uint32_t offset, uint64_t mrmac_reset_offset) {


#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
    printf("[INFO] Configuring MR-MAC\n");
    printf("\tConfiguring MR-MAC at offset %x\n", offset);
    printf("[INFO] MR-MAC reset control GPIO @ \n");
    printf("INFO : READ MRMAC CORE VERSION ..........\n");
#endif
    uint32_t mrmac_core_ver_reg = dev->axil_bar[(offset + CONFIGURATION_REVISION_REG_OFFSET) >> 2];
#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
    printf(" MR-MAC Core_Version  =  %x  \n", mrmac_core_ver_reg);
    printf("INFO : START MRMAC CONFIGURATION ..........\n");
#endif
    dev->axil_bar[(offset + RESET_REG_0_OFFSET) >> 2] = 0x00000FFF;
    dev->axil_bar[(offset + MODE_REG_0_OFFSET) >> 2] = 0x40000A64;
    dev->axil_bar[(offset + CONFIGURATION_RX_REG1_0_OFFSET) >> 2] = 0x00000033;
    dev->axil_bar[(offset + CONFIGURATION_TX_REG1_0_OFFSET) >> 2] = 0x00000C03;
    dev->axil_bar[(offset + FEC_CONFIGURATION_REG1_0_OFFSET) >> 2] = 0x00000000;
    dev->axil_bar[(offset + FEC_CONFIGURATION_REG1_1_OFFSET) >> 2] = 0x00000000;
    dev->axil_bar[(offset + FEC_CONFIGURATION_REG1_2_OFFSET) >> 2] = 0x00000000;
    dev->axil_bar[(offset + FEC_CONFIGURATION_REG1_3_OFFSET) >> 2] = 0x00000000;
    dev->axil_bar[(offset + RESET_REG_0_OFFSET) >> 2] = 0x00000000;
    dev->axil_bar[(offset + TICK_REG_0_OFFSET) >> 2] = 0x00000001;

#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
    printf(" *** check for stat_mst_reset_done = 1 *** \n");
#endif
    // Poll on stat_mst_reset_done = 0x000000FF; 
    uint32_t stat_mst_reset_done_reg  = dev->axil_bar[(mrmac_reset_offset + 0x8) >> 2];
    while(stat_mst_reset_done_reg != 0x000000FF)  {
        stat_mst_reset_done_reg = dev->axil_bar[(mrmac_reset_offset + 0x8) >> 2];
        sleep(1);
    }
#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
    printf("INFO :Polling on RX ALIGNED ..........\n");
#endif
    uint32_t rx_align_num_attempts = 0;
    // first write STAT_RX_STATUS_REG1_0_OFFSET  1s to clear latches and read status
    dev->axil_bar[(offset + STAT_RX_STATUS_REG1_0_OFFSET) >> 2] = 0xFFFFFFFF;
    uint32_t rx_status_reg1_0_reg = dev->axil_bar[(offset + STAT_RX_STATUS_REG1_0_OFFSET) >> 2];
    while(!(rx_status_reg1_0_reg & 0x00000001) || !(rx_status_reg1_0_reg & 0x00000002) || !(rx_status_reg1_0_reg & 0x00000004)){
        if(rx_align_num_attempts >= MRMAC_RX_ALIGN_TIMEOUT) {
            printf("[WARNING] Timout on RX align of MRMAC at offset 0x%x. Assuming not connected and returning\n", offset);
            return;
        }
        else if(rx_status_reg1_0_reg & 0x00020000) {
            printf("[WARNING] RX synced error of MRMAC at offset 0x%x. Assuming not connected and reutrning\n", offset);
            return;
        }
        rx_align_num_attempts++;
#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
        printf("[INFO] Polling on MRMAC RX align. RX align status reg  =  %x \n", rx_status_reg1_0_reg);
#endif
        dev->axil_bar[(offset + STAT_RX_STATUS_REG1_0_OFFSET) >> 2] = 0xFFFFFFFF;
        rx_status_reg1_0_reg = dev->axil_bar[(offset + STAT_RX_STATUS_REG1_0_OFFSET) >> 2];
        sleep(1);
    }
#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
    printf("[INFO] RX align status reg  =  %x \n", rx_status_reg1_0_reg);
#endif
    printf("[INFO] MRMAC at offset %x is aligned!\n", offset);

}

void print_single_mrmac_stats(struct pcie_ernic_dev* dev, uint32_t offset) {

    uint32_t tx_total_pkt_0_MSB, tx_total_pkt_0_LSB, tx_total_bytes_0_MSB, tx_total_bytes_0_LSB, tx_total_good_pkts_0_MSB, tx_total_good_pkts_0_LSB, tx_total_good_bytes_0_MSB, tx_total_good_bytes_0_LSB;
    uint32_t rx_total_pkt_0_MSB, rx_total_pkt_0_LSB, rx_total_bytes_0_MSB, rx_total_bytes_0_LSB, rx_total_good_pkts_0_MSB, rx_total_good_pkts_0_LSB, rx_total_good_bytes_0_MSB, rx_total_good_bytes_0_LSB;
    uint64_t tx_total_pkt_0, tx_total_bytes_0, tx_total_good_bytes_0, tx_total_good_pkts_0, rx_total_pkt_0, rx_total_bytes_0, rx_total_good_bytes_0, rx_total_good_pkts_0;

    // Tick register is used to snapshot the statistics.
    // TODO: Figure out if I need to reset this to 0
    dev->axil_bar[(offset + TICK_REG_0_OFFSET) >> 2] = 0x00000001; 
    sleep(5);

    tx_total_pkt_0_MSB        = dev->axil_bar[(offset + STAT_TX_TOTAL_PACKETS_0_MSB_OFFSET) >> 2];
    tx_total_pkt_0_LSB        = dev->axil_bar[(offset + STAT_TX_TOTAL_PACKETS_0_LSB_OFFSET) >> 2];
    tx_total_good_pkts_0_MSB  = dev->axil_bar[(offset + STAT_TX_TOTAL_GOOD_PACKETS_0_MSB_OFFSET) >> 2];
    tx_total_good_pkts_0_LSB  = dev->axil_bar[(offset + STAT_TX_TOTAL_GOOD_PACKETS_0_LSB_OFFSET) >> 2];
    tx_total_bytes_0_MSB      = dev->axil_bar[(offset + STAT_TX_TOTAL_BYTES_0_MSB_OFFSET) >> 2];
    tx_total_bytes_0_LSB      = dev->axil_bar[(offset + STAT_TX_TOTAL_BYTES_0_LSB_OFFSET) >> 2];
    tx_total_good_bytes_0_LSB = dev->axil_bar[(offset + STAT_TX_TOTAL_GOOD_BYTES_0_LSB_OFFSET) >> 2];
    tx_total_good_bytes_0_MSB = dev->axil_bar[(offset + STAT_TX_TOTAL_GOOD_BYTES_0_MSB_OFFSET) >> 2];

    rx_total_pkt_0_MSB        = dev->axil_bar[(offset + STAT_RX_TOTAL_PACKETS_0_MSB_OFFSET) >> 2];      
    rx_total_pkt_0_LSB        = dev->axil_bar[(offset + STAT_RX_TOTAL_PACKETS_0_LSB_OFFSET) >> 2];      
    rx_total_good_pkts_0_MSB  = dev->axil_bar[(offset + STAT_RX_TOTAL_GOOD_PACKETS_0_MSB_OFFSET) >> 2]; 
    rx_total_good_pkts_0_LSB  = dev->axil_bar[(offset + STAT_RX_TOTAL_GOOD_PACKETS_0_LSB_OFFSET) >> 2]; 
    rx_total_bytes_0_MSB      = dev->axil_bar[(offset + STAT_RX_TOTAL_BYTES_0_MSB_OFFSET) >> 2];        
    rx_total_bytes_0_LSB      = dev->axil_bar[(offset + STAT_RX_TOTAL_BYTES_0_LSB_OFFSET) >> 2];        
    rx_total_good_bytes_0_LSB = dev->axil_bar[(offset + STAT_RX_TOTAL_GOOD_BYTES_0_LSB_OFFSET) >> 2];   
    rx_total_good_bytes_0_MSB = dev->axil_bar[(offset + STAT_RX_TOTAL_GOOD_BYTES_0_MSB_OFFSET) >> 2];   
      
    printf( "\n\rMRMAC at Offset 0x%x Statistics           \n\r\n\r", offset );
        
    tx_total_pkt_0 = (uint64_t) tx_total_pkt_0_MSB << 32 | tx_total_pkt_0_LSB;
    tx_total_bytes_0 = (uint64_t) tx_total_bytes_0_MSB << 32 | tx_total_bytes_0_LSB;
    tx_total_good_pkts_0 = (uint64_t) tx_total_good_pkts_0_MSB << 32 | tx_total_good_pkts_0_LSB;
    rx_total_pkt_0 = (uint64_t) rx_total_pkt_0_MSB << 32 | rx_total_pkt_0_LSB;
    rx_total_bytes_0 = (uint64_t) rx_total_bytes_0_MSB << 32 | rx_total_bytes_0_LSB;
    rx_total_good_pkts_0 = (uint64_t) rx_total_good_pkts_0_MSB << 32 | rx_total_good_pkts_0_LSB;
    tx_total_good_bytes_0 =(uint64_t) tx_total_good_bytes_0_MSB << 32 | tx_total_good_bytes_0_LSB;
    rx_total_good_bytes_0 =(uint64_t) rx_total_good_bytes_0_MSB << 32 | rx_total_good_bytes_0_LSB;
     
    printf("  STAT_TX_TOTAL_PACKETS           = %ld,     \t STAT_RX_TOTAL_PACKETS           = %ld\n\r\n\r", tx_total_pkt_0,rx_total_pkt_0);              
    printf("  STAT_TX_TOTAL_GOOD_PACKETS      = %ld,     \t STAT_RX_TOTAL_GOOD_PACKETS      = %ld\n\r\n\r", tx_total_good_pkts_0,rx_total_good_pkts_0);   
    printf("  STAT_TX_TOTAL_BYTES             = %ld,     \t STAT_RX_BYTES                   = %ld\n\r\n\r", tx_total_bytes_0,rx_total_bytes_0);           
    printf("  STAT_TX_TOTAL_GOOD_BYTES        = %ld,     \t STAT_RX_TOTAL_GOOD_BYTES        = %ld\n\r\n\r", tx_total_good_bytes_0,rx_total_good_bytes_0);

}

void print_both_mrmac_stats(struct pcie_ernic_dev* dev) {

    // Printing the statistics from both MRMACs
    print_single_mrmac_stats(dev, dev->mac_0_csr_offset);
    print_single_mrmac_stats(dev, dev->mac_1_csr_offset); 
    
}

// This is used to implement both read_sq_cidb_db and read_rq_pidb_db as they are the same 
// just with different addresses
int read_db_axil(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp, uint32_t db_addr) {
    
    // Checking if all passed objects are legit
    if(dev == NULL) {
        printf("[ERROR] read_rq_pidb_db given NULL dev\n");
        return -1;
    }

    if(qp == NULL) {
        printf("[ERROR] read_rq_pidb_db given NULL qp\n");
        return -1;
    }

    // Reading from the DB
    uint32_t db_count = dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, db_addr + dev->axil_bar_offset)]; 

    return (int)db_count;
}

// Reads the most recent RQ pidb from the shim
int read_rq_pidb_db(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp, bool poll) {
    int db_count = read_db_axil(dev, qp, STATRQPIDBi);

    // If poll, read until greater than what we previously have read
#if defined(VERBOSE_DEBUG) && defined(DEBUG_DB)
    printf("Polling on RQ PIDB. Count: 0x%x\n", db_count);
#endif
    while(poll && db_count == qp->rq_pidb) {
        db_count = read_db_axil(dev, qp, STATRQPIDBi);

#if defined(VERBOSE_DEBUG) && defined(DEBUG_DB)
        printf("Read %d from RQ DB. Polling on it to reach %d\n", db_count, qp->rq_pidb);
        sleep(5); // If printing in this loop, need to make sure it doesn't get too crazy
#endif

    }

    /* our local rq_pidb will only increase by one if we poll, as it is assumed
    that it is waiting for a single RQE. If no polling, than just setting it 
    equal to the actual db with the assumption that they will handle it asynchronously */
    // TODO: Not sure if this logic makes sense....too tired rn :/
    if(poll) {
        qp->rq_pidb++;        
    }
    else {
        qp->rq_pidb = db_count;
    }
    return qp->rq_pidb;
 
}

// Reads the most recent CQ cidb from the shim
int read_sq_cidb_db(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp, bool poll) {
    int db_count = read_db_axil(dev, qp, CQHEADi);

#if defined(VERBOSE_DEBUG) && defined(DEBUG_DB)
    printf("[INFO] Polling on SQ CIDB. Count: 0x%x\n", db_count);
#endif
    while(poll && db_count == qp->sq_cidb) {
        db_count = read_db_axil(dev, qp, CQHEADi);
    }

    /* our local sq_cidb will only increase by one if we poll, as it is assumed
    that it is waiting for a single RQE. If no polling, than just setting it 
    equal to the actual db with the assumption that they will handle it asynchronously */
    if(poll) {
        qp->sq_cidb++;        
    }
    else {
        qp->sq_cidb = db_count;
    }
    return qp->sq_cidb;
}


void write_db_axil(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp, uint32_t db_val, uint32_t db_addr) {
    
    if(dev == NULL) {
        printf("[ERROR] write_cq_pidb_db given NULL dev\n");
        return;
    }

    if(qp == NULL) {
        printf("[ERROR] write_cq_pidb_db given NULL qp\n");
        return;
    }

    dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, db_addr + dev->axil_bar_offset)] = db_val;

#if defined(VERBOSE_DEBUG) && defined(DEBUG_DB)
    printf("Wrote 0x%x to DB. db_addr: 0x%x. Real addr: 0x%x\n", db_val, db_addr, ERNIC_QP_ADDR(qp->qpid, db_addr + dev->axil_bar_offset));
#endif

    return;
}


void write_rq_cidb_db(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp, uint32_t db_val) {
    
    // Keeping note of what the cidb is at
    qp->rq_cidb = db_val;
    
    write_db_axil(dev, qp, db_val, RQCIi);
    
    return;
}

void write_sq_pidb_db(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp, uint32_t db_val) {

    // Keeping note of what the pidb is at
    qp->sq_pidb = db_val;    

    // Writing to the card
    write_db_axil(dev, qp, db_val, SQPIi); 
    
    return;
}

void *pcie_ernic_post_recv(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp) {

    if(dev == NULL) {
        printf("[ERROR] pcie_ernic_poll_cq was given NULL dev\n");
        return NULL;
    }

    if(qp == NULL) {
        printf("[ERROR] pcie_ernic_poll_cq was given NULL cq\n");
        return NULL;
    }

    void *rqe = NULL;
    
    int rq_pidb = read_rq_pidb_db(dev, qp, true);
    if(rq_pidb == -1) {
        printf("[ERROR] pcie_ernic_post_recv failed, returning NULL\n");
        return NULL;
    }

    // Have to flush the cache here so we can actually read the RQE
    pcie_ernic_flush_cache();

    // Pointing to the RQE
    if(rq_pidb == 0) {
        rqe = qp->rq->buff + (qp->qdepth - 1) * RQE_SIZE;
    }
    else {
        rqe = qp->rq->buff + (rq_pidb - 1) * RQE_SIZE;
    }

    // Writing to the cidb .. TODO: Don't do this here as we are losing owenership 
    // of the buffer. Fine for now but will mess up under a stress test
    write_rq_cidb_db(dev, qp, rq_pidb);

    return rqe;

}

/* This function writes to the BDF configuration registers
of QDMA for a particular function. Versal's BDF is different
so we need to know what kind of device we are writing to */
void pcie_ernic_init_bdf(struct pcie_ernic_dev *dev, bool is_versal) {
    
    if(dev == NULL) {
        printf("[ERROR] pcie_ernic_init_bdf was given NULL dev\n");
        return;
    }

#if defined(VERBOSE_DEBUG) && defined(DEBUG_PCIE)
    printf("[INFO] Configuring QDMA BDF. Versal: %d\n", is_versal);
#endif
    dev->axil_bar[BDF_LOW_ADDR >> 2]    = 0;
    dev->axil_bar[BDF_HIGH_ADDR >> 2]   = 0;
    dev->axil_bar[BDF_PASID >> 2]       = 0;
    dev->axil_bar[BDF_FUNC_NUM >> 2]    = 0; // Always using function 0
    if(is_versal) {
        dev->axil_bar[BDF_MISC >> 2]    = VERSAL_BDF_MISC_WRITE_DATA;
    }
    else {
        dev->axil_bar[BDF_MISC >> 2]    = USP_BDF_MISC_WRITE_DATA;
    }
    dev->axil_bar[BDF_RESERVED >> 2]    = 0;	
    
    return;
}

/* Allocates memory that can be used by the pcie_ernic library. Support host memory in standalone but not
for AIR so removing that functionality */
struct pcie_ernic_buff *pcie_ernic_malloc(struct pcie_ernic_dev *dev, uint32_t size, bool on_device) {
 
    struct pcie_ernic_buff *ret_struct = NULL;

    // Allocating the struct
    ret_struct = (struct pcie_ernic_buff *)malloc(sizeof(struct pcie_ernic_buff));
    if(ret_struct == NULL) {
        printf("[ERROR] Out of memory, couldn't allocated pcie_ernic_buff\n");
        return NULL;
    }

    // Bookeeping
    ret_struct->size = size;
    ret_struct->on_device = on_device;

    if(on_device) {
        
        // If we are using device memory, need to allocate memory through the 
        if(dev == NULL) {
            printf("[ERROR] pcie_ernic_malloc requested device memory but gave NULL pcie_ernic_dev\n");
            return NULL;
        } 

        // Calling device memory allocator and getting the physical address -- Note
        // that is the physical address in the device memory map, not the physical address
        // of the host. 
        ret_struct->buff = dev_mem_alloc(dev->allocator, size, &ret_struct->pa);

    }
    else {
        printf("[ERROR] Don't currently support allocating host memory with PCIe ERNIC. Returning NULL\n");
        return NULL;
    }

    return ret_struct;
    
}

void pcie_ernic_free_buff(struct pcie_ernic_buff *buff) {
    
    if(buff == NULL) {
        printf("[ERROR] pcie_ernic_free was given a NULL buff\n");
        return;
    }
  
    // Right now we don't have a way of freeing device memory, 
    // but on the host we need to unlock it and unmap the huge
    // pages
    if(!(buff->on_device)) {
        // Freeing the associated memory
        if(munlock(buff->buff, 1 << HUGE_PAGE_SHIFT) == -1) {
            printf("[ERROR] Failed to munlock buffer\n");
            return;
        }
        munmap(buff->buff, 1 << HUGE_PAGE_SHIFT); 
    }

    // Freeing the buffer itself
    free(buff);
    buff = NULL;
    
    return;
}

/* Creates a pointer to a pcie ernic dev. This takes in
paths to the sys fs interfaces to the two BARs, as well
as the function number used. This will then map the 
two bars into userspace in the pcie_ernic_dev struct, 
configure the BDFs with the correct function number, 
and then configure the global CSRs of the ERNIC */
struct pcie_ernic_dev *pcie_ernic_open_dev(char *axil_bar_filename, 
                                        uint32_t axil_bar_size,
                                        uint32_t axil_bar_offset,
                                        char *dev_mem_bar_filename,
                                        uint32_t dev_mem_bar_size,
                                        uint64_t dev_mem_global_offset,
                                        uint64_t dev_mem_partition_offset,
                                        uint64_t mrmac_reset_offset,
                                        uint64_t mac_0_csr_offset,
                                        uint64_t mac_1_csr_offset,
                                        uint32_t ernic_id,
                                        uint32_t ipv4_addr,
                                        uint32_t mac_addr_lsb,
                                        uint32_t mac_addr_msb,
                                        bool configure_cmac,
                                        bool configure_bdf,
                                        bool is_versal,
                                        bool dual_reset) {

    int i = 0;
    struct pcie_ernic_dev *dev = NULL;
    
    // Allocating the pcie_ernic_dev
    dev = (struct pcie_ernic_dev *)malloc(sizeof(struct pcie_ernic_dev));
    dev->ernic_id = ernic_id;
    dev->qps = (struct pcie_ernic_qp **)malloc(NUM_QPS * sizeof(struct pcie_ernic_qp *));
    for(i = 0; i < NUM_QPS; i++) {
        dev->qps[i] = NULL;
    }
	
    // Allocating AXIL MMIO
    int axil_fd;
    if((axil_fd = open(axil_bar_filename, O_RDWR | O_SYNC)) == -1) {
            printf("[ERROR] Failed to open axil device file. Given: %s\n", axil_bar_filename);
            return NULL;
    }
    void *axil_base =  mmap(NULL,                 	// virtual address
                            axil_bar_size,          // length
                            PROT_READ | PROT_WRITE, // prot
                            MAP_SHARED,             // flags
                            axil_fd,                // device fd
                            0);                     // offset
    
    dev->axil_bar = (uint32_t *)axil_base;
    dev->axil_bar_size = axil_bar_size;
    dev->axil_bar_offset = axil_bar_offset;
    dev->mac_0_csr_offset = mac_0_csr_offset;
    dev->mac_1_csr_offset = mac_1_csr_offset;
#if defined(VERBOSE_DEBUG) && defined(DEBUG_PCIE)
    printf("[INFO] AXIL memory mapped into userspace\n");
    printf("\tVA: %p\n", dev->axil_bar);
    printf("\tSize: %d\n", dev->axil_bar_size);
    printf("\tOffset: %d\n", dev->axil_bar_offset);
#endif

    // Allocating the device memory allocator
    dev->allocator = init_dev_mem_allocator(dev_mem_bar_filename, dev_mem_bar_size, dev_mem_global_offset, dev_mem_partition_offset);

    // Configuring the BDF which allows the device 
    // to access host memory
    if(configure_bdf) {
        pcie_ernic_init_bdf(dev, is_versal);
    }

    // Allocate memory for the three error buffers. Storing them on the device.
    dev->err_buff = pcie_ernic_malloc(dev, 16 * 4096 /* 64 256B error buffers */, true);
    dev->inc_packet_err_q = pcie_ernic_malloc(dev, 4096, true);
    dev->resp_err_buff = pcie_ernic_malloc(dev, 4096, true);

    // Configuring ERNIC Global CSRs
    dev->axil_bar[(XRNICCONF        + axil_bar_offset)  >> 2]  = 0xe348078b;  // XRNIC Conf 
    dev->axil_bar[(INTEN            + axil_bar_offset)  >> 2]  = 0x00000070;  // Interrupt Enable
    dev->axil_bar[(ERRBUFBA         + axil_bar_offset)  >> 2]  = (uint32_t)(dev->err_buff->pa & 0x00000000FFFFFFFF);  // Error buffer base address
    dev->axil_bar[(ERRBUFBAMSB      + axil_bar_offset)  >> 2]  = (uint32_t)(dev->err_buff->pa >> 32);  // Error buffer base address
    dev->axil_bar[(ERRBUFSZ         + axil_bar_offset)  >> 2]  = 0x01000040;  // Error buffer size, this is what was used in the testbench so just keeping the same
    dev->axil_bar[(RESPERRPKTBA     + axil_bar_offset) >> 2]   = (uint32_t)(dev->resp_err_buff->pa & 0x00000000FFFFFFFF); // Response error pkt buffer base address
    dev->axil_bar[(RESPERRPKTBAMSB  + axil_bar_offset) >> 2]   = (uint32_t)(dev->resp_err_buff->pa >> 32); // Response error pkt buffer base address
    dev->axil_bar[(RESPERRSZ        + axil_bar_offset) >> 2]   = 0x00200000; // Making 4KiB (TODO: I am not sure if this is the # of entries in the buffer or the size in bytes)
    dev->axil_bar[(IPKTERRQBA       + axil_bar_offset) >> 2]   = (uint32_t)(dev->inc_packet_err_q->pa & 0x00000000FFFFFFFF); // incoming packet error status queue buffer base address
    dev->axil_bar[(IPKTERRQBAMSB    + axil_bar_offset) >> 2]   = (uint32_t)(dev->inc_packet_err_q->pa >> 32); // incoming packet error status queue buffer base address
    dev->axil_bar[(IPKTERRQSZ       + axil_bar_offset) >> 2]   = 0x00000200/*0x00200000*/; // Making 4KiB (Each entry is 8B)
    dev->axil_bar[(DATABUFSZ        + axil_bar_offset)  >> 2]  = RETRY_BUFF_SIZE;  // Data Buffer Size
    dev->axil_bar[(MACXADDMSB       + axil_bar_offset)  >> 2]  = mac_addr_msb;  // MAC MSB XRNIC
    dev->axil_bar[(IPv4XADD         + axil_bar_offset)  >> 2]  = ipv4_addr;  // IPv4 address
    dev->axil_bar[(MACXADDLSB       + axil_bar_offset)  >> 2]  = mac_addr_lsb;  // MAC LSB XRNIC
    dev->axil_bar[(DATBUFBA         + axil_bar_offset)  >> 2]  = RETRY_BUFF_BASE;  // Data Buffer Base address (Where the data of writes incase there is a retry

    // If US+ configure the CMAC
    if(configure_cmac && !is_versal) {
        do_configure_cmac(dev, 0x00090000); // TODO: Update the software to handle this dynamically as well
        do_configure_cmac(dev, 0x000A0000);
    }

    // If versal configure the MRMAC
    if(configure_cmac && is_versal) {

#ifdef VERBOSE_DEBUG
      printf("[INFO] Configuring MRMAC\n");
      printf("\tMRMAC 0 Offset: 0x%x\n", mac_0_csr_offset);
      printf("\tMRMAC 1 Offset: 0x%x\n", mac_1_csr_offset);
      printf("\tMRMAC Reset Offset: 0x%x\n", mrmac_reset_offset);
#endif


        // first double reset MRMAC xcrvrs and mrmac cores 
#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
      printf("[INFO] MRMAC Bring up process:\n");
      printf("\t*** Write M_AXI_LITE GPIO to assert gt_reset_all_in  ***\n");
      printf("\tdual_reset is set to: %d\n", dual_reset);
#endif
      dev->axil_bar[(mrmac_reset_offset) >> 2] = 0x00000000;
  
      // Currently switching between two reset structures. One has resets for both 
      // MRMACs connected to [3:0] of our AXI GPIO, while the other allows the 
      // software to individually reset MRMACs. Currently running into issues
      // with the new reset hardware, so providing the flexibility here
      if(dual_reset) {
        dev->axil_bar[(mrmac_reset_offset) >> 2] = 0x0000000F;
      }
      else {
        dev->axil_bar[(mrmac_reset_offset) >> 2] = 0x000000FF;
      }
      dev->axil_bar[(mrmac_reset_offset) >> 2] = 0x00000000;

#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
      printf("\t*** check for stat_mst_reset_done = 1 from test bench reset control*** \n");
#endif
      // Poll on stat_mst_reset_done = 0x000000FF; 
      uint32_t stat_mst_reset_done_reg = dev->axil_bar[(mrmac_reset_offset + 0x8) >> 2];
      while(stat_mst_reset_done_reg != 0x000000FF)  {
        stat_mst_reset_done_reg = dev->axil_bar[(mrmac_reset_offset + 0x8) >> 2];
        sleep(1);
      }

#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
      printf("\t*** First GT reset done***\n");
#endif

      dev->axil_bar[(mrmac_reset_offset) >> 2] = 0x00000000;

      // Need to reset the MRMACs again [3:0] as well as strobe the reset
      // for the PL components of the MRMAC which is [4]. Again, have two
      // reset options because running into issues with the single reset
      // hardware
      if(dual_reset) {
        dev->axil_bar[(mrmac_reset_offset) >> 2] = 0x0000001F; // Resetting MRMAC and PL
      }
      else {
        dev->axil_bar[(mrmac_reset_offset) >> 2] = 0x000003FF; // New reset structure
      }
      dev->axil_bar[(mrmac_reset_offset) >> 2] = 0x00000000;

#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
      printf("\t*** check for stat_mst_reset_done = 1 from test bench reset control*** \n");
#endif
      // Poll on stat_mst_reset_done = 0x0000000F; 
      stat_mst_reset_done_reg = dev->axil_bar[(mrmac_reset_offset + 0x8) >> 2];
      while(stat_mst_reset_done_reg != 0x000000FF)  {
        stat_mst_reset_done_reg = dev->axil_bar[(mrmac_reset_offset + 0x8) >> 2];
        sleep(1);
      }

      // configure and align MRMACS
      do_configure_mrmac(dev, mac_0_csr_offset, mrmac_reset_offset);
      do_configure_mrmac(dev, mac_1_csr_offset, mrmac_reset_offset);
    }

    return dev;
}






/* Creates a pcie_ernic_pd struct. This will write
the protection domain number to the PD Table of the ERNIC*/ 
struct pcie_ernic_pd *pcie_ernic_alloc_pd(struct pcie_ernic_dev *dev, uint32_t pd_num) {

    struct pcie_ernic_pd *ret_struct = NULL;

    if(dev != NULL) {

        // Allocating the structures
        ret_struct = (struct pcie_ernic_pd *)malloc(sizeof(struct pcie_ernic_pd));
        ret_struct->pd_num = pd_num;
        ret_struct->mr = NULL;

        /* Writing to the ERNIC PD Table, only writes the pd_num
        the other stuff comes when we register a memory region */
        dev->axil_bar[ERNIC_PD_ADDR(pd_num, PDPDNUM + dev->axil_bar_offset)] = pd_num;
    }
    else {
        printf("[ERROR] pcie_ernic_alloc_pd given NULL dev\n");
    }

    return ret_struct;
}

/* Creats a pcie_ernic_qp struct. This will write to all 
of the per-queue CSRs in the ERNIC. It will also allocate
the sq, cq, and rq and the corresponding doorbells */
struct pcie_ernic_qp *pcie_ernic_create_qp(struct pcie_ernic_dev *dev,
                            struct pcie_ernic_pd *pd,
                            uint32_t qpid,
                            uint32_t dest_qpid,
                            uint32_t qdepth,
                            uint32_t dest_ip,
                            uint32_t dest_mac_msb,
                            uint32_t dest_mac_lsb,
                            bool enable_cq,
                            bool on_device) {

    if(dev == NULL) {
        printf("[ERROR] pcie_ernic_create_qp given a NULL pcie_ernic_dev\n");
        return NULL;
    }

    if(qpid == 1) {
        printf("[ERROR] Users cannot create QP with ID 1. Use a larger ID\n");
        return NULL;
    }
    
    // Configuring the elements of the queue pair struct
    struct pcie_ernic_qp *qp = (struct pcie_ernic_qp *)malloc(sizeof(struct pcie_ernic_qp));
    
    // Allocate RDMA QPs
    qp->sq                      = pcie_ernic_malloc(dev, 1 << PAGE_SHIFT, on_device);
    qp->sq_pidb                 = 0;
    qp->sq_cidb                 = 0;
    qp->cq                      = pcie_ernic_malloc(dev, 1 << PAGE_SHIFT, on_device);
    qp->cq_pidb                 = 0;
    qp->cq_cidb                 = 0;
    qp->rq                      = pcie_ernic_malloc(dev, 1 << PAGE_SHIFT, on_device);
    qp->rq_pidb                 = 0;
    qp->rq_cidb                 = 0;
    qp->dev_written_dbs         = pcie_ernic_malloc(dev, 1 << PAGE_SHIFT, on_device);
    qp->pd                      = pd;
    qp->qpid                    = qpid;
    qp->dest_qpid               = dest_qpid;
    qp->dest_ip                 = dest_ip;
    qp->dest_mac_msb            = dest_mac_msb;
    qp->dest_mac_lsb            = dest_mac_lsb;
    qp->qdepth                  = qdepth;
    qp->enable_cq               = enable_cq;

    // Assign the queue to the array of qps in the pcie_ernic_dev object
    if(dev->qps[qpid] != NULL) {
        printf("[WARNING] Overwiting detected at QP %d\n", qpid);
    }
    dev->qps[qpid] = qp;
   
    // ERNIC QP Configurations
    dev->axil_bar[ERNIC_QP_ADDR(qpid, IPDESADDR1i + dev->axil_bar_offset)     ] = dest_ip;  //22// IP destination address 1
    dev->axil_bar[ERNIC_QP_ADDR(qpid, MACDESADDMSBi + dev->axil_bar_offset)   ] = dest_mac_msb;  //23// MAC Destination Address MSB
    dev->axil_bar[ERNIC_QP_ADDR(qpid, MACDESADDLSBi + dev->axil_bar_offset)   ] = dest_mac_lsb;  //24// MAC Destination address LSB
    dev->axil_bar[ERNIC_QP_ADDR(qpid, RQBAi + dev->axil_bar_offset)           ] = (uint32_t)(qp->rq->pa & 0x00000000FFFFFFFF);  //29// RQ Base Address
    dev->axil_bar[ERNIC_QP_ADDR(qpid, RQBAMSBi + dev->axil_bar_offset)        ] = (uint32_t)(qp->rq->pa >> 32);  //29// EXTRA
    dev->axil_bar[ERNIC_QP_ADDR(qpid, CQDBADDi + dev->axil_bar_offset)        ] = (uint32_t)(qp->dev_written_dbs->pa & 0x00000000FFFFFFFF);  //30// CQ DB Addres 
    dev->axil_bar[ERNIC_QP_ADDR(qpid, CQDBADDMSBi + dev->axil_bar_offset)     ] = (uint32_t)(qp->dev_written_dbs->pa >> 32);  //30// CQ DB Addres 
    dev->axil_bar[ERNIC_QP_ADDR(qpid, RQWPTRDBADDi + dev->axil_bar_offset)    ] = (uint32_t)((qp->dev_written_dbs->pa + 4) & 0x00000000FFFFFFFF);  //32// RQ DB address
    dev->axil_bar[ERNIC_QP_ADDR(qpid, RQWPTRDBADDMSBi + dev->axil_bar_offset) ] = (uint32_t)(qp->dev_written_dbs->pa >> 32);  //32// RQ DB address
    dev->axil_bar[ERNIC_QP_ADDR(qpid, CQBAi + dev->axil_bar_offset)           ] = (uint32_t)(qp->cq->pa & 0x00000000FFFFFFFF);  //34 // CQ Base Address
    dev->axil_bar[ERNIC_QP_ADDR(qpid, CQBAMSBi + dev->axil_bar_offset)        ] = (uint32_t)(qp->cq->pa >> 32);  //34 // EXTRA
    dev->axil_bar[ERNIC_QP_ADDR(qpid, DESTQPCONFi + dev->axil_bar_offset)     ] = dest_qpid;
    dev->axil_bar[ERNIC_QP_ADDR(qpid, QDEPTHi + dev->axil_bar_offset)         ] = qdepth;  //36 // Queue Depth
    dev->axil_bar[ERNIC_QP_ADDR(qpid, SQBAi + dev->axil_bar_offset)           ] = (uint32_t)(qp->sq->pa & 0x00000000FFFFFFFF);  //37 // SQ Base Address
    dev->axil_bar[ERNIC_QP_ADDR(qpid, SQBAMSBi + dev->axil_bar_offset)        ] = (uint32_t)(qp->sq->pa >> 32);  //37 // EXTRA
    

    // These are the configuration registers I want to change how they are implemented, but for now I keep the hardcoded values used in the example design
    if(enable_cq) {
        dev->axil_bar[ERNIC_QP_ADDR(qpid, QPCONFi + dev->axil_bar_offset)         ] = 0x00010437;  //31// QP Configuration
    }
    else {
        dev->axil_bar[ERNIC_QP_ADDR(qpid, QPCONFi + dev->axil_bar_offset)         ] = 0x00010417;  //31// QP Configuration
    }
    dev->axil_bar[ERNIC_QP_ADDR(qpid, SQPSNi + dev->axil_bar_offset)          ] = 0x00a02aaa;  //66// SQ PSN
    dev->axil_bar[ERNIC_QP_ADDR(qpid, LSTRQREQi + dev->axil_bar_offset)       ] = 0x04a02aa9;  //65// Last incoming RQ packet details
    dev->axil_bar[ERNIC_QP_ADDR(qpid, QPADVCONFi + dev->axil_bar_offset)      ] = 0x9bf1b002;  //38 // QP Advanced Configuration
    dev->axil_bar[ERNIC_QP_ADDR(qpid, STATQPi + dev->axil_bar_offset)         ] = 0x70000600;  //33 // Status QPI
    dev->axil_bar[ERNIC_QP_ADDR(qpid, STATMSNi + dev->axil_bar_offset)        ] = 0x004e503b;  //39// Status Message Sequence Number.
    // For some reason when I configure the timeouts the system crashes. Think it is because ERNIC stops responding to MMIO. Need to debug.
    //dev->axil_bar[ERNIC_QP_ADDR(qpid, TIMEOUTCONFi + dev->axil_bar_offset)    ] = 0x00021714;  // Timeout configuration 
                                                                                               //   [5:0] Time Out Value (timeout = 4.096us * 2^TIMEOUTVAL)
                                                                                               //   [10:8] Maximum Retry Count 
                                                                                               //   [13:11] Maximum RNR Retry Count 
                                                                                               //   [20:16] RNR NACK Timeout value for outgoing packets
 
    // If a PD is associated with a queue associate it in the ERNIC
    if(pd != NULL) {
        dev->axil_bar[ERNIC_QP_ADDR(qpid, PDi + dev->axil_bar_offset)] = pd->pd_num;
    }

    return qp;
}

/* This function is used to print all the information
regarding a pcie_ernic_buff, typically used if users
want to get into the details */
void print_buff(struct pcie_ernic_buff* buff) {

    if(buff == NULL) {
        printf("[ERROR] print_buff give NULL buff\n");
        return;
    }
   
    printf("Data buffer information:\n"); 
    printf("\tVA: %p\n", buff->buff);
    printf("\tPA: 0x%lx\n", buff->pa);
    printf("\tSize: %ld\n", buff->size);
    printf("\tOn device: %d\n", buff->on_device);

    return;
}

/* This function is used to just print all of the 
addresses and doorbells of each queue pair */
void print_qp_state(struct pcie_ernic_qp* qp) {
    
    if(qp == NULL) {
        printf("[ERROR] print_qp_state given NULL qp\n");
        return;
    }

    printf("QP %d information:\n", qp->qpid);

#ifdef DEBUG_NETWORK
    printf("\tNetwork information:\n");
    printf("\t\tDestination MAC MSB: 0x%x\n", qp->dest_mac_msb);
    printf("\t\tDestination MAC LSB: 0x%x\n", qp->dest_mac_lsb);
    printf("\t\tDestination IP: 0x%x\n", qp->dest_ip);
    printf("\t\tDestination QP ID: %d\n", qp->dest_qpid);
#endif

    // Printing where the queues are in memory
#ifdef DEBUG_MEMORY
    printf("\tMemory addresses:\n");
    printf("\t\tQueue Depth: %d\n", qp->qdepth); 
    printf("\t\tRQ:\n");
    printf("\t\t\tVA: %p\n", qp->rq->buff);
    printf("\t\t\tPA: 0x%lx\n", qp->rq->pa);
    printf("\t\t\tAllocated Memory: %ld\n", qp->rq->size);
    printf("\t\t\tQueue Depth: %d\n", (uint32_t)(qp->qdepth >> 16));
    printf("\t\t\tOn device: %d\n", qp->rq->on_device);
    printf("\t\tSQ:\n");
    printf("\t\t\tVA: %p\n", qp->sq->buff);
    printf("\t\t\tPA: 0x%lx\n", qp->sq->pa);
    printf("\t\t\tAllocated Memory: %ld\n", qp->sq->size);
    printf("\t\t\tQueue Depth: %d\n", (uint32_t)(qp->qdepth & 0x0000FFFF));
    printf("\t\t\tOn device: %d\n", qp->sq->on_device);
    printf("\t\tCQ:\n");
    printf("\t\t\tEnabled: %d\n", qp->enable_cq);
    printf("\t\t\tVA: %p\n", qp->cq->buff);
    printf("\t\t\tPA: 0x%lx\n", qp->cq->pa);
    printf("\t\t\tAllocated Memory: %ld\n", qp->cq->size);
    printf("\t\t\tQueue Depth: %d\n", (uint32_t)(qp->qdepth & 0x0000FFFF));
    printf("\t\t\tOn device: %d\n", qp->cq->on_device);
#endif

#ifdef DEBUG_DB
    printf("\tDoorbell values:\n");
    printf("\t\tDev Written DB Address:\n");
    printf("\t\tVA: %p\n", qp->dev_written_dbs->buff);
    printf("\t\tPA: 0x%lx\n", qp->dev_written_dbs->pa);
    printf("\t\tSize: %ld\n", qp->dev_written_dbs->size);
    printf("\t\tOn Device: %ld\n", qp->dev_written_dbs->on_device);
    printf("\t\tCQ DB Val: 0x%x\n", *(uint32_t *)(qp->dev_written_dbs->buff));
    printf("\t\tRQ DB Val: 0x%x\n", *(uint32_t *)(qp->dev_written_dbs->buff + 4));
#endif

}

/* This function is used to print the state 
of an ERNIC device. Typically consisting of 
statistics of the number and types of packets
sent out */
void print_dev_state(struct pcie_ernic_dev* dev) {
    
    if(dev == NULL) {
        printf("[ERROR] print_dev_state given NULL dev\n");
        return;
    }
    
    printf("Checking ERNIC %d Status Registers...\n", dev->ernic_id);

    uint32_t num_send_packets_and_read_response_rcvd = dev->axil_bar[(INSRRPKTCNT + dev->axil_bar_offset) >> 2];
    uint32_t num_send_packets_rcvd = num_send_packets_and_read_response_rcvd & 0x0000FFFF; // least significant 16 bits
    uint32_t num_read_response_rcvd = (num_send_packets_and_read_response_rcvd & 0xFFFF0000) >> 16; // most significant 16 bits
    
    printf("\tNumber of SEND + WRITE packets received: %d\n", num_send_packets_rcvd);
    printf("\tNumber of READ RESPONSE packets received: %d\n", num_read_response_rcvd);
    
    
    uint32_t num_rd_wr_wqes = (dev->axil_bar[(OUTIOPKTCNT + dev->axil_bar_offset) >> 2] & 0xFFFF0000) >> 16;
    printf("\tNumber of READ + WRITE WQEs processed: %d\n", num_rd_wr_wqes);
   
    uint32_t num_acks_rcvd = dev->axil_bar[(INAMPKTCNT + dev->axil_bar_offset) >> 2] & 0x0000FFFF;
    printf("\tNumber of ACKs received: %d\n", num_acks_rcvd);

    uint32_t num_ack_sent = dev->axil_bar[(OUTAMPKTCNT + dev->axil_bar_offset) >> 2] & 0x0000FFFF;
    printf("\tNumber of ACKs sent: %d\n", num_ack_sent);

    uint32_t num_nack_sent = dev->axil_bar[(OUTNAKPKTCNT + dev->axil_bar_offset) >> 2] & 0x0000FFFF;
    printf("\tNumber of NACKs sent: %d\n", num_nack_sent);

    uint32_t num_rd_rsp_sent = dev->axil_bar[(OUTRDRSPPKTCNT + dev->axil_bar_offset) >> 2];
    printf("\tNumber of READ RESPONSE sent: %d\n", num_rd_rsp_sent);

#ifdef DEBUG_MEMORY
    // Printing debug information. Can be useful if you are looking at a waveform
    // and trying to see if it is writing to the error buffer
    printf("\tError buffer PA: 0x%lx\n", dev->err_buff->pa);
    printf("\tInc Packet Error Buffer PA: 0x%lx\n", dev->inc_packet_err_q->pa);
    printf("\tResponse Error BUffer PA: 0x%lx\n", dev->resp_err_buff->pa);
#endif
    
}

void print_op(int op) {

    // From PG332
    switch(op) {
        case 0x0: printf("RDMA WRITE\n"); break;
        case 0x1: printf("RDMA WRITE IMMDT\n"); break;
        case 0x2: printf("RDMA SEND\n"); break;
        case 0x3: printf("RDMA SEND IMMDT\n"); break;
        case 0x4: printf("RDMA READ\n"); break;
        case 0xC: printf("RDMA SEND INVLDT\n"); break;
        default: printf("UNKNOWN OP\n");
    }
    
}


// Prints the WQE at the provided index in the provided SQ
void print_wqe(struct pcie_ernic_qp *qp, uint32_t index) {

    if(qp == NULL) {
        printf("[ERROR] print_wqe was given NULL QP\n");
        return;
    }
    
    struct pcie_ernic_wqe wqe = ((struct pcie_ernic_wqe *)qp->sq->buff)[index];

    printf("WQE at CQ %d index %d:\n", qp->qpid, index);
    printf("\twrid: 0x%x\n", wqe.wrid);
    printf("\tladdr_hi: 0x%x\n", wqe.laddr_hi);
    printf("\tladdr_lo: 0x%x\n", wqe.laddr_lo);
    printf("\tlength: 0x%x\n", wqe.length);
    printf("\top: "); print_op(wqe.op);
    printf("\toffset_hi: 0x%x\n", wqe.offset_hi);
    printf("\toffset_lo: 0x%x\n", wqe.offset_lo);
    printf("\trtag: 0x%x\n", wqe.rtag);
    printf("\tsend_data_dw_0: 0x%x\n", wqe.send_data_dw_0);
    printf("\tsend_data_dw_1: 0x%x\n", wqe.send_data_dw_1);
    printf("\tsend_data_dw_2: 0x%x\n", wqe.send_data_dw_2);
    printf("\tsend_data_dw_3: 0x%x\n", wqe.send_data_dw_3);
    printf("\timmdt_data: 0x%x\n", wqe.immdt_data);

}


// Prints the CQE at the provided index in the provided CQ 
void print_cqe(struct pcie_ernic_qp *qp, uint32_t index) {
    
    if(qp == NULL) {
        printf("[ERROR] print_cqe was given NULL QP\n");
        return;
    }
    
    struct pcie_ernic_cqe cqe = ((struct pcie_ernic_cqe *)qp->cq->buff)[index];

    printf("CQE at CQ %d index %d:\n", qp->qpid, index);
    printf("\twrid: 0x%x\n", cqe.wrid);
    printf("\top: "); print_op(cqe.op);
    printf("\terr_flags: %d\n", cqe.err_flags);
    
    return;
}

/* Creates a pcie_ernic_mr struct. This will write
the permissions, key, length, VA, and PA of a buffer 
to be registered to the PD table */
struct pcie_ernic_mr *pcie_ernic_reg_mr(struct pcie_ernic_dev *dev,
                                        struct pcie_ernic_pd *pd,
                                        struct pcie_ernic_buff *buff,
                                        uint8_t key,
                                        uint32_t length,
                                        enum pd_access_flags flags) {

    if(dev == NULL) {
        printf("[ERROR] pcie_ernic_reg_mr given NULL dev\n");
        return NULL;
    } 

    if(pd == NULL) {
        printf("[ERROR] pcie_ernic_reg_mr given NULL pd\n");
        return NULL;
    }

    if(buff == NULL) {
        printf("[ERROR] pcie_ernic_reg_mr given NULL buff\n");
        return NULL;
    }

    if(pd->mr != NULL) {
        printf("[WARNING] Overwriting previous MR in PD %d\n", pd->pd_num);
        printf("Current PD Info:\n");
        printf("\tPA: 0x%lx\n", pd->mr->buff->pa);
        printf("\tLength: 0x%lx\n", pd->mr->buff->size);
        printf("New PD Info:\n");
        printf("\tPA: 0x%lx\n", buff->pa);
        printf("\tLength: 0x%lx\n", buff->size);

        // Checking if the buffers are physically contigous
        bool is_physically_contigous = (pd->mr->buff->pa + pd->mr->buff->size) == buff->pa;
        bool keys_match = pd->mr->key == key;
        bool permissions_match = pd->mr->flags == flags;
        if(is_physically_contigous && keys_match && permissions_match) {
            printf("Can combine memory regions in protection domain %d\n", pd->pd_num);
            printf("\tUpdating size of protection domain %d from %ld to %ld\n", pd->pd_num, pd->mr->buff->size, pd->mr->buff->size + length);
    
            // Just overwriting the length because these memory regions can be combined
            dev->axil_bar[ERNIC_PD_ADDR(pd->pd_num, WRRDBUFLEN + dev->axil_bar_offset)] = (uint32_t)(pd->mr->buff->size + length);

            // Returning the updated mr
            pd->mr->length = pd->mr->buff->size + length;
            return pd->mr;
        }
        else {
            printf("[WARNING] Buffers are physically contigous, keys don't match, or premisions are not the same. Overwriting previous memory region in protection domain %d\n", pd->pd_num);
        }

    }

    // Decoding the flag
    uint32_t flag_to_write = 0;
    if(flags == PD_READ_ONLY) {
        flag_to_write = 0;
    }
    else if(flags == PD_WRITE_ONLY) {
        flag_to_write = 1;
    }
    else if(flags == PD_READ_WRITE) {
        flag_to_write = 2;
    }
    else {
        printf("[ERROR] pcie_ernic_reg_mr given unrecognized flag");
        return NULL;
    }

    struct pcie_ernic_mr *ret_struct = (struct pcie_ernic_mr *)malloc(sizeof(struct pcie_ernic_mr));
    if(ret_struct == NULL) {
        printf("pcie_ernic_mr couldn't allocate memory\n");
        return NULL;
    }

    // Writing everything to the struct for book keeping
    ret_struct->pd = pd;
    ret_struct->buff = buff;
    ret_struct->key = key;
    ret_struct->length = length;
    ret_struct->flags = flags;

    // Writing everything to the ERNIC
#if defined(VERBOSE_DEBUG) && defined(DEBUG_MEMORY)
    printf("Registering memory at VA %p\n", buff->buff);
#endif

    dev->axil_bar[ERNIC_PD_ADDR(pd->pd_num, VIRTADDRLSB + dev->axil_bar_offset)     ] = (uint32_t)((unsigned long)(buff->buff) & 0x00000000FFFFFFFF);
    dev->axil_bar[ERNIC_PD_ADDR(pd->pd_num, VIRTADDRMSB + dev->axil_bar_offset)     ] = (uint32_t)((unsigned long)(buff->buff) >> 32);
    dev->axil_bar[ERNIC_PD_ADDR(pd->pd_num, BUFBASEADDRLSB + dev->axil_bar_offset)  ] = (uint32_t)(buff->pa & 0x00000000FFFFFFFF);
    dev->axil_bar[ERNIC_PD_ADDR(pd->pd_num, BUFBASEADDRMSB + dev->axil_bar_offset)  ] = (uint32_t)(buff->pa >> 32);
    dev->axil_bar[ERNIC_PD_ADDR(pd->pd_num, BUFRKEY + dev->axil_bar_offset)         ] = (uint32_t)key;
    dev->axil_bar[ERNIC_PD_ADDR(pd->pd_num, WRRDBUFLEN + dev->axil_bar_offset)      ] = (uint32_t)length;
    dev->axil_bar[ERNIC_PD_ADDR(pd->pd_num, ACCESSDESC + dev->axil_bar_offset)      ] = flag_to_write; 

    // Creating cyclic relationship between mr and pd
    pd->mr = ret_struct;

    return ret_struct;
}



/* This function will take in a pointer to a 
submission queue pcie_ernic_buff struct, as 
well as all of the information to create a WQE.
It will write the WQE to the index specified
on the submission queue. As of right now it 
does not ring the doorbell */
int write_wqe_to_sq(struct pcie_ernic_buff *sq,
    uint32_t    index,
    uint32_t	wrid,
    uint64_t    laddr,
    uint32_t    length,
    uint32_t    op,
    uint64_t    offset,
    uint32_t    rtag,
    uint64_t    send_data_hi,
    uint64_t    send_data_lo,
    uint32_t	immdt_data) {
                        
    int i = 0;
    
    // If the sq is null or the buff is null than return failure 
    if(sq == NULL || sq->buff == NULL) {
        printf("[ERROR] Failed to write wqe to sq\n");
        return 0;
    }
    
    // Create a pointer to the index of the SQ we want to write to
    struct pcie_ernic_wqe *wqe = &(((struct pcie_ernic_wqe *)(sq->buff))[index]);

    wqe->wrid           = wrid & 0x0000FFFF;
    wqe->laddr_lo       = (uint32_t)(laddr & 0x00000000FFFFFFFF);
    wqe->laddr_hi       = (uint32_t)(laddr >> 32);
    wqe->length         = length;
    wqe->op             = op & 0x000000FF;
    wqe->offset_lo      = (uint32_t)(offset & 0x00000000FFFFFFFF);
    wqe->offset_hi      = (uint32_t)(offset >> 32);
    wqe->rtag           = rtag;
    wqe->send_data_dw_0 = 0; // TODO
    wqe->send_data_dw_1 = 0; 
    wqe->send_data_dw_2 = 0;
    wqe->send_data_dw_3 = 0;
    wqe->immdt_data     = 0;
    wqe->reserved_1     = 0;
    wqe->reserved_2     = 0;
    wqe->reserved_3     = 0;
    
    // Write all of the components of the WQE
    /*wqe->wrid           = wrid & 0x0000FFFF; // wrid is only 16 bits, the other 16 bits are reserved
    wqe->laddr          = laddr;
    wqe->length         = length;
    wqe->op             = op;
    wqe->offset         = offset;
    wqe->rtag           = rtag;
    wqe->send_data_hi   = send_data_hi;
    wqe->send_data_lo   = send_data_lo;
    wqe->immdt_data     = immdt_data;
    
    // Writing all of the reserved components to 0
    wqe->reserved_2[0]  = 0;
    wqe->reserved_2[1]  = 0;
    wqe->reserved_2[2]  = 0;
    wqe->reserved_3     = 0;
    wqe->reserved_4     = 0;


*/

    return 1;
                        
}

int pcie_ernic_post_wqe(
    struct pcie_ernic_dev *dev,
    struct pcie_ernic_qp *qp,
    uint32_t	wrid,
    uint64_t    laddr,
    uint32_t    length,
    uint32_t    op,
    uint64_t    offset,
    uint32_t    rtag,
    uint64_t    send_data_hi,
    uint64_t    send_data_lo,
    uint32_t	immdt_data,
    bool        poll) {

   
    if(dev == NULL) {
        printf("[ERROR] pcie_ernic_post_wqe given NULL dev\n");
        return 0;
    }

    if(qp == NULL) {
        printf("[ERROR] pcie_ernic_post_wqe given NULL qp\n");
        return 0;
    }

    int ret_val = write_wqe_to_sq(qp->sq,   // pcie_ernic_qp
                               qp->sq_pidb, // index
                               wrid,        // wrid
                               laddr,       // laddr 
                               length,      // length
                               op,          // op
                               offset,      // offset
                               rtag,        // rtag
                               0,           // send_data_hi
                               0,           // send_data_lo
                               0);          // immdt_data

    if(ret_val == 0) {
        printf("[ERROR] Couldn't write wqe to sq\n");
        return 0;
    }

    // Incrementing our internal count of the doorbell
    qp->sq_pidb++;

    // Ringing the SQ doorbell
    write_sq_pidb_db(dev, qp, qp->sq_pidb);

    // If poll is set, poll on completion
    if(poll) {
        read_sq_cidb_db(dev, qp, true);
    }

    return ret_val;
    

}

void pcie_ernic_free_qp(struct pcie_ernic_dev *dev, struct pcie_ernic_qp *qp) {
    
    if(qp == NULL) {
        printf("[ERROR] pcie_ernic_free_qp was given a NULL qp\n");
        return;
    }
    
    // 1. Wait for SQ and outstanding queues to become empty. The status bits are in STATQPi register
    uint32_t temp_val = dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, STATQPi + dev->axil_bar_offset)];
    while(!((temp_val >> 9) & 0x00000001) || !((temp_val >> 10) & 0x00000001)) {
        sleep(1);
        printf("[INFO] Waiting on SQ and oustanding queues to become empty before deleting QP...");
        temp_val = dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, STATQPi + dev->axil_bar_offset)];
    }
    

    // 2. Wait for all completions received for WQEs in SQ. This is done by checking SQPIi and CQHEADi registers.
    while(dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, SQPIi + dev->axil_bar_offset)] != dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, CQHEADi + dev->axil_bar_offset)]) {
        sleep(1);
        printf("[INFO] Waiting to receieve all completions before deleting QP...\n");
    }

    // 3. Enable the software override by writing to XRNIC_ADV_CONF register and disable the QP by writing to QPCONFi register.
    dev->axil_bar[(XRNICADCONF        + dev->axil_bar_offset)  >> 2]                = 0x00000001;
    //dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, QPCONFi + dev->axil_bar_offset)         ] = 0x0000000;  //31// QP Configuration

    // 4. Reset the QP pointers by writing 0 to RQWPTRDBADDi, SQPIi, CQHEADi, RQCIi, 
    // STATRQPIDBi, STATCURSQPTRi, SQPSNi, LSTRQREQi, STATMSN, and then disable the QP 
    // and kept it in recovery mode by writing QPCONFi.
    // TODO: I am not sure what to write to QPCONFi, nowhere else is reset mode discussed
    dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, RQWPTRDBADDi + dev->axil_bar_offset)      ] = 0x0000000;  
    dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, RQWPTRDBADDMSBi + dev->axil_bar_offset)   ] = 0x0000000;
    dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, SQPIi + dev->axil_bar_offset)             ] = 0x0000000;  
    dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, CQHEADi + dev->axil_bar_offset)           ] = 0x0000000;  
    dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, RQCIi + dev->axil_bar_offset)             ] = 0x0000000;  
    dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, STATRQPIDBi + dev->axil_bar_offset)       ] = 0x0000000;  
    dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, STATCURSQPTRi + dev->axil_bar_offset)     ] = 0x0000000;  
    dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, SQPSNi + dev->axil_bar_offset)            ] = 0x0000000;  
    dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, LSTRQREQi + dev->axil_bar_offset)         ] = 0x0000000;  
    dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, STATMSN + dev->axil_bar_offset)           ] = 0x0000000;  
    dev->axil_bar[ERNIC_QP_ADDR(qp->qpid, QPCONFi + dev->axil_bar_offset)           ] = 0x0000000;  

    // 5. After resetting pointers, software override should be disabled by writing to XRNIC_ADV_CONF.
    dev->axil_bar[(XRNICADCONF        + dev->axil_bar_offset)  >> 2]  = 0x00000000;

    // 6. After this, software should free memory allocated for SQ, RQ, and CQs.
    pcie_ernic_free_buff(qp->sq);
    pcie_ernic_free_buff(qp->cq);
    pcie_ernic_free_buff(qp->rq);
    
    
    // Setting t
    free(qp);
    qp = NULL;
    
    return;
}

void pcie_ernic_free_dev(struct pcie_ernic_dev *dev) {
    
    int i = 0;
    
    if(dev == NULL) {
        printf("[ERROR] pcie_ernic_free_dev was given NULL dev\n");
        return;
    }
    
    // Iterating over each QP, and if it is not null then freeing it
    for(i = 0; i < NUM_QPS; i++) {
        if(dev->qps[i] != NULL) {
            printf("Freeing QP %d\n", i);
            pcie_ernic_free_qp(dev, dev->qps[i]);
        }
    }
    
    // Clean up the AXIL BAR resources
    if(munmap(dev->axil_bar, dev->axil_bar_size) == -1) {
        printf("[ERROR] Failed to unmap axil BAR\n");
    }
   
    // Free the device memory allocator
    free_dev_mem_allocator(dev->allocator); 
    
    // Freeing the device itself
    free(dev);
    dev = NULL;
    
    
}


// Wrapper to register and then advertise a buffer to another machine
void pcie_ernic_reg_adv(struct pcie_ernic_dev *dev,
                        struct pcie_ernic_buff *buff, 
                        struct pcie_ernic_qp *qp,
                        uint8_t key,
                        uint32_t length,
                        enum pd_access_flags flags) 
{

    if(dev == NULL) {
        printf("[ERROR] pcie_ernic_reg_adv given NULL dev\n");
        return;
    }

    if(buff == NULL) {
        printf("[ERROR] pcie_ernic_reg_adv given NULL buff\n");
        return;
    }

    if(qp == NULL) {
        printf("[ERROR] pcie_ernic_reg_adv given NULL qp\n");
        return;
    }

    if(qp->pd == NULL) {
        printf("[ERROR] pcie_ernic_reg_adv given NULL qp->pd\n");
        return;
    }
   
    // Registering the buffer with the QP via the protection domain 
    struct pcie_ernic_mr *mr = pcie_ernic_reg_mr(dev, qp->pd, buff, key, length, flags);

    // Allocating rkey and vaddr buffers so we can send them
    struct pcie_ernic_buff *rkey_buff = pcie_ernic_malloc(dev, 4096 /* size */, true /* on_device */);
    *(uint32_t *)(rkey_buff->buff) = key;

    struct pcie_ernic_buff *vaddr_buff = pcie_ernic_malloc(dev, 4096 /* size */, true /* on_device */);
    *(uint64_t *)(vaddr_buff->buff) = (uint64_t)(buff->buff);
    
    // Writing a SEND with the r_key
#ifdef VERBOSE_DEBUG
    printf("[INFO] Sending SEND Packet 1 to advertise buffer\n");
#endif
    int ret_val = pcie_ernic_post_wqe(
                                dev,                                    // dev
                                qp,                                     // pcie_ernic_qp
                                0xe0a6,                                 // wrid
                                rkey_buff->pa,                          // laddr 
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
        return;
    }

    // Writing a SEND with the r_key
#ifdef VERBOSE_DEBUG
    printf("[INFO] Sending SEND Packet 2 to advertise buffer\n");
#endif
    ret_val = pcie_ernic_post_wqe(
                                dev,                                    // dev
                                qp,                                     // pcie_ernic_qp
                                0xe0a6,                                 // wrid
                                vaddr_buff->pa,                         // laddr 
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
        return;
    }

    // Freeing the buffers. We can do this because 
    // when we do the SENDs we poll on the completions
    pcie_ernic_free_buff(rkey_buff);
    pcie_ernic_free_buff(vaddr_buff);

    return;
}

void pcie_ernic_recv_buff(struct pcie_ernic_dev *dev, 
                        struct pcie_ernic_qp *qp,
                        uint32_t *rkey,
                        uint64_t *vaddr)
{

    if(dev == NULL) {
        printf("[ERROR] pcie_ernic_recv_buff was given NULL dev\n");
        return;
    }

    if(qp == NULL) {
        printf("[ERROR] pcie_ernic_recv_buff was given NULL qp\n");
        return;
    }

    if(rkey == NULL) {
        printf("[ERROR] pcie_ernic_recv_buff was given NULL rkey\n");
        return;
    }

    if(vaddr == NULL) {
        printf("[ERROR] pcie_Ernic_recv_buff was given NULL vaddr\n");
        return;
    }

#ifdef VERBOSE_DEBUG
    printf("[INFO] Polling on SEND 1 when receiving buffer\n");
#endif
    pcie_ernic_flush_cache();
    void *rqe_ptr = pcie_ernic_post_recv(dev, qp);
    if(rqe_ptr == NULL) {
        printf("[ERROR] pcie_ernic_post_recv returned NULL\n");
        return;
    }

    // Copying it so it isn't in the RQ
    *rkey = *(uint32_t *)(rqe_ptr);
#ifdef VERBOSE_DEBUG
    printf("[INFO] key of remote buffer: 0x%x\n", *rkey);
    printf("[INFO] Polling on SEND 2 when receiving buffer\n");
#endif
    rqe_ptr = pcie_ernic_post_recv(dev, qp);
    if(rqe_ptr == NULL) {
        printf("[ERROR] pcie_ernic_post_recv returned NULL\n");
        return;
    }

    // Copying it so it isn't in the RQ
    *vaddr = *(uint64_t *)(rqe_ptr);
#ifdef VERBOSE_DEBUG
    printf("[INFO] vaddr of remote buffer: 0x%lx\n", *vaddr);
#endif

    return;

}


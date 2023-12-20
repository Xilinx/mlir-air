//===- queue.cpp ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "air.hpp"
#include "air_host.h"
#include "air_host_impl.h"
#include "air_queue.h"
#include "airbin.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa_ext_air.h"
#include <gelf.h>

#define DEBUG_QUEUE

#ifdef DEBUG_QUEUE
#include <stdio.h>
#define DBG_PRINT printf
#else
#define DBG_PRINT(...)
#endif // DEBUG_QUEUE

#define ALIGN(_x, _size) (((_x) + ((_size)-1)) & ~((_size)-1))

hsa_status_t air_get_agent_info(hsa_agent_t *agent, hsa_queue_t *queue,
                                hsa_air_agent_info_t attribute, void *data) {
  if ((data == nullptr) || (queue == nullptr)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Getting our slot in the queue
  uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queue, 1);
  uint64_t packet_id = wr_idx % queue->size;

  // Getting a pointer to where the packet is in the queue
  hsa_agent_dispatch_packet_t aql_pkt;

  // Writing the fields to the packet
  aql_pkt.arg[0] = attribute;
  aql_pkt.type = AIR_PKT_TYPE_GET_INFO;
  aql_pkt.return_address = 0;
  aql_pkt.header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  // Now, we need to send the packet out
  air_queue_dispatch_and_wait(agent, queue, packet_id, wr_idx, &aql_pkt);

  // We encode the response in the packet, so need to peek in to get the data
  hsa_agent_dispatch_packet_t *pkt_peek =
      &reinterpret_cast<hsa_agent_dispatch_packet_t *>(
          queue->base_address)[packet_id];
  std::memcpy(data, &pkt_peek->return_address, 8);

  return HSA_STATUS_SUCCESS;
}

// TODO: Get rid of this complications with C++ templates, will need to move
// thing around a bit
hsa_status_t air_queue_dispatch(hsa_queue_t *q, uint64_t packet_id,
                                uint64_t doorbell,
                                hsa_agent_dispatch_packet_t *pkt) {

  // Write the packet to the queue
  air_write_pkt<hsa_agent_dispatch_packet_t>(q, packet_id, pkt);

  // Ringing the doorbell
  hsa_signal_store_screlease(q->doorbell_signal, doorbell);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_dispatch(hsa_queue_t *q, uint64_t packet_id,
                                uint64_t doorbell,
                                hsa_barrier_and_packet_t *pkt) {

  // Write the packet to the queue
  air_write_pkt<hsa_barrier_and_packet_t>(q, packet_id, pkt);

  // Ringing the doorbell
  hsa_signal_store_screlease(q->doorbell_signal, doorbell);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_wait(hsa_queue_t *q, hsa_agent_dispatch_packet_t *pkt) {
  // wait for packet completion
  while (hsa_signal_wait_scacquire(pkt->completion_signal,
                                   HSA_SIGNAL_CONDITION_EQ, 0, 0x80000,
                                   HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout!\n");
    printf("%x\n", pkt->header);
    printf("%x\n", pkt->type);
    printf("%lx\n", pkt->completion_signal.handle);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_wait(hsa_queue_t *q, hsa_barrier_and_packet_t *pkt) {
  // wait for packet completion
  while (hsa_signal_wait_scacquire(pkt->completion_signal,
                                   HSA_SIGNAL_CONDITION_EQ, 0, 0x80000,
                                   HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout!\n");
    printf("%x\n", pkt->header);
    printf("%lx\n", pkt->completion_signal.handle);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_dispatch_and_wait(hsa_agent_t *agent, hsa_queue_t *q,
                                         uint64_t packet_id, uint64_t doorbell,
                                         hsa_agent_dispatch_packet_t *pkt,
                                         bool destroy_signal) {

  // dispatch and wait has blocking semantics so we can internally create the
  // signal
  hsa_amd_signal_create_on_agent(1, 0, nullptr, agent, 0,
                                 &(pkt->completion_signal));

  // Write the packet to the queue
  air_write_pkt<hsa_agent_dispatch_packet_t>(q, packet_id, pkt);

  // Ringing the doorbell
  hsa_signal_store_screlease(q->doorbell_signal, doorbell);

  // wait for packet completion
  while (hsa_signal_wait_scacquire(pkt->completion_signal,
                                   HSA_SIGNAL_CONDITION_EQ, 0, 0x80000,
                                   HSA_WAIT_STATE_ACTIVE) != 0)
    ;

  // Optionally destroying the signal
  if (destroy_signal) {
    hsa_signal_destroy(pkt->completion_signal);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_dispatch_and_wait(hsa_agent_t *agent, hsa_queue_t *q,
                                         uint64_t packet_id, uint64_t doorbell,
                                         hsa_barrier_and_packet_t *pkt,
                                         bool destroy_signal) {

  // dispatch and wait has blocking semantics so we can internally create the
  // signal
  hsa_amd_signal_create_on_agent(1, 0, nullptr, agent, 0,
                                 &(pkt->completion_signal));

  // Write the packet to the queue
  air_write_pkt<hsa_barrier_and_packet_t>(q, packet_id, pkt);

  // Ringing the doorbell
  hsa_signal_store_screlease(q->doorbell_signal, doorbell);

  // wait for packet completion
  while (hsa_signal_wait_scacquire(pkt->completion_signal,
                                   HSA_SIGNAL_CONDITION_EQ, 0, 0x80000,
                                   HSA_WAIT_STATE_ACTIVE) != 0)
    ;

  // Optionally destroying the signal
  if (destroy_signal) {
    hsa_signal_destroy(pkt->completion_signal);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_segment_init(hsa_agent_dispatch_packet_t *pkt,
                                     uint16_t herd_id, uint8_t start_col,
                                     uint8_t num_cols, uint8_t start_row,
                                     uint8_t num_rows) {

  pkt->arg[0] = 0;
  pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  pkt->arg[0] |= ((uint64_t)num_cols) << 40;
  pkt->arg[0] |= ((uint64_t)start_col) << 32;
  pkt->arg[0] |= ((uint64_t)num_rows) << 24;
  pkt->arg[0] |= ((uint64_t)start_row) << 16;

  pkt->arg[1] = herd_id; // Herd ID
  // pkt->arg[1] |= ((char)dma0) << 16;
  // pkt->arg[1] |= ((char)dma1) << 24;
  pkt->arg[2] = 0; // unused
  pkt->arg[3] = 0; // unused

  pkt->type = AIR_PKT_TYPE_SEGMENT_INITIALIZE;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_device_init(hsa_agent_dispatch_packet_t *pkt,
                                    uint32_t num_cols) {

  pkt->arg[0] = 0;
  pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  pkt->arg[0] |= ((uint64_t)num_cols << 40);

  pkt->type = AIR_PKT_TYPE_DEVICE_INITIALIZE;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_get_capabilities(hsa_agent_dispatch_packet_t *pkt,
                                         uint64_t return_address) {

  pkt->return_address = reinterpret_cast<void *>(return_address);

  pkt->type = AIR_PKT_TYPE_GET_CAPABILITIES;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_hello(hsa_agent_dispatch_packet_t *pkt,
                              uint64_t value) {

  pkt->arg[0] = value;

  pkt->type = AIR_PKT_TYPE_HELLO;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_post_rdma_wqe(hsa_agent_dispatch_packet_t *pkt,
                                      uint64_t remote_vaddr,
                                      uint64_t local_paddr, uint32_t length,
                                      uint8_t op, uint8_t key, uint8_t qpid,
                                      uint8_t ernic_sel) {

  // Creating the arguments localy and then writing over PCIe
  uint64_t arg2 = ((uint64_t)ernic_sel << 56) | ((uint64_t)qpid << 48) |
                  ((uint64_t)key << 40) | ((uint64_t)op << 32) |
                  ((uint64_t)length);

  pkt->arg[0] = remote_vaddr;
  pkt->arg[1] = local_paddr;
  pkt->arg[2] = arg2;

  pkt->type = AIR_PKT_TYPE_POST_RDMA_WQE;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_post_rdma_recv(hsa_agent_dispatch_packet_t *pkt,
                                       uint64_t local_paddr, uint32_t length,
                                       uint8_t qpid, uint8_t ernic_sel) {

  // Creating the arguments localy and then writing over PCIe
  uint64_t arg1 =
      ((uint64_t)ernic_sel << 48) | ((uint64_t)length << 16) | ((uint64_t)qpid);

  pkt->arg[0] = local_paddr;
  pkt->arg[1] = arg1;

  pkt->type = AIR_PKT_TYPE_POST_RDMA_RECV;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_tile_status(hsa_agent_dispatch_packet_t *pkt,
                                    uint8_t col, uint8_t row) {

  pkt->arg[0] = col;
  pkt->arg[1] = row;

  pkt->type = AIR_PKT_TYPE_CORE_STATUS;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_dma_status(hsa_agent_dispatch_packet_t *pkt,
                                   uint8_t col, uint8_t row) {

  pkt->arg[0] = col;
  pkt->arg[1] = row;

  pkt->type = AIR_PKT_TYPE_TDMA_STATUS;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_shimdma_status(hsa_agent_dispatch_packet_t *pkt,
                                       uint8_t col) {

  pkt->arg[0] = col;
  pkt->arg[1] = 0;

  pkt->type = AIR_PKT_TYPE_SDMA_STATUS;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_put_stream(hsa_agent_dispatch_packet_t *pkt,
                                   uint64_t stream, uint64_t value) {

  pkt->arg[0] = stream;
  pkt->arg[1] = value;

  pkt->type = AIR_PKT_TYPE_PUT_STREAM;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_get_stream(hsa_agent_dispatch_packet_t *pkt,
                                   uint64_t stream, uint64_t return_address) {

  pkt->arg[0] = stream;
  pkt->return_address = reinterpret_cast<void *>(return_address);

  pkt->type = AIR_PKT_TYPE_GET_STREAM;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_cdma_configure(hsa_agent_dispatch_packet_t *pkt,
                                       uint64_t dest, uint64_t source,
                                       uint32_t length) {

  pkt->arg[0] = dest;   // Destination (BD for SG mode)
  pkt->arg[1] = source; // Source (BD for SG mode)
  pkt->arg[2] = length; // Num Bytes (0xFFFFFFFF for SG mode)

  pkt->type = AIR_PKT_TYPE_CONFIGURE;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_cdma_memcpy(hsa_agent_dispatch_packet_t *pkt,
                                    uint64_t dest, uint64_t source,
                                    uint32_t length) {

  pkt->arg[0] = dest;   // Destination
  pkt->arg[1] = source; // Source
  pkt->arg[2] = length; // Num Bytes

  pkt->type = AIR_PKT_TYPE_CDMA;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_aie_lock_range(hsa_agent_dispatch_packet_t *pkt,
                                       uint16_t herd_id, uint64_t lock_id,
                                       uint64_t acq_rel, uint64_t value,
                                       uint8_t start_col, uint8_t num_cols,
                                       uint8_t start_row, uint8_t num_rows) {

  pkt->arg[0] = 0;
  pkt->arg[0] |= (AIR_ADDRESS_HERD_RELATIVE_RANGE << 48);
  pkt->arg[0] |= ((uint64_t)num_cols) << 40;
  pkt->arg[0] |= ((uint64_t)start_col) << 32;
  pkt->arg[0] |= ((uint64_t)num_rows) << 24;
  pkt->arg[0] |= ((uint64_t)start_row) << 16;
  pkt->arg[1] = lock_id;
  pkt->arg[2] = acq_rel;
  pkt->arg[3] = value;

  pkt->type = AIR_PKT_TYPE_XAIE_LOCK;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t
air_packet_nd_memcpy(hsa_agent_dispatch_packet_t *pkt, uint16_t herd_id,
                     uint8_t col, uint8_t direction, uint8_t channel,
                     uint8_t burst_len, uint8_t memory_space,
                     uint64_t phys_addr, uint32_t transfer_length1d,
                     uint32_t transfer_length2d, uint32_t transfer_stride2d,
                     uint32_t transfer_length3d, uint32_t transfer_stride3d,
                     uint32_t transfer_length4d, uint32_t transfer_stride4d) {

  pkt->arg[0] = 0;
  pkt->arg[0] |= ((uint64_t)memory_space) << 16;
  pkt->arg[0] |= ((uint64_t)channel) << 24;
  pkt->arg[0] |= ((uint64_t)col) << 32;
  pkt->arg[0] |= ((uint64_t)burst_len) << 52;
  pkt->arg[0] |= ((uint64_t)direction) << 60;

  pkt->arg[1] = phys_addr;
  pkt->arg[2] = transfer_length1d;
  pkt->arg[2] |= ((uint64_t)transfer_length2d) << 32;
  pkt->arg[2] |= ((uint64_t)transfer_stride2d) << 48;
  pkt->arg[3] = transfer_length3d;
  pkt->arg[3] |= ((uint64_t)transfer_stride3d) << 16;
  pkt->arg[3] |= ((uint64_t)transfer_length4d) << 32;
  pkt->arg[3] |= ((uint64_t)transfer_stride4d) << 48;

  pkt->type = AIR_PKT_TYPE_ND_MEMCPY;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_aie_lock(hsa_agent_dispatch_packet_t *pkt,
                                 uint16_t herd_id, uint64_t lock_id,
                                 uint64_t acq_rel, uint64_t value, uint8_t col,
                                 uint8_t row) {
  return air_packet_aie_lock_range(pkt, herd_id, lock_id, acq_rel, value, col,
                                   1, row, 1);
}

hsa_status_t
air_packet_barrier_and(hsa_barrier_and_packet_t *pkt, hsa_signal_t dep_signal0,
                       hsa_signal_t dep_signal1, hsa_signal_t dep_signal2,
                       hsa_signal_t dep_signal3, hsa_signal_t dep_signal4) {

  pkt->dep_signal[0] = dep_signal0;
  pkt->dep_signal[1] = dep_signal1;
  pkt->dep_signal[2] = dep_signal2;
  pkt->dep_signal[3] = dep_signal3;
  pkt->dep_signal[4] = dep_signal4;

  pkt->header = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t
air_packet_barrier_or(hsa_barrier_or_packet_t *pkt, hsa_signal_t dep_signal0,
                      hsa_signal_t dep_signal1, hsa_signal_t dep_signal2,
                      hsa_signal_t dep_signal3, hsa_signal_t dep_signal4) {

  pkt->dep_signal[0] = dep_signal0;
  pkt->dep_signal[1] = dep_signal1;
  pkt->dep_signal[2] = dep_signal2;
  pkt->dep_signal[3] = dep_signal3;
  pkt->dep_signal[4] = dep_signal4;

  pkt->header = (HSA_PACKET_TYPE_BARRIER_OR << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

/*
  'table' is an offset from the beginning of device memory
*/
hsa_status_t air_packet_load_airbin(hsa_agent_dispatch_packet_t *pkt,
                                    uint64_t table, uint16_t column) {
  DBG_PRINT("%s: table @ %lx\r\n", __func__, table);
  pkt->type = AIR_PKT_TYPE_AIRBIN;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);
  pkt->arg[0] = table;
  pkt->arg[1] = column;

  return HSA_STATUS_SUCCESS;
}

/*
  Load an airbin from a file into a device
*/
hsa_status_t air_load_airbin(hsa_agent_t *agent, hsa_queue_t *q,
                             const char *filename, uint8_t column,
                             uint32_t device_id) {
  hsa_status_t ret = HSA_STATUS_SUCCESS;
  int drv_fd = 0, elf_fd = 0;
  uint32_t dram_size;
  uint8_t *dram_ptr = NULL;
  uint8_t *data_ptr = NULL;
  struct timespec ts_start;
  struct timespec ts_end;
  Elf *inelf = NULL;
  GElf_Ehdr *ehdr = NULL;
  GElf_Ehdr ehdr_mem;
  uint64_t wr_idx = 0;
  hsa_agent_dispatch_packet_t pkt;
  size_t shnum;
  uint32_t table_idx = 0;
  airbin_table_entry *airbin_table;
  uint32_t data_offset = 0;
  uint32_t table_size = 0;
  struct stat elf_stat;

  auto time_spec_diff = [](struct timespec &start, struct timespec &end) {
    return (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
  };

  DBG_PRINT("%s fname=%s col=%u\r\n", __func__, filename, column);

  // open the AIRBIN file
  elf_fd = open(filename, O_RDONLY);
  if (elf_fd < 0) {
    printf("Can't open %s\n", filename);
    ret = HSA_STATUS_ERROR_INVALID_FILE;
    goto err_elf_open;
  }

  // calculate the size needed to load
  fstat(elf_fd, &elf_stat);
  // dram_size = elf_stat.st_size;
  dram_size = 6 * 1024 * 1024;
  if (table_size > dram_size) {
    printf("[ERROR] table size is larger than allocated DRAM. Exiting\n");
    ret = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    goto err_elf_open;
  }

  // get some DRAM from the device
  dram_ptr = (uint8_t *)air_malloc(dram_size);

  if (dram_ptr == MAP_FAILED) {
    printf("Error allocating %u DRAM\n", dram_size);
    ret = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    goto err_dev_mem_alloc;
  }

  DBG_PRINT("Allocated %u device memory HVA=0x%lx\r\n", dram_size,
            (uint64_t)dram_ptr);

  // check the characteristics
  elf_version(EV_CURRENT);
  inelf = elf_begin(elf_fd, ELF_C_READ, NULL);
  ehdr = gelf_getehdr(inelf, &ehdr_mem);
  if (ehdr == NULL) {
    printf("cannot get ELF header: %s\n", elf_errmsg(-1));
    ret = HSA_STATUS_ERROR_INVALID_FILE;
    goto err_elf_read;
  }

  // Read data as 64-bit little endian
  if ((ehdr->e_ident[EI_CLASS] != ELFCLASS64) ||
      (ehdr->e_ident[EI_DATA] != ELFDATA2LSB)) {
    printf("unexpected ELF format\n");
    ret = HSA_STATUS_ERROR_INVALID_FILE;
    goto err_elf_read;
  }

  if (elf_getshdrnum(inelf, &shnum) != 0) {
    printf("cannot get program header count: %s", elf_errmsg(-1));
    ret = HSA_STATUS_ERROR_INVALID_FILE;
    goto err_elf_read;
  }

  /*
    Even though not all sections are loadable, we use the section count as an
    upper bound for how much memory the table will take. We can then safely
    place data after that point and avoid any conflicts. A small amount of
    memory will be wasted but it is usually only two entries (32 bytes) so
    not a big deal. This allows us to do only a single pass on the ELF
    sections so it seems like a good trade-off.
  */
  DBG_PRINT("There are %lu sections\n", shnum);
  table_size = shnum * sizeof(airbin_table_entry);
  airbin_table = (airbin_table_entry *)dram_ptr;
  data_offset = table_size; // The data offset starts at the end of the table
  data_ptr = dram_ptr + table_size;

  // Iterate through all sections to create a table in device-readable format.
  for (unsigned int ndx = 0; ndx < shnum; ndx++) {
    GElf_Shdr shdr;
    Elf_Scn *sec = elf_getscn(inelf, ndx);
    if (sec == NULL) {
      printf("cannot get section %d: %s", ndx, elf_errmsg(-1));
      ret = HSA_STATUS_ERROR_INVALID_FILE;
      goto err_elf_read;
    }

    gelf_getshdr(sec, &shdr);

    // for each loadable program header
    if (shdr.sh_type != SHT_PROGBITS || !(shdr.sh_flags & SHF_ALLOC))
      continue;

    // copy the data from into device memory
    Elf_Data *desc;
    desc = elf_getdata(sec, NULL);
    if (!desc) {
      printf("Error reading data for section %u\n", ndx);
      ret = HSA_STATUS_ERROR_INVALID_FILE;
      goto err_elf_read;
    }
    memcpy(data_ptr, desc->d_buf, desc->d_size);

    airbin_table[table_idx].offset = data_offset;
    airbin_table[table_idx].size = shdr.sh_size;
    airbin_table[table_idx].addr = shdr.sh_addr;
    DBG_PRINT("table[%u] VA=0x%lx offset=0x%lx size=0x%lx addr=0x%lx\n",
              table_idx, (uint64_t)data_ptr, (uint64_t)data_offset,
              shdr.sh_size, shdr.sh_addr);

    table_idx++;
    data_offset += shdr.sh_size;
    data_ptr += shdr.sh_size;

    if (data_offset > dram_size) {
      printf("[ERROR] Overwriting allocated DRAM size. Exiting\n");
      ret = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      goto err_elf_read;
    }
  }

  // the last entry must be all 0's
  airbin_table[table_idx].offset = 0;
  airbin_table[table_idx].size = 0;
  airbin_table[table_idx].addr = 0;

  // Send configuration packet
  DBG_PRINT("Notifying device\n");
  wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
  air_packet_load_airbin(&pkt, (uint64_t)airbin_table, (uint16_t)column);

  // clock_gettime(CLOCK_BOOTTIME, &ts_start);
  air_queue_dispatch_and_wait(agent, q, wr_idx % q->size, wr_idx, &pkt);
  hsa_signal_destroy(pkt.completion_signal);
  // clock_gettime(CLOCK_BOOTTIME, &ts_end);

  // printf("airbin loading time: %0.8f sec\n", time_spec_diff(ts_start,
  // ts_end));

err_elf_read:
  elf_end(inelf);
  close(elf_fd);

err_elf_open:
err_dev_mem_alloc:
  return ret;
}

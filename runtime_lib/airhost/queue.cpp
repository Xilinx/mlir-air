//===- queue.cpp ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "air_host.h"
#include "air_host_impl.h"
#include "air_queue.h"
#include "airbin.h"

// Need access to the physical devices to determine where to
// write the queue
#ifdef AIR_PCIE
extern std::vector<air_physical_device_t> physical_devices;
#endif

hsa_status_t air_get_agent_info(queue_t *queue, air_agent_info_t attribute,
                                void *data) {
  if ((data == nullptr) || (queue == nullptr)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  uint64_t wr_idx = queue_add_write_index(queue, 1);
  uint64_t packet_id = wr_idx % queue->size;

  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(queue->base_address_vaddr) + packet_id;
  initialize_packet(pkt);
  pkt->arg[0] = attribute;
  // pkt->return_address = data; // FIXME this won't work without address
  // translation
  pkt->type = AIR_PKT_TYPE_GET_INFO;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);
  air_queue_dispatch_and_wait(queue, wr_idx, pkt);

  // fake it because of no address translation
  if (attribute <= AIR_AGENT_INFO_VENDOR_NAME) {
    std::memcpy(data, &pkt->return_address, 8);
  } else {
    uint64_t *p = static_cast<uint64_t *>(data);
    *p = pkt->return_address;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_create(uint32_t size, uint32_t type, queue_t **queue,
                              uint64_t paddr, uint32_t device_id) {
#ifdef AIR_PCIE

  if (device_id >= physical_devices.size()) {
    printf("[ERROR] No device id %d in system\n", device_id);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  // Right now assuming single device
  std::string bar_dev_file = air_get_bram_bar(device_id);
  int fd = open(bar_dev_file.c_str(), O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;

  paddr -= AIR_VCK190_SHMEM_BASE;
  uint64_t paddr_aligned = paddr & 0xfffffffffffff000;
  uint64_t paddr_offset = paddr & 0x0000000000000fff;

  uint64_t *bram_ptr = (uint64_t *)mmap(NULL, 0x100000, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd, paddr_aligned);

  // printf("Opened shared memory paddr: %p vaddr: %p\n", paddr, bram_ptr);
  uint64_t q_paddr =
      bram_ptr[paddr_offset / sizeof(uint64_t)] - AIR_VCK190_SHMEM_BASE;
  uint64_t q_offset = q_paddr;
  queue_t *q = (queue_t *)(((size_t)bram_ptr) + q_offset + paddr_offset);
  // printf("Queue location at paddr: %p vaddr: %p\n",
  // bram_ptr[paddr_offset/sizeof(uint64_t)]-AIR_VCK190_SHMEM_BASE, q);

#else
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;

  uint64_t paddr_aligned = paddr & 0xfffffffffffff000;
  uint64_t paddr_offset = paddr & 0x0000000000000fff;

  uint64_t *bram_ptr = (uint64_t *)mmap(NULL, 0x8000, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd, paddr_aligned);

  // printf("Opened shared memory paddr: %p vaddr: %p\n", paddr, bram_ptr);
  uint64_t q_paddr = bram_ptr[paddr_offset / sizeof(uint64_t)];
  uint64_t q_offset = q_paddr - paddr;
  queue_t *q = (queue_t *)(((size_t)bram_ptr) + q_offset + paddr_offset);
  // printf("Queue location at paddr: %p vaddr: %p\n",
  // bram_ptr[paddr_offset/sizeof(uint64_t)], q);
#endif

  if (q->id != 0xacdc) {
    // printf("%s error invalid id %x\n", __func__, q->id);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->size != size) {
    // printf("%s error size mismatch %d\n", __func__, q->size);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->type != type) {
    // printf("%s error type mismatch %d\n", __func__, q->type);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

#ifdef AIR_PCIE
  uint64_t base_address_offset = q->base_address - AIR_VCK190_SHMEM_BASE;
  q->base_address_vaddr = ((size_t)bram_ptr) + base_address_offset;
  q->base_address_paddr = q->base_address;
#else
  uint64_t base_address_offset = q->base_address - paddr_aligned;
  q->base_address_vaddr = ((size_t)bram_ptr) + base_address_offset;
  q->base_address_paddr = q->base_address;
#endif

  *queue = q;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_dispatch(queue_t *q, uint64_t doorbell,
                                dispatch_packet_t *pkt) {
  // dispatch packet
  signal_create(1, 0, NULL, (signal_t *)&pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t *)&q->doorbell);
  signal_store_release((signal_t *)&q->doorbell, doorbell);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_wait(queue_t *q, dispatch_packet_t *pkt) {
  // wait for packet completion
  while (signal_wait_acquire((signal_t *)&pkt->completion_signal,
                             HSA_SIGNAL_CONDITION_EQ, 0, 0x80000,
                             HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout!\n");
    printf("%x\n", pkt->header);
    printf("%x\n", pkt->type);
    printf("%x\n", (unsigned)pkt->completion_signal);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_dispatch_and_wait(queue_t *q, uint64_t doorbell,
                                         dispatch_packet_t *pkt) {
  // dispatch packet
  signal_create(1, 0, NULL, (signal_t *)&pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t *)&q->doorbell);
  signal_store_release((signal_t *)&q->doorbell, doorbell);

  // wait for packet completion
  while (signal_wait_acquire((signal_t *)&pkt->completion_signal,
                             HSA_SIGNAL_CONDITION_EQ, 0, 0x80000,
                             HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout!\n");
    printf("%x\n", pkt->header);
    printf("%x\n", pkt->type);
    printf("%x\n", (unsigned)pkt->completion_signal);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_herd_init(dispatch_packet_t *pkt, uint16_t herd_id,
                                  uint8_t start_col, uint8_t num_cols,
                                  uint8_t start_row, uint8_t num_rows) {
  // uint8_t start_row, uint8_t num_rows,
  // uint16_t dma0, uint16_t dma1) {
  initialize_packet(pkt);

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

  pkt->type = AIR_PKT_TYPE_HERD_INITIALIZE;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_device_init(dispatch_packet_t *pkt, uint32_t num_cols) {
  initialize_packet(pkt);

  pkt->arg[0] = 0;
  pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  pkt->arg[0] |= ((uint64_t)num_cols << 40);

  pkt->type = AIR_PKT_TYPE_DEVICE_INITIALIZE;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_get_capabilities(dispatch_packet_t *pkt,
                                         uint64_t return_address) {
  initialize_packet(pkt);

  pkt->return_address = return_address;

  pkt->type = AIR_PKT_TYPE_GET_CAPABILITIES;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_hello(dispatch_packet_t *pkt, uint64_t value) {
  initialize_packet(pkt);

  pkt->arg[0] = value;

  pkt->type = AIR_PKT_TYPE_HELLO;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_post_rdma_wqe(dispatch_packet_t *pkt,
                                      uint64_t remote_vaddr,
                                      uint64_t local_paddr, uint32_t length,
                                      uint8_t op, uint8_t key, uint8_t qpid,
                                      uint8_t ernic_sel) {
  initialize_packet(pkt);

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

hsa_status_t air_packet_post_rdma_recv(dispatch_packet_t *pkt,
                                       uint64_t local_paddr, uint32_t length,
                                       uint8_t qpid, uint8_t ernic_sel) {
  initialize_packet(pkt);

  // Creating the arguments localy and then writing over PCIe
  uint64_t arg1 =
      ((uint64_t)ernic_sel << 48) | ((uint64_t)length << 16) | ((uint64_t)qpid);

  pkt->arg[0] = local_paddr;
  pkt->arg[1] = arg1;

  pkt->type = AIR_PKT_TYPE_POST_RDMA_RECV;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_tile_status(dispatch_packet_t *pkt, uint8_t col,
                                    uint8_t row) {
  initialize_packet(pkt);

  pkt->arg[0] = col;
  pkt->arg[1] = row;

  pkt->type = AIR_PKT_TYPE_CORE_STATUS;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_dma_status(dispatch_packet_t *pkt, uint8_t col,
                                   uint8_t row) {
  initialize_packet(pkt);

  pkt->arg[0] = col;
  pkt->arg[1] = row;

  pkt->type = AIR_PKT_TYPE_TDMA_STATUS;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_shimdma_status(dispatch_packet_t *pkt, uint8_t col) {
  initialize_packet(pkt);

  pkt->arg[0] = col;
  pkt->arg[1] = 0;

  pkt->type = AIR_PKT_TYPE_SDMA_STATUS;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_put_stream(dispatch_packet_t *pkt, uint64_t stream,
                                   uint64_t value) {
  initialize_packet(pkt);

  pkt->arg[0] = stream;
  pkt->arg[1] = value;

  pkt->type = AIR_PKT_TYPE_PUT_STREAM;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_get_stream(dispatch_packet_t *pkt, uint64_t stream,
                                   uint64_t return_address) {
  initialize_packet(pkt);

  pkt->arg[0] = stream;
  pkt->return_address = return_address;

  pkt->type = AIR_PKT_TYPE_GET_STREAM;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_l2_dma(dispatch_packet_t *pkt, uint64_t stream,
                               l2_dma_cmd_t cmd) {
  initialize_packet(pkt);

  pkt->arg[0] = stream;
  pkt->arg[1] = 0;
  pkt->arg[1] |= ((uint64_t)cmd.select) << 32;
  pkt->arg[1] |= cmd.length << 18;
  pkt->arg[1] |= cmd.uram_addr << 5;
  pkt->arg[1] |= cmd.id;

  pkt->type = AIR_PKT_TYPE_PUT_STREAM;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_cdma_configure(dispatch_packet_t *pkt, uint64_t dest,
                                       uint64_t source, uint32_t length,
                                       airbin_size *airbin_data) {
  initialize_packet(pkt);

  pkt->arg[0] = dest;   // Destination (BD for SG mode)
  pkt->arg[1] = source; // Source (BD for SG mode)
  pkt->arg[2] = length; // Num Bytes (0xFFFFFFFF for SG mode)

  pkt->arg[3] = 0;
  if (airbin_data != nullptr) {
    pkt->arg[3] |= ((uint64_t)(airbin_data->num_cols)) << 24u;
    pkt->arg[3] |= ((uint64_t)(airbin_data->start_col)) << 16u;
    pkt->arg[3] |= ((uint64_t)(airbin_data->num_rows)) << 8u;
    pkt->arg[3] |= ((uint64_t)(airbin_data->start_row));
  }

  pkt->type = AIR_PKT_TYPE_CONFIGURE;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_cdma_memcpy(dispatch_packet_t *pkt, uint64_t dest,
                                    uint64_t source, uint32_t length) {
  initialize_packet(pkt);

  pkt->arg[0] = dest;   // Destination
  pkt->arg[1] = source; // Source
  pkt->arg[2] = length; // Num Bytes

  pkt->type = AIR_PKT_TYPE_CDMA;
  pkt->header = (HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_aie_lock_range(dispatch_packet_t *pkt, uint16_t herd_id,
                                       uint64_t lock_id, uint64_t acq_rel,
                                       uint64_t value, uint8_t start_col,
                                       uint8_t num_cols, uint8_t start_row,
                                       uint8_t num_rows) {
  initialize_packet(pkt);

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
air_packet_nd_memcpy(dispatch_packet_t *pkt, uint16_t herd_id, uint8_t col,
                     uint8_t direction, uint8_t channel, uint8_t burst_len,
                     uint8_t memory_space, uint64_t phys_addr,
                     uint32_t transfer_length1d, uint32_t transfer_length2d,
                     uint32_t transfer_stride2d, uint32_t transfer_length3d,
                     uint32_t transfer_stride3d, uint32_t transfer_length4d,
                     uint32_t transfer_stride4d) {

  initialize_packet(pkt);

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

hsa_status_t air_packet_aie_lock(dispatch_packet_t *pkt, uint16_t herd_id,
                                 uint64_t lock_id, uint64_t acq_rel,
                                 uint64_t value, uint8_t col, uint8_t row) {
  return air_packet_aie_lock_range(pkt, herd_id, lock_id, acq_rel, value, col,
                                   1, row, 1);
}

hsa_status_t air_packet_barrier_and(barrier_and_packet_t *pkt,
                                    uint64_t dep_signal0, uint64_t dep_signal1,
                                    uint64_t dep_signal2, uint64_t dep_signal3,
                                    uint64_t dep_signal4) {

  pkt->dep_signal[0] = dep_signal0;
  pkt->dep_signal[1] = dep_signal1;
  pkt->dep_signal[2] = dep_signal2;
  pkt->dep_signal[3] = dep_signal3;
  pkt->dep_signal[4] = dep_signal4;

  pkt->header = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_barrier_or(barrier_or_packet_t *pkt,
                                   uint64_t dep_signal0, uint64_t dep_signal1,
                                   uint64_t dep_signal2, uint64_t dep_signal3,
                                   uint64_t dep_signal4) {

  pkt->dep_signal[0] = dep_signal0;
  pkt->dep_signal[1] = dep_signal1;
  pkt->dep_signal[2] = dep_signal2;
  pkt->dep_signal[3] = dep_signal3;
  pkt->dep_signal[4] = dep_signal4;

  pkt->header = (HSA_PACKET_TYPE_BARRIER_OR << HSA_PACKET_HEADER_TYPE);

  return HSA_STATUS_SUCCESS;
}

int air_load_airbin(queue_t *q, const char *filename, uint8_t column,
                    uint32_t device_id) {

  // Initializing the device memory allocator
  if (air_init_dev_mem_allocator(0x2000000 /* dev_mem_size */,
                                 device_id /* device_id (optional)*/)) {
    std::cout << "Error creating device memory allocator" << std::endl;
    return -1;
  }

  auto *data_p = air_dev_mem_alloc(2 * 65536);
  volatile uint32_t *bram_ptr = reinterpret_cast<volatile uint32_t *>(data_p);
  auto *paddr = reinterpret_cast<uint32_t *>(air_dev_mem_get_pa(data_p));

  auto *bd_p = air_dev_mem_alloc(0x8000);
  volatile uint32_t *bd_ptr = reinterpret_cast<volatile uint32_t *>(bd_p);
  auto bd_paddr = uint64_t(air_dev_mem_get_pa(bd_p));

  std::ifstream infile{filename};

  auto size = readairbinsize(infile, column);

  // AIRBIN from file to memory
  uint64_t last_td = airbin2mem(infile, bd_ptr, (uint32_t *)bd_paddr, bram_ptr,
                                paddr, size.start_col);

  // Send configuration packet
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt =
      reinterpret_cast<dispatch_packet_t *>(q->base_address_vaddr) + packet_id;
  air_packet_cdma_configure(pkt, last_td, uint64_t(bd_paddr), 0xffffffff,
                            &size);

  // struct timespec ts_start;
  // struct timespec ts_end;
  // clock_gettime(CLOCK_BOOTTIME, &ts_start);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);
  // clock_gettime(CLOCK_BOOTTIME, &ts_end);

  // auto time_spec_diff = [](struct timespec &start, struct timespec &end) {
  //  return (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
  //};

  // printf("config time: %0.8f sec\n", time_spec_diff(ts_start, ts_end));
  return 0;
}

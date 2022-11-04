//===- air_host.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_HOST_H
#define AIR_HOST_H

#include "air_network.h"
#include "air_queue.h"
#include "air_tensor.h"
#include "hsa_defs.h"

#include <stdlib.h>
#include <string>

extern "C" {

#define AIR_VCK190_SHMEM_BASE 0x020100000000LL
#define AIR_VCK190_L2_DMA_BASE 0x020240000000LL
#define AIR_VCK190_DDR_BASE 0x2000LL
#ifdef AIR_PCIE
#define AIR_BBUFF_BASE 0x8001C0000LL
#else
#define AIR_BBUFF_BASE 0x81C000000LL
#endif
// library operations

typedef uint64_t air_libxaie_ctx_t;

air_libxaie_ctx_t air_init_libxaie(uint32_t device_id = 0);
void air_deinit_libxaie(air_libxaie_ctx_t);

// agent operations
//

typedef struct air_agent_s {
  uint64_t handle;
} air_agent_t;

hsa_status_t air_iterate_agents(hsa_status_t (*callback)(air_agent_t agent,
                                                         void *data),
                                void *data);
hsa_status_t air_get_agent_info(queue_t *queue, air_agent_info_t attribute,
                                void *data);

#ifdef AIR_PCIE
hsa_status_t air_get_physical_devices();
#endif

// device memory operations
//

/* Initializes the device memory allocator. On initialization,
the device memory allocator opens the DDR BAR of the device with the
corresponding device_id, which is the pool of memory that gets allocated.
Once initialized, the runtime creates a handle named `dev_mem_allocator`
which it can use to service calls the process makes to air_dev_mem_alloc() */
int air_init_dev_mem_allocator(uint64_t dev_mem_size, uint32_t device_id = 0);

/* Frees the handle to the device memory allocator held by the runtime */
void air_dev_mem_allocator_free();

/* Interface for the process to request allocations and obtain pointers
to device memory. Device memory is managed as a stack, where the allocator
keeps track of the current stack pointer, and allocates a `size`-byte region
of memory backed by the device DDR BAR and returns the virtual address to that
region. */
void *air_dev_mem_alloc(uint32_t size);

/* Used to obtain the physical address of a buffer allocated using the
device memory allocator. */
uint64_t air_dev_mem_get_pa(void *buff_va);

// memory operations
//

void *air_mem_alloc(size_t size);
int air_mem_free(void *vaddr, size_t size);

uint64_t air_mem_get_paddr(void *vaddr);

// queue operations
//

hsa_status_t air_queue_create(uint32_t size, uint32_t type, queue_t **queue,
                              uint64_t paddr, uint32_t device_id = 0);

hsa_status_t air_queue_dispatch(queue_t *queue, uint64_t doorbell,
                                dispatch_packet_t *pkt);
hsa_status_t air_queue_wait(queue_t *queue, dispatch_packet_t *pkt);
hsa_status_t air_queue_dispatch_and_wait(queue_t *queue, uint64_t doorbell,
                                         dispatch_packet_t *pkt);

// packet utilities
//

struct l2_dma_cmd_t {
  uint8_t select;
  uint16_t length;
  uint16_t uram_addr;
  uint8_t id;
};

struct l2_dma_rsp_t {
  uint8_t id;
};

// initialize pkt as a herd init packet with given parameters
hsa_status_t air_packet_herd_init(dispatch_packet_t *pkt, uint16_t herd_id,
                                  uint8_t start_col, uint8_t num_cols,
                                  uint8_t start_row, uint8_t num_rows);
// uint8_t start_row, uint8_t num_rows,
// uint16_t dma0, uint16_t dma1);

hsa_status_t air_packet_device_init(dispatch_packet_t *pkt, uint32_t num_cols);

hsa_status_t air_packet_get_capabilities(dispatch_packet_t *pkt,
                                         uint64_t return_address);

hsa_status_t air_packet_hello(dispatch_packet_t *pkt, uint64_t value);

// RDMA operations for the AIR Scale out platform
hsa_status_t air_packet_post_rdma_wqe(dispatch_packet_t *pkt,
                                      uint64_t remote_vaddr,
                                      uint64_t local_paddr, uint32_t length,
                                      uint8_t op, uint8_t key, uint8_t qpid,
                                      uint8_t ernic_sel);
hsa_status_t air_packet_post_rdma_recv(dispatch_packet_t *pkt,
                                       uint64_t local_paddr, uint32_t length,
                                       uint8_t qpid, uint8_t ernic_sel);

hsa_status_t air_packet_put_stream(dispatch_packet_t *pkt, uint64_t stream,
                                   uint64_t value);
hsa_status_t air_packet_get_stream(dispatch_packet_t *pkt, uint64_t stream);

hsa_status_t air_packet_l2_dma(dispatch_packet_t *pkt, uint64_t stream,
                               l2_dma_cmd_t cmd);

hsa_status_t air_packet_cdma_memcpy(dispatch_packet_t *pkt, uint64_t dest,
                                    uint64_t source, uint32_t length);

hsa_status_t air_packet_cdma_configure(dispatch_packet_t *pkt, uint64_t dest,
                                       uint64_t source, uint32_t length);

hsa_status_t air_packet_aie_lock_range(dispatch_packet_t *pkt, uint16_t herd_id,
                                       uint64_t lock_id, uint64_t acq_rel,
                                       uint64_t value, uint8_t start_col,
                                       uint8_t num_cols, uint8_t start_row,
                                       uint8_t num_rows);

hsa_status_t air_packet_aie_lock(dispatch_packet_t *pkt, uint16_t herd_id,
                                 uint64_t lock_id, uint64_t acq_rel,
                                 uint64_t value, uint8_t col, uint8_t row);

hsa_status_t air_packet_tile_status(dispatch_packet_t *pkt, uint8_t col,
                                    uint8_t row);
hsa_status_t air_packet_dma_status(dispatch_packet_t *pkt, uint8_t col,
                                   uint8_t row);
hsa_status_t air_packet_shimdma_status(dispatch_packet_t *pkt, uint8_t col);

hsa_status_t
air_packet_nd_memcpy(dispatch_packet_t *pkt, uint16_t herd_id, uint8_t col,
                     uint8_t direction, uint8_t channel, uint8_t burst_len,
                     uint8_t memory_space, uint64_t phys_addr,
                     uint32_t transfer_length1d, uint32_t transfer_length2d,
                     uint32_t transfer_stride2d, uint32_t transfer_length3d,
                     uint32_t transfer_stride3d, uint32_t transfer_length4d,
                     uint32_t transfer_stride4d);

hsa_status_t air_packet_barrier_and(barrier_and_packet_t *pkt,
                                    uint64_t dep_signal0, uint64_t dep_signal1,
                                    uint64_t dep_signal2, uint64_t dep_signal3,
                                    uint64_t dep_signal4);
hsa_status_t air_packet_barrier_or(barrier_or_packet_t *pkt,
                                   uint64_t dep_signal0, uint64_t dep_signal1,
                                   uint64_t dep_signal2, uint64_t dep_signal3,
                                   uint64_t dep_signal4);

// herd descriptors generated by compiler
//

struct air_herd_shim_desc_t {
  int64_t *location_data;
  int64_t *channel_data;
};

struct air_herd_desc_t {
  int64_t name_length;
  char *name;
  air_herd_shim_desc_t *shim_desc;
};

struct air_rt_herd_desc_t {
  queue_t *q;
  air_herd_desc_t *herd_desc;
};

struct air_partition_desc_t {
  int64_t name_length;
  char *name;
  uint64_t herd_length;
  air_herd_desc_t **herd_descs;
};

struct air_rt_partition_desc_t {
  queue_t *q;
  air_partition_desc_t *partition_desc;
};

struct air_module_desc_t {
  uint64_t partition_length;
  air_partition_desc_t **partition_descs;
};

// AIR module shared library helpers
//

typedef size_t air_module_handle_t;

// return 0 on failure, nonzero otherwise
air_module_handle_t air_module_load_from_file(const char *filename,
                                              queue_t *q = 0,
                                              uint32_t device_id = 0);

// return 0 on success, nonzero otherwise
int32_t air_module_unload(air_module_handle_t handle);

air_module_desc_t *air_module_get_desc(air_module_handle_t handle);

air_partition_desc_t *air_partition_get_desc(air_module_handle_t handle,
                                             const char *name);

air_herd_desc_t *air_herd_get_desc(air_module_handle_t handle,
                                   air_partition_desc_t *partition,
                                   const char *name);

uint64_t air_partition_load(const char *name);

uint64_t air_herd_load(const char *name);
}

std::string air_get_ddr_bar(uint32_t device_id);
std::string air_get_aie_bar(uint32_t device_id);
std::string air_get_bram_bar(uint32_t device_id);

#endif // AIR_HOST_H

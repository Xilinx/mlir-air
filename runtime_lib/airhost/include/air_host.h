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
#include "hsa/hsa.h"
#include "hsa_ext_air.h"

#include <stdlib.h>
#include <string>
#include <vector>

extern "C" {

// init/deinit
//

hsa_status_t air_init();
hsa_status_t air_shut_down();

void *air_malloc(size_t size);
void air_free(void *mem);

// libxaie context operations
//

typedef uint64_t air_libxaie_ctx_t;

air_libxaie_ctx_t air_init_libxaie(uint32_t device_id = 0);
air_libxaie_ctx_t air_get_libxaie_ctx();
void air_deinit_libxaie(air_libxaie_ctx_t);

// debug operations
//

uint64_t air_get_tile_addr(uint32_t col, uint32_t row);
uint32_t air_read32(uint64_t addr);
void air_write32(uint64_t addr, uint32_t val);

// agent operations
//

hsa_status_t air_get_agent_info(hsa_agent_t *agent, hsa_queue_t *queue,
                                hsa_air_agent_info_t attribute, void *data);

// initialize pkt as a segment init packet with given parameters
hsa_status_t air_packet_segment_init(hsa_agent_dispatch_packet_t *pkt,
                                     uint16_t herd_id, uint8_t start_col,
                                     uint8_t num_cols, uint8_t start_row,
                                     uint8_t num_rows);

hsa_status_t air_packet_device_init(hsa_agent_dispatch_packet_t *pkt,
                                    uint32_t num_cols);

hsa_status_t air_packet_get_capabilities(hsa_agent_dispatch_packet_t *pkt,
                                         uint64_t return_address);

hsa_status_t air_packet_hello(hsa_agent_dispatch_packet_t *pkt, uint64_t value);

// RDMA operations for the AIR Scale out platform
hsa_status_t air_packet_post_rdma_wqe(hsa_agent_dispatch_packet_t *pkt,
                                      uint64_t remote_vaddr,
                                      uint64_t local_paddr, uint32_t length,
                                      uint8_t op, uint8_t key, uint8_t qpid,
                                      uint8_t ernic_sel);
hsa_status_t air_packet_post_rdma_recv(hsa_agent_dispatch_packet_t *pkt,
                                       uint64_t local_paddr, uint32_t length,
                                       uint8_t qpid, uint8_t ernic_sel);

hsa_status_t air_packet_put_stream(hsa_agent_dispatch_packet_t *pkt,
                                   uint64_t stream, uint64_t value);
hsa_status_t air_packet_get_stream(hsa_agent_dispatch_packet_t *pkt,
                                   uint64_t stream);

hsa_status_t air_packet_cdma_memcpy(hsa_agent_dispatch_packet_t *pkt,
                                    uint64_t dest, uint64_t source,
                                    uint32_t length);

hsa_status_t air_packet_cdma_configure(hsa_agent_dispatch_packet_t *pkt,
                                       uint64_t dest, uint64_t source,
                                       uint32_t length);

hsa_status_t air_packet_aie_lock_range(hsa_agent_dispatch_packet_t *pkt,
                                       uint16_t herd_id, uint64_t lock_id,
                                       uint64_t acq_rel, uint64_t value,
                                       uint8_t start_col, uint8_t num_cols,
                                       uint8_t start_row, uint8_t num_rows);

hsa_status_t air_packet_aie_lock(hsa_agent_dispatch_packet_t *pkt,
                                 uint16_t herd_id, uint64_t lock_id,
                                 uint64_t acq_rel, uint64_t value, uint8_t col,
                                 uint8_t row);

hsa_status_t air_packet_tile_status(hsa_agent_dispatch_packet_t *pkt,
                                    uint8_t col, uint8_t row);
hsa_status_t air_packet_dma_status(hsa_agent_dispatch_packet_t *pkt,
                                   uint8_t col, uint8_t row);
hsa_status_t air_packet_shimdma_status(hsa_agent_dispatch_packet_t *pkt,
                                       uint8_t col);

hsa_status_t
air_packet_nd_memcpy(hsa_agent_dispatch_packet_t *pkt, uint16_t herd_id,
                     uint8_t col, uint8_t direction, uint8_t channel,
                     uint8_t burst_len, uint8_t memory_space,
                     uint64_t phys_addr, uint32_t transfer_length1d,
                     uint32_t transfer_length2d, uint32_t transfer_stride2d,
                     uint32_t transfer_length3d, uint32_t transfer_stride3d,
                     uint32_t transfer_length4d, uint32_t transfer_stride4d);

hsa_status_t
air_packet_barrier_and(hsa_barrier_and_packet_t *pkt, hsa_signal_t dep_signal0,
                       hsa_signal_t dep_signal1, hsa_signal_t dep_signal2,
                       hsa_signal_t dep_signal3, hsa_signal_t dep_signal4);
hsa_status_t
air_packet_barrier_or(hsa_barrier_or_packet_t *pkt, hsa_signal_t dep_signal0,
                      hsa_signal_t dep_signal1, hsa_signal_t dep_signal2,
                      hsa_signal_t dep_signal3, hsa_signal_t dep_signal4);

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
  hsa_queue_t *q;
  hsa_agent_t *agent;
  air_herd_desc_t *herd_desc;
};

struct air_segment_desc_t {
  int64_t name_length;
  char *name;
  uint64_t herd_length;
  air_herd_desc_t **herd_descs;
};

struct air_rt_segment_desc_t {
  hsa_queue_t *q;
  hsa_agent_t *agent;
  air_segment_desc_t *segment_desc;
};

struct air_module_desc_t {
  uint64_t segment_length;
  air_segment_desc_t **segment_descs;
};

// AIR module shared library helpers
//

typedef size_t air_module_handle_t;

// return 0 on failure, nonzero otherwise
air_module_handle_t air_module_load_from_file(const char *filename,
                                              hsa_agent_t *agent = 0,
                                              hsa_queue_t *q = 0,
                                              uint32_t device_id = 0);

// return 0 on success, nonzero otherwise
int32_t air_module_unload(air_module_handle_t handle);

air_module_desc_t *air_module_get_desc(air_module_handle_t handle);

air_segment_desc_t *air_segment_get_desc(air_module_handle_t handle,
                                         const char *name);

air_herd_desc_t *air_herd_get_desc(air_module_handle_t handle,
                                   air_segment_desc_t *segment,
                                   const char *name);

uint64_t air_segment_load(const char *name);

uint64_t air_herd_load(const char *name);
}

// queue operations
//

hsa_status_t air_queue_dispatch(hsa_queue_t *queue, uint64_t packet_id,
                                uint64_t doorbell,
                                hsa_agent_dispatch_packet_t *pkt);
hsa_status_t air_queue_wait(hsa_queue_t *queue,
                            hsa_agent_dispatch_packet_t *pkt);
hsa_status_t air_queue_dispatch_and_wait(hsa_agent_t *agent, hsa_queue_t *q,
                                         uint64_t packet_id, uint64_t doorbell,
                                         hsa_agent_dispatch_packet_t *pkt,
                                         bool destroy_signal = true);

// TODO: Remove this duplicaiton and use C++ templates, need to move some things
// around for that
hsa_status_t air_queue_dispatch(hsa_queue_t *queue, uint64_t packet_id,
                                uint64_t doorbell,
                                hsa_barrier_and_packet_t *pkt);
hsa_status_t air_queue_wait(hsa_queue_t *queue, hsa_barrier_and_packet_t *pkt);
hsa_status_t air_queue_dispatch_and_wait(hsa_agent_t *agent, hsa_queue_t *queue,
                                         uint64_t packet_id, uint64_t doorbell,
                                         hsa_barrier_and_packet_t *pkt,
                                         bool destroy_signal = true);

hsa_status_t find_aie(hsa_agent_t agent, void *data);
hsa_status_t air_get_agents(std::vector<hsa_agent_t> &agents);

#endif // AIR_HOST_H

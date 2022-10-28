//===- air_queue.h ---------------------------------------------*- C++ -*-===//
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

#ifndef ACDC_QUEUE_H
#define ACDC_QUEUE_H

#include "hsa_defs.h"
#include <stdint.h>

#include <stdint.h>

// Define the number of HSA packets we can have in a queue
#define MB_QUEUE_SIZE 48

// Define the amount of shared memory accessible to each controller
// This includes the queues, events, doorbells etc
#define MB_SHMEM_SEGMENT_SIZE 0x1000

// See
// https://confluence.xilinx.com/display/XRLABS/AIR+Controller+HSA+Packet+Formats
// All defined as longs, so we can shift them into 64 registers

#define AIR_PKT_TYPE_INVALID 0x0000L
#define AIR_PKT_TYPE_PUT_STREAM 0x0001L
#define AIR_PKT_TYPE_GET_STREAM 0x0002L
#define AIR_PKT_TYPE_SDMA_STATUS 0x0003L
#define AIR_PKT_TYPE_TDMA_STATUS 0x0004L
#define AIR_PKT_TYPE_CORE_STATUS 0x0005L

#define AIR_PKT_TYPE_DEVICE_INITIALIZE 0x0010L
#define AIR_PKT_TYPE_HERD_INITIALIZE 0x0011L
#define AIR_PKT_TYPE_HELLO 0x0012L
#define AIR_PKT_TYPE_ALLOCATE_HERD_SHIM_DMAS 0x0013L
#define AIR_PKT_TYPE_GET_CAPABILITIES 0x0014L
#define AIR_PKT_TYPE_GET_INFO 0x0015L

#define AIR_PKT_TYPE_XAIE_LOCK 0x0020L

#define AIR_PKT_TYPE_CDMA 0x030L
#define AIR_PKT_TYPE_CONFIGURE 0x031L

#define AIR_PKT_TYPE_POST_RDMA_WQE 0x040L
#define AIR_PKT_TYPE_POST_RDMA_RECV 0x041L

#define AIR_PKT_TYPE_SHIM_DMA_MEMCPY 0x0100L
#define AIR_PKT_TYPE_HERD_SHIM_DMA_MEMCPY 0x0101L
#define AIR_PKT_TYPE_HERD_SHIM_DMA_1D_STRIDED_MEMCPY 0x0102L
#define AIR_PKT_TYPE_ND_MEMCPY 0x0103L

#define AIR_ADDRESS_ABSOLUTE 0x0L
#define AIR_ADDRESS_ABSOLUTE_RANGE 0x1L
#define AIR_ADDRESS_HERD_RELATIVE 0x2L
#define AIR_ADDRESS_HERD_RELATIVE_RANGE 0x3L

// Note below that "__attribute__((packed))" also asserts that the whole
// structure is unaligned in some compilers.  This helps to silence errors from
// -waddress-of-packed-struct

typedef struct dispatch_packet_s {

  // HSA-like interface
  volatile uint16_t header;
  volatile uint16_t type;
  uint32_t reserved0;
  uint64_t return_address;
  uint64_t arg[4];
  uint64_t reserved1;
  uint64_t completion_signal;

} __attribute__((packed, aligned(__alignof__(uint64_t)))) dispatch_packet_t;

typedef struct barrier_and_packet_s {

  // HSA-like interface
  volatile uint16_t header;
  uint16_t reserved0;
  uint32_t reserved1;
  uint64_t dep_signal[5];
  uint64_t reserved2;
  uint64_t completion_signal;

} __attribute__((packed, aligned(__alignof__(uint64_t)))) barrier_and_packet_t;

typedef struct barrier_or_packet_s {

  // HSA-like interface
  volatile uint16_t header;
  uint16_t reserved0;
  uint32_t reserved1;
  uint64_t dep_signal[5];
  uint64_t reserved2;
  uint64_t completion_signal;

} __attribute__((packed, aligned(__alignof__(uint64_t)))) barrier_or_packet_t;

typedef struct queue_s {

  // HSA-like interface
  uint32_t type;
  uint32_t features;
  uint64_t base_address;
  volatile uint64_t doorbell;
  uint32_t size;
  uint32_t reserved0;
  uint64_t id;

  // implementation detail
  uint64_t read_index;
  uint64_t write_index;
  uint64_t last_doorbell;

  uint64_t base_address_paddr;
  uint64_t base_address_vaddr;

} __attribute__((packed, aligned(__alignof__(uint64_t)))) queue_t;

typedef struct signal_s {
  uint64_t handle;
} signal_t;

typedef uint64_t signal_value_t;

namespace {

inline uint64_t queue_add_write_index(queue_t *q, uint64_t v) {
  auto r = q->write_index;
  q->write_index = r + v;
  return r;
}

inline uint64_t queue_add_read_index(queue_t *q, uint64_t v) {
  auto r = q->read_index;
  q->read_index = r + v;
  return r;
}

inline uint64_t queue_load_read_index(queue_t *q) { return q->read_index; }

inline uint64_t queue_load_write_index(queue_t *q) { return q->write_index; }

inline uint64_t queue_paddr_from_index(queue_t *q, uint64_t idx) {
  return q->base_address + idx;
}

inline bool packet_get_active(dispatch_packet_t *pkt) {
  return pkt->reserved1 & 0x1;
}

inline void packet_set_active(dispatch_packet_t *pkt, bool b) {
  pkt->reserved1 = (pkt->reserved1 & ~0x1) | b;
}

inline void initialize_packet(dispatch_packet_t *pkt) {
  pkt->header = HSA_PACKET_TYPE_INVALID;
  // pkt->type = AIR_PKT_TYPE_INVALID;
}

inline hsa_status_t signal_create(signal_value_t initial_value,
                                  uint32_t num_consumers, void *consumers,
                                  signal_t *signal) {
  // auto s = (signal_value_t*)malloc(sizeof(signal_value_t));
  //*s = initial_value;
  signal->handle = (uint64_t)initial_value;
  return HSA_STATUS_SUCCESS;
}

inline hsa_status_t signal_destroy(signal_t signal) {
  // free((void*)signal.handle);
  return HSA_STATUS_SUCCESS;
}

inline void signal_store_release(signal_t *signal, signal_value_t value) {
  signal->handle = (uint64_t)value;
}

inline signal_value_t signal_wait_acquire(volatile signal_t *signal,
                                          hsa_signal_condition_t condition,
                                          signal_value_t compare_value,
                                          uint64_t timeout_hint,
                                          hsa_wait_state_t wait_state_hint) {
  signal_value_t ret = 0;
  uint64_t timeout = timeout_hint;
  do {
    ret = signal->handle;
    if (ret == compare_value)
      return compare_value;
  } while (timeout--);
  return ret;
}

inline void signal_subtract_acq_rel(signal_t *signal, signal_value_t value) {
  signal->handle = signal->handle - (uint64_t)value;
  // uint64_t i;
  // memcpy((void*)&i, (void*)signal, sizeof(signal_value_t));
  // i = i - (uint64_t)value;
  // memcpy((void*)signal, (void*)&i, sizeof(signal_value_t));
}

} // namespace

#endif

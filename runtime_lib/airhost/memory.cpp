//===- memory.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air.hpp"
#include "air_host.h"
#include "air_host_impl.h"
#include "pcie-ernic.h"
#include "runtime.h"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fcntl.h>    /* for open() */
#include <string.h>   /* for memset() */
#include <sys/mman.h> /* for mlock() */
#include <unistd.h>   /* for getpagesize() */
#include <vector>

extern "C" {

extern air_rt_herd_desc_t _air_host_active_herd;
extern aie_libxaie_ctx_t *_air_host_active_libxaie;
extern uint32_t *_air_host_bram_ptr;
extern uint64_t _air_host_bram_paddr;
}

void *air_malloc(size_t size) {
  void *mem(air::rocm::Runtime::runtime_->AllocateMemory(size));
  return mem;
}

void air_free(void *mem) { air::rocm::Runtime::runtime_->FreeMemory(mem); }

// Data structure internal to the runtime to map air tensors
// to the information to access a remote buffer
extern std::map<void *, tensor_to_qp_map_entry *> tensor_to_qp_map;

static int64_t shim_location_data(air_herd_shim_desc_t *sd, int i, int j,
                                  int k) {
  return sd->location_data[i * 8 * 8 + j * 8 + k];
}

static int64_t shim_channel_data(air_herd_shim_desc_t *sd, int i, int j,
                                 int k) {
  return sd->channel_data[i * 8 * 8 + j * 8 + k];
}

template <typename T, int R>
static void air_mem_shim_nd_memcpy_queue_impl(
    hsa_signal_t *s, uint32_t id, uint64_t x, uint64_t y, tensor_t<T, R> *t,
    uint32_t space, uint64_t offset_3, uint64_t offset_2, uint64_t offset_1,
    uint64_t offset_0, uint64_t length_4d, uint64_t length_3d,
    uint64_t length_2d, uint64_t length_1d, uint64_t stride_4d,
    uint64_t stride_3d, uint64_t stride_2d) {
  assert(_air_host_active_herd.herd_desc &&
         "cannot shim memcpy without active herd");
  assert(_air_host_active_herd.q &&
         "cannot shim memcpy using a queue without active queue");
  assert(_air_host_active_herd.agent &&
         "cannot shim memcpy using an agent without an active agent");

  auto shim_desc = _air_host_active_herd.herd_desc->shim_desc;
  auto shim_col = shim_location_data(shim_desc, id - 1, x, y);
  auto shim_chan = shim_channel_data(shim_desc, id - 1, x, y);

  // printf("Do transfer %p with id %d on behalf of x=%ld, y=%ld space %d, col
  // %d, dir %d, chan %d, offset [%ld,%ld,%ld,%ld], length [%ld,%ld,%ld,%ld],
  // stride [%ld,%ld,%ld]\n",
  //       t->data, id, x, y, space, shim_col, shim_chan>=2, (shim_chan>=2) ?
  //       shim_chan-2 : shim_chan, offset_3, offset_2, offset_1, offset_0,
  //       length_4d, length_3d, length_2d, length_1d,
  //       stride_4d, stride_3d, stride_2d);

  // Checking our internal representation to determine if the buffer is
  // remote or local. If we don't find anything in our map, we say it
  // is local
  struct tensor_to_qp_map_entry *rdma_entry = tensor_to_qp_map[t->alloc];
  bool is_local = true;
  if (rdma_entry != NULL) {
    is_local = (rdma_entry->qp == 0);
  } else {
    is_local = true; // If not in our map make it local
  }

  bool isMM2S = shim_chan >= 2;

  bool uses_pa = (space == 1); // t->uses_pa;
  if (uses_pa) {
    if (isMM2S)
      shim_chan = shim_chan - 2;

    size_t stride = 1;
    size_t offset = 0;
    std::vector<uint64_t> offsets{offset_0, offset_1, offset_2, offset_3};
    for (int i = 0; i < R; i++) {
      offset += offsets[i] * stride * sizeof(T);
      stride *= t->shape[R - i - 1];
    }

    uint64_t wr_idx =
        hsa_queue_add_write_index_relaxed(_air_host_active_herd.q, 1);
    uint64_t packet_id = wr_idx % _air_host_active_herd.q->size;

    hsa_agent_dispatch_packet_t pkt;

    air_packet_nd_memcpy(
        &pkt, /*herd_id=*/0, shim_col, /*direction=*/isMM2S, shim_chan,
        /*burst_len=*/4, /*memory_space=*/space, (uint64_t)t->data + offset,
        length_1d * sizeof(T), length_2d, stride_2d * sizeof(T), length_3d,
        stride_3d * sizeof(T), length_4d, stride_4d * sizeof(T));

    // TODO: Right now we don't have a way of knowing when we can destroy these
    // signals
    if (s) {
      // Fire off the packet
      hsa_amd_signal_create_on_agent(1, 0, nullptr, _air_host_active_herd.agent,
                                     0, &pkt.completion_signal);
      air_queue_dispatch(_air_host_active_herd.q, packet_id, wr_idx, &pkt);

      // Set the signal that we were passed in equal to the completion signal
      s->handle = pkt.completion_signal.handle;
    } else {
      air_queue_dispatch_and_wait(_air_host_active_herd.agent,
                                  _air_host_active_herd.q, packet_id, wr_idx,
                                  &pkt);
    }
    return;
  } else {
    uint32_t *bounce_buffer = _air_host_bram_ptr;

    // Only used for RDMA requests
    uint64_t bounce_buffer_pa = _air_host_bram_paddr;

    size_t stride = 1;
    size_t offset = 0;
    std::vector<uint64_t> offsets{offset_0, offset_1, offset_2, offset_3};
    for (int i = 0; i < R; i++) {
      offset += offsets[i] * stride * sizeof(T);
      stride *= t->shape[R - i - 1];
    }

    uint64_t length = 0;
    for (uint32_t index_4d = 0; index_4d < length_4d; index_4d++)
      for (uint32_t index_3d = 0; index_3d < length_3d; index_3d++)
        for (uint32_t index_2d = 0; index_2d < length_2d; index_2d++)
          length += length_1d;

    size_t p;

    // Setting the virtual address depending on
    // if the buffer is local or remote
    if (is_local) {
      p = (size_t)t->data + offset;
    } else {
      p = (size_t)rdma_entry->vaddr + offset;
    }

    uint64_t paddr_4d = p;
    uint64_t paddr_3d = p;
    uint64_t paddr_2d = p;
    uint64_t paddr_1d = p;

    uint64_t wr_idx, packet_id;
    hsa_agent_dispatch_packet_t rdma_read_pkt;

    if (isMM2S) {
      shim_chan = shim_chan - 2;
      for (uint32_t index_4d = 0; index_4d < length_4d; index_4d++) {
        paddr_2d = paddr_3d;
        for (uint32_t index_3d = 0; index_3d < length_3d; index_3d++) {
          paddr_1d = paddr_2d;
          for (uint32_t index_2d = 0; index_2d < length_2d; index_2d++) {
            if (is_local) {
              memcpy((size_t *)bounce_buffer, (size_t *)paddr_1d,
                     length_1d * sizeof(T));
            } else {
              wr_idx =
                  hsa_queue_add_write_index_relaxed(_air_host_active_herd.q, 1);
              packet_id = wr_idx % _air_host_active_herd.q->size;
              air_packet_post_rdma_wqe(
                  &rdma_read_pkt, (uint64_t)paddr_1d,
                  (uint64_t)bounce_buffer_pa, (uint32_t)length_1d * sizeof(T),
                  (uint8_t)OP_READ, (uint8_t)rdma_entry->rkey,
                  (uint8_t)rdma_entry->qp, (uint8_t)0);

              // air_write_pkt<hsa_agent_dispatch_packet_t>(_air_host_active_herd.q,
              // packet_id, &rdma_read_pkt);
              air_queue_dispatch_and_wait(_air_host_active_herd.agent,
                                          _air_host_active_herd.q, packet_id,
                                          wr_idx, &rdma_read_pkt);
            }

            // Update physical address of the bounce buffer we are writing to
            bounce_buffer_pa += length_1d * sizeof(T);
            bounce_buffer += length_1d;
            paddr_1d += stride_2d * sizeof(T);
          }
          paddr_2d += stride_3d * sizeof(T);
        }
        paddr_3d += stride_4d * sizeof(T);
      }
    }

    wr_idx = hsa_queue_add_write_index_relaxed(_air_host_active_herd.q, 1);
    packet_id = wr_idx % _air_host_active_herd.q->size;

    hsa_agent_dispatch_packet_t memcpy_pkt;
    air_packet_nd_memcpy(
        &memcpy_pkt, /*herd_id=*/0, shim_col, /*direction=*/isMM2S, shim_chan,
        /*burst_len=*/4, /*memory_space=*/2,
        /*_air_host_bram_paddr*/ reinterpret_cast<uint64_t>(_air_host_bram_ptr),
        length * sizeof(T), 1, 0, 1, 0, 1, 0);

    // TODO: Right now we don't have a way of knowing when we can destroy these
    // signals
    if (s) {
      // Fire off the packet
      // TODO: Don't wait here
      air_queue_dispatch_and_wait(_air_host_active_herd.agent,
                                  _air_host_active_herd.q, packet_id, wr_idx,
                                  &memcpy_pkt, false);

      // Having the signal that we were passed point to the same signal value
      s->handle = memcpy_pkt.completion_signal.handle;
    } else {
      air_queue_dispatch_and_wait(_air_host_active_herd.agent,
                                  _air_host_active_herd.q, packet_id, wr_idx,
                                  &memcpy_pkt);
    }

    if (!isMM2S) {
      for (uint32_t index_4d = 0; index_4d < length_4d; index_4d++) {
        paddr_2d = paddr_3d;
        for (uint32_t index_3d = 0; index_3d < length_3d; index_3d++) {
          paddr_1d = paddr_2d;
          for (uint32_t index_2d = 0; index_2d < length_2d; index_2d++) {
            if (is_local) {
              memcpy((size_t *)paddr_1d, (size_t *)bounce_buffer,
                     length_1d * sizeof(T));
            } else {
              wr_idx =
                  hsa_queue_add_write_index_relaxed(_air_host_active_herd.q, 1);
              packet_id = wr_idx % _air_host_active_herd.q->size;
              hsa_agent_dispatch_packet_t rdma_write_pkt;

              air_packet_post_rdma_wqe(
                  &rdma_write_pkt, (uint64_t)paddr_1d,
                  (uint64_t)bounce_buffer_pa, (uint32_t)length_1d * sizeof(T),
                  (uint8_t)OP_WRITE, (uint8_t)rdma_entry->rkey,
                  (uint8_t)rdma_entry->qp, (uint8_t)0);

              // air_write_pkt<hsa_agent_dispatch_packet_t>(_air_host_active_herd.q,
              // packet_id, &rdma_write_pkt);
              air_queue_dispatch_and_wait(_air_host_active_herd.agent,
                                          _air_host_active_herd.q, packet_id,
                                          wr_idx, &rdma_write_pkt);
            }

            bounce_buffer_pa += length_1d * sizeof(T);
            bounce_buffer += length_1d;
            paddr_1d += stride_2d * sizeof(T);
          }
          paddr_2d += stride_3d * sizeof(T);
        }
        paddr_3d += stride_4d * sizeof(T);
      }
    }
  }
}

#define mlir_air_dma_nd_memcpy(mangle, rank, space, type)                      \
  void _mlir_ciface___airrt_dma_nd_memcpy_##mangle(                            \
      hsa_signal_t *s, uint32_t id, uint64_t x, uint64_t y, void *t,           \
      uint64_t offset_3, uint64_t offset_2, uint64_t offset_1,                 \
      uint64_t offset_0, uint64_t length_3, uint64_t length_2,                 \
      uint64_t length_1, uint64_t length_0, uint64_t stride_2,                 \
      uint64_t stride_1, uint64_t stride_0) {                                  \
    tensor_t<type, rank> *tt = (tensor_t<type, rank> *)t;                      \
    if (_air_host_active_herd.q) {                                             \
      air_mem_shim_nd_memcpy_queue_impl(                                       \
          s, id, x, y, tt, space, offset_3, offset_2, offset_1, offset_0,      \
          length_3, length_2, length_1, length_0, stride_2, stride_1,          \
          stride_0);                                                           \
    } else {                                                                   \
      printf(                                                                  \
          "WARNING: no queue provided. ND memcpy will not be performed.\n");   \
    }                                                                          \
  }

extern "C" {

mlir_air_dma_nd_memcpy(1d0i32, 1, 2, uint32_t);
mlir_air_dma_nd_memcpy(2d0i32, 2, 2, uint32_t);
mlir_air_dma_nd_memcpy(3d0i32, 3, 2, uint32_t);
mlir_air_dma_nd_memcpy(4d0i32, 4, 2, uint32_t);
mlir_air_dma_nd_memcpy(1d0f32, 1, 2, float);
mlir_air_dma_nd_memcpy(2d0f32, 2, 2, float);
mlir_air_dma_nd_memcpy(3d0f32, 3, 2, float);
mlir_air_dma_nd_memcpy(4d0f32, 4, 2, float);
mlir_air_dma_nd_memcpy(1d1i32, 1, 1, uint32_t);
mlir_air_dma_nd_memcpy(2d1i32, 2, 1, uint32_t);
mlir_air_dma_nd_memcpy(3d1i32, 3, 1, uint32_t);
mlir_air_dma_nd_memcpy(4d1i32, 4, 1, uint32_t);
mlir_air_dma_nd_memcpy(1d1f32, 1, 1, float);
mlir_air_dma_nd_memcpy(2d1f32, 2, 1, float);
mlir_air_dma_nd_memcpy(3d1f32, 3, 1, float);
mlir_air_dma_nd_memcpy(4d1f32, 4, 1, float);

} // extern "C"

#define mlir_air_nd_memcpy(mangle, rank0, space0, type0, rank1, space1, type1) \
  void _mlir_ciface___airrt_nd_memcpy_##mangle(                                \
      void *t0, void *t1, uint64_t offset_3, uint64_t offset_2,                \
      uint64_t offset_1, uint64_t offset_0, uint64_t length_3,                 \
      uint64_t length_2, uint64_t length_1, uint64_t length_0,                 \
      uint64_t stride_2, uint64_t stride_1, uint64_t stride_0) {               \
    tensor_t<type0, rank0> *tt0 = (tensor_t<type0, rank0> *)t0;                \
    tensor_t<type1, rank1> *tt1 = (tensor_t<type1, rank1> *)t1;                \
    if (_air_host_active_herd.q) {                                             \
      printf("WARNING: ND memcpy will not be performed.\n");                   \
    } else {                                                                   \
      printf(                                                                  \
          "WARNING: no queue provided. ND memcpy will not be performed.\n");   \
    }                                                                          \
  }

extern "C" {

mlir_air_nd_memcpy(1d0i32_1d1i32, 1, 2, uint32_t, 1, 1, uint32_t);
mlir_air_nd_memcpy(1d1i32_1d0i32, 1, 1, uint32_t, 1, 2, uint32_t);
mlir_air_nd_memcpy(2d0i32_2d1i32, 2, 2, uint32_t, 2, 1, uint32_t);
mlir_air_nd_memcpy(2d1i32_2d0i32, 2, 1, uint32_t, 2, 2, uint32_t);
mlir_air_nd_memcpy(3d0i32_3d1i32, 3, 2, uint32_t, 3, 1, uint32_t);
mlir_air_nd_memcpy(3d1i32_3d0i32, 3, 1, uint32_t, 3, 2, uint32_t);
mlir_air_nd_memcpy(4d0i32_4d1i32, 4, 2, uint32_t, 4, 1, uint32_t);
mlir_air_nd_memcpy(4d1i32_4d0i32, 4, 1, uint32_t, 4, 2, uint32_t);
mlir_air_nd_memcpy(1d0f32_1d1f32, 1, 2, float, 1, 1, float);
mlir_air_nd_memcpy(1d1f32_1d0f32, 1, 1, float, 1, 2, float);
mlir_air_nd_memcpy(2d0f32_2d1f32, 2, 2, float, 2, 1, float);
mlir_air_nd_memcpy(2d1f32_2d0f32, 2, 1, float, 2, 2, float);
mlir_air_nd_memcpy(3d0f32_3d1f32, 3, 2, float, 3, 1, float);
mlir_air_nd_memcpy(3d1f32_3d0f32, 3, 1, float, 3, 2, float);
mlir_air_nd_memcpy(4d0f32_4d1f32, 4, 2, float, 4, 1, float);
mlir_air_nd_memcpy(4d1f32_4d0f32, 4, 1, float, 4, 2, float);

} // extern "C"

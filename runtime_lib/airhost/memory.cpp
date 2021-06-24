#include "air_host.h"
#include "air_tensor.h"

#include <cassert>
#include <vector>
#include <cstdio>
#include <cstring>

#include <xaiengine.h>

namespace {

static std::vector<XAieLib_MemInst*> AirAllocations;

}

void* air_mem_alloc(size_t size) {
  XAieLib_MemInst *mem = XAieLib_MemAllocate(size, 0);
  if (!mem)
    return nullptr;
  AirAllocations.push_back(mem);
  void *vaddr = (void*)XAieLib_MemGetVaddr(mem);
  return vaddr;
}

void air_mem_dealloc(void *vaddr) {
  for (auto it = AirAllocations.begin(); it != AirAllocations.end(); ) {
    void *p = (void*)XAieLib_MemGetVaddr(*it);
    if (p == vaddr) {
      XAieLib_MemFree(*it);
      it = AirAllocations.erase(it);
    }
    else {
      ++it;
    }
  }
}

void *air_mem_get_paddr(void *vaddr) {
  for (auto *m : AirAllocations) {
    void *p = (void*)XAieLib_MemGetVaddr(m);
    if (p == vaddr)
      return (void*)XAieLib_MemGetPaddr(m);
  }
  return nullptr;
}

void *air_mem_get_vaddr(void *paddr) {
  for (auto *m : AirAllocations) {
    void *p = (void*)XAieLib_MemGetPaddr(m);
    if (p == paddr)
      return (void*)XAieLib_MemGetVaddr(m);
  }
  return nullptr;
}

namespace {

int64_t shim_location_data(air_herd_shim_desc_t *sd, int i, int j, int k) {
  return sd->location_data[i*16 + j*4 +k];
}

int64_t shim_channel_data(air_herd_shim_desc_t *sd, int i, int j, int k) {
  return sd->channel_data[i*16 + j*4 +k];
}

}

extern "C" {

extern air_rt_herd_desc_t _air_host_active_herd;
extern air_libxaie1_ctx_t *_air_host_active_libxaie1;
extern uint32_t *_air_host_bram_ptr;

}

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)


namespace {

void air_mem_shim_memcpy_impl(uint32_t id, uint64_t x, uint64_t y, void* t, uint64_t offset, uint64_t length)
{
  assert(_air_host_active_herd.herd_desc && "cannot shim memcpy without active herd");

  auto shim_desc = _air_host_active_herd.herd_desc->shim_desc;
  auto shim_col = shim_location_data(shim_desc, id-1, x, y);
  auto shim_chan = shim_channel_data(shim_desc, id-1, x, y);

  // printf("Do transfer with id %d of length %ld on behalf of x=%ld, y=%ld using shim DMA %ld channel %ld, offset is %ld\n",
  //        id, length, x, y, shim_col, shim_chan, offset);

  tensor_t<uint32_t,1> *tt = (tensor_t<uint32_t,1> *)t;

  uint64_t addr = (u64)AIR_VCK190_SHMEM_BASE+0x4000;
  uint32_t *bounce_buffer = _air_host_bram_ptr;
  bool isMM2S = shim_chan >= 2;

  if (isMM2S) {
    // This is the input, so we need to take what is in t and put it into the BRAM
    for (int i=0; i<length; i++) {
      bounce_buffer[i] = tt->d[offset + i];
    }
  }

  auto burstlen = 4;
  XAieDma_Shim dmaInst;
  XAieDma_ShimInitialize(&(_air_host_active_libxaie1->TileInst[shim_col][0]), &dmaInst);
  u8 bd = 1+shim_chan; // We don't really care, we just want these to be unique and none zero
  XAieDma_ShimBdSetAddr(&dmaInst, bd, HIGH_ADDR(addr), LOW_ADDR(addr), length*sizeof(uint32_t));
  XAieDma_ShimBdSetAxi(&dmaInst, bd, 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&dmaInst, bd);

  XAieDma_ShimSetStartBd((&dmaInst), shim_chan, bd);

  auto ret = XAieDma_ShimPendingBdCount(&dmaInst, shim_chan);
  if (ret)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

  XAieDma_ShimChControl((&dmaInst), shim_chan, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  auto count = 0;
  while (XAieDma_ShimPendingBdCount(&dmaInst, shim_chan)) {
    count++;
    if (!(count % 1000)) {
      printf("count %d\n",count/1000);
      if (count == 5000) break;
    }
  }
  if (!isMM2S) {
    // This is the output, so we need to take what is in the BRAM and put it into t
    //printf("Copy %ld samples to the output starting at %ld\n",length, offset);
    for (int i=0; i<length; i++) {
      tt->d[offset + i] = bounce_buffer[i];
    }
  }

}

void air_mem_shim_memcpy2d_impl(uint32_t id, uint64_t x, uint64_t y, void* t,
                                uint64_t offset_y, uint64_t offset_x,
                                uint64_t length, uint64_t stride, uint64_t elem_per_stride) {

  assert(_air_host_active_herd.herd_desc && "cannot shim memcpy without active herd");

  auto shim_desc = _air_host_active_herd.herd_desc->shim_desc;
  auto shim_col = shim_location_data(shim_desc, id-1, x, y);
  auto shim_chan = shim_channel_data(shim_desc, id-1, x, y);

  tensor_t<uint32_t,2> *tt = (tensor_t<uint32_t,2> *)t;

  // printf("Do transfer %p with id %ld of length %ld on behalf of x=%ld, y=%ld shim col %ld channel %ld, offset %ld,%ld, stride %ld, elem %ld\n",
  //         tt->d, id, length, x, y, shim_col, shim_chan, offset_y, offset_x, stride, elem_per_stride);

  uint64_t addr = (u64)AIR_VCK190_SHMEM_BASE+0x4000;
  uint32_t *bounce_buffer = _air_host_bram_ptr;
  bool isMM2S = shim_chan >= 2;

  XAieDma_Shim dmaInst;
  XAieDma_ShimInitialize(&(_air_host_active_libxaie1->TileInst[shim_col][0]), &dmaInst);

  if (isMM2S) {
    uint32_t *data_ptr = tt->d + (offset_y * tt->shape[1] + offset_x);
    uint32_t *bounce_ptr = bounce_buffer;
    for (int n=0; n<length; n+=elem_per_stride) {
      // This is the input, so we need to take what is in t and put it into the BRAM
      memcpy(bounce_ptr, data_ptr, elem_per_stride*sizeof(uint32_t));
      data_ptr += stride;
      bounce_ptr += elem_per_stride;
    }
  }

  //for (int n=0; n<length; n+=elem_per_stride) {
    uint32_t bd = shim_chan+1;
    auto burstlen = 4;
    XAieDma_ShimBdSetAddr(&dmaInst, bd, HIGH_ADDR(addr), LOW_ADDR(addr), sizeof(uint32_t) * length);
    XAieDma_ShimBdSetAxi(&dmaInst, bd, 0, burstlen, 0, 0, XAIE_ENABLE);
    XAieDma_ShimBdWrite(&dmaInst, bd);
    XAieDma_ShimSetStartBd((&dmaInst), shim_chan, bd); 

    XAieDma_ShimChControl((&dmaInst), shim_chan, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

    int count = 0;
    while (XAieDma_ShimPendingBdCount(&dmaInst, shim_chan)) {
      XAieLib_usleep(1000);
      count++;
      if (!(count % 1000)) {
        printf("%d seconds\n",count/1000);
        if (count == 2000) break;
      }
    }
    //addr += elem_per_stride;
  //}

  if (!isMM2S) {
    uint32_t *data_ptr = tt->d + (offset_y * tt->shape[1] + offset_x);
    uint32_t *bounce_ptr = bounce_buffer;
    for (int n=0; n<length; n+=elem_per_stride) {
      // This is the input, so we need to take what is in t and put it into the BRAM
      memcpy(data_ptr, bounce_ptr, elem_per_stride*sizeof(uint32_t));
      data_ptr += stride;
      bounce_ptr += elem_per_stride;
    }
  }
}

void air_shim_memcpy_queue_impl(uint32_t id, uint64_t x, uint64_t y, void* t, uint64_t offset, uint64_t length) {
  assert(_air_host_active_herd.herd_desc && "cannot shim memcpy without active herd");
  assert(_air_host_active_herd.q && "cannot shim memcpy using a queue without active queue");

  auto shim_desc = _air_host_active_herd.herd_desc->shim_desc;
  auto shim_col = shim_location_data(shim_desc, id-1, x, y);
  auto shim_chan = shim_channel_data(shim_desc, id-1, x, y);

  tensor_t<uint32_t,1> *tt = (tensor_t<uint32_t,1> *)t;
  uint32_t *bounce_buffer = _air_host_bram_ptr;
  bool isMM2S = shim_chan >= 2;

  if (isMM2S) {
    shim_chan = shim_chan - 2;
    uint32_t *data_ptr = tt->d + offset;
    // This is the input, so we need to take what is in t and put it into the BRAM
    memcpy(bounce_buffer, data_ptr, length*sizeof(uint32_t));
  }

  uint64_t wr_idx = queue_add_write_index(_air_host_active_herd.q, 1);
  uint64_t packet_id = wr_idx % _air_host_active_herd.q->size;

  dispatch_packet_t *pkt = (dispatch_packet_t*)(_air_host_active_herd.q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt, /*herd_id=*/0, shim_col, /*direction=*/isMM2S, shim_chan, /*burst_len=*/4, /*memory_space=*/2,
                       AIR_VCK190_SHMEM_BASE+0x4000, length*sizeof(uint32_t), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(_air_host_active_herd.q, wr_idx, pkt);

  if (!isMM2S) {
    uint32_t *data_ptr = tt->d + offset;
    memcpy(data_ptr, bounce_buffer, length*sizeof(uint32_t));
  }
}

}

extern "C"  {

void _mlir_ciface_air_shim_memcpy(uint32_t id, uint64_t x, uint64_t y, void* t, uint64_t offset, uint64_t length) {
  assert(_air_host_active_herd.herd_desc && "cannot shim memcpy without active herd");
  if (_air_host_active_herd.q)
    air_shim_memcpy_queue_impl(id, x, y, t, offset, length);
  else
    air_mem_shim_memcpy_impl(id, x, y, t, offset, length);
}

void _mlir_ciface_air_shim_memcpy2d(uint32_t id, uint64_t x, uint64_t y, void* t,
                                    uint64_t offset_y, uint64_t offset_x,
                                    uint64_t length, uint64_t stride, uint64_t elem_per_stride) {
  air_mem_shim_memcpy2d_impl(id, x, y, t, offset_y, offset_x, length, stride, elem_per_stride);
}

}

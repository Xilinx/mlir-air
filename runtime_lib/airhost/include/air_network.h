//===- air_network.h ---------------------------------------------*- C++ -*-===//
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

#ifndef AIR_NETWORK_H
#define AIR_NETWORK_H

#include "acdc_queue.h"
#include "hsa_defs.h"
#include "air_tensor.h"

//#include "test_library.h"

// Just putting this here for now
struct tensor_to_qp_map_entry {
  uint32_t qp;
  uint32_t rkey;
  uint64_t vaddr;
  struct pcie_ernic_buff *local_buff;
};

struct world_view_entry {
  char ip[9];
  char mac[17];
  uint32_t rank;
  uint32_t qps[128];
};

// RDMA operations and variables
hsa_status_t air_set_hostname(char hostname[100]);
hsa_status_t air_get_hostname(char hostname[100]);
hsa_status_t air_explore_world(uint32_t ernic_id, uint64_t dev_mem_offset, uint64_t bar_offset);
hsa_status_t air_ernic_free();
hsa_status_t air_ernic_mem_alloc(char buff_name[100], uint32_t size, void *t, bool register_mem);

void air_recv(signal_t *s, tensor_t<uint32_t, 1> *t, uint32_t size, uint32_t offset, uint32_t src_rank, queue_t *q, uint8_t ernic_sel);
void air_send(signal_t *s, tensor_t<uint32_t, 1> *t, uint32_t size, uint32_t offset, uint32_t dst_rank, queue_t *q, uint8_t ernic_sel);
void air_barrier(tensor_t<uint32_t, 1> *dummy_tensor, queue_t *q, uint8_t ernic_sel);



#endif

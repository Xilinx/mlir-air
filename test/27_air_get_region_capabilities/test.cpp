//===- test.cpp -------------------------------------------------*- C++ -*-===//
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

#include <cstdio>
#include <cassert>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <xaiengine.h>

#include "air_host.h"
#include "acdc_queue.h"

#define MAX_CONTROLLERS 64

int main(int argc, char *argv[])
{

  uint64_t controller_id = 0;
  uint64_t total_controllers;
  bool done = false;
  uint32_t errors = 0;

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return 1;

  while (!done) {

    // create a queue
    queue_t *q = nullptr;
    uint64_t* qaddrs = (uint64_t*)AIR_VCK190_SHMEM_BASE;
    auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, (uint64_t)&qaddrs[controller_id]);

    assert(ret == 0 && "failed to create queue!");

    // reserve a packet in the queue
    uint64_t wr_idx = queue_add_write_index(q, 1);
    uint64_t packet_id = wr_idx % q->size;

    // capabilities packet
    dispatch_packet_t *cap_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    air_packet_get_capabilities(cap_pkt, AIR_VCK190_SHMEM_BASE + 0x300); // avoid the UART semaphore stuff
    air_queue_dispatch_and_wait(q, wr_idx, cap_pkt);

    // check the data we got back
    printf("Checking capabilities for MB %ld\n", controller_id);
    uint64_t *capabilities = (uint64_t *)mmap(NULL, 0x100, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_SHMEM_BASE);

    uint64_t expected[8] = {controller_id, 0L, 0x10001L, 16L, 32768L, 8L, 16384L, 0L};
    uint8_t to_check[8] = {1, 0, 1, 1, 1, 1, 1, 1};
    for (int i=0;i<8;i++) {
      if (to_check[i] && (capabilities[i+0x60] != expected[i])) {
        printf("Register %X: expected 0x%016lX: read 0x%016lX\n", i, expected[i], capabilities[i+0x60]);
        errors++;
      }
    }
    if (!errors) {
      total_controllers = capabilities[0x61];
      printf("MB %ld of %ld total is good\n", capabilities[0x60], capabilities[0x61]);
    }
    controller_id++;

    if ((errors) || (controller_id == capabilities[0x61]))
      done = true;
  }

  if (errors == 0x0) {
    printf("Checked %ld controllers, now make them all say hello\n", total_controllers);
    for (int c=0;c<total_controllers;c++) {
      queue_t *q = nullptr;
      uint64_t *qaddrs = (uint64_t*)AIR_VCK190_SHMEM_BASE;
      auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, (uint64_t)&qaddrs[c]);

      assert(ret == 0 && "failed to create queue!");

      // reserve a packet in the queue
      uint64_t wr_idx = queue_add_write_index(q, 1);
      uint64_t packet_id = wr_idx % q->size;

      dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
      air_packet_hello(pkt, 0xacdc0000LL + c);

      air_queue_dispatch_and_wait(q, wr_idx, pkt);
    } 
  }  

  if (errors == 0x0) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail.\n");
    return -1;
  }
}

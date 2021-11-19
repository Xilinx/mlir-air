// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include <assert.h>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>
#include <vector>

#include "acdc_queue.h"
#include "air_host.h"
#include "hsa_defs.h"

// Defined in acdc_queue.h
//typedef enum {
//  AIR_AGENT_INFO_NAME = 0,        // NUL-terminated char[8]
//  AIR_AGENT_INFO_VENDOR_NAME = 1, // NUL-terminated char[8]
//  AIR_AGENT_INFO_CONTROLLER_ID = 2,
//  AIR_AGENT_INFO_FIRMWARE_VER = 3,
//  AIR_AGENT_INFO_NUM_REGIONS = 4,
//  AIR_AGENT_INFO_HERD_SIZE = 5,
//  AIR_AGENT_INFO_HERD_ROWS = 6,
//  AIR_AGENT_INFO_HERD_COLS = 7,
//  AIR_AGENT_INFO_TILE_DATA_MEM_SIZE = 8,
//  AIR_AGENT_INFO_TILE_PROG_MEM_SIZE = 9,
//  AIR_AGENT_INFO_L2_MEM_SIZE = 10 // Per region
//} air_agent_info_t;

int main(int argc, char *argv[]) {
  std::vector<air_agent_t> agents;
  auto ret = air_get_agents(&agents);
  assert(ret == 0 && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "fail." << std::endl;
    return -1;
  }

  std::vector<queue_t *> queues;
  for (auto agent : agents) {
    // create the queue
    queue_t *q = nullptr;
    ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                           agent.handle);
    assert(ret == 0 && "failed to create queue!");
    queues.push_back(q);
  }
 
  char vend[8];
  for (auto q : queues) {
    std::cout << std::endl << "Requesting attribute: AIR_AGENT_INFO_CONTROLLER_ID... ";
    // reserve a packet in the queue
    uint64_t wr_idx = queue_add_write_index(q, 1);
    uint64_t packet_id = wr_idx % q->size;

    dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    initialize_packet(herd_pkt);
    herd_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
    herd_pkt->arg[0] = AIR_PKT_TYPE_GET_INFO;
    herd_pkt->arg[1] = AIR_AGENT_INFO_CONTROLLER_ID;

    air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

    std::cout << "Agent ID is: " << herd_pkt->arg[2] << std::endl;

    std::cout << "Requesting attribute: AIR_AGENT_INFO_VENDOR_NAME... ";
    // reserve a packet in the queue
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;

    herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    initialize_packet(herd_pkt);
    herd_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
    herd_pkt->arg[0] = AIR_PKT_TYPE_GET_INFO;
    herd_pkt->arg[1] = AIR_AGENT_INFO_VENDOR_NAME;

    air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

    memcpy(vend,&herd_pkt->arg[2],8);
    std::cout << "Vendor is: " << vend << std::endl;

    std::cout << "Requesting attribute: AIR_AGENT_INFO_L2_MEM_SIZE... ";
    // reserve a packet in the queue
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;

    herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    initialize_packet(herd_pkt);
    herd_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
    herd_pkt->arg[0] = AIR_PKT_TYPE_GET_INFO;
    herd_pkt->arg[1] = AIR_AGENT_INFO_L2_MEM_SIZE;

    air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

    std::cout << "L2 size is: " << std::dec << herd_pkt->arg[2] << "B" << std::endl;
  }

  std::cout << std::endl << "PASS!" << std::endl;
  return 0;
}

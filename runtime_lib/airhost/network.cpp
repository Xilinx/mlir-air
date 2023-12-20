//===- network.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "air.hpp"
#include "air_host.h"
#include "pcie-ernic.h"

#define QP_DEPTH 0x01000100

// Storing some state in the runtime
char air_hostname[100];
struct pcie_ernic_dev *air_ernic_dev;
std::map<std::string, int> hostname_to_qp_map;
std::map<void *, tensor_to_qp_map_entry *> tensor_to_qp_map;
std::map<std::string, world_view_entry *> world_view;
std::map<std::string, std::string> data_placement;

// Used to set this hosts hostname
hsa_status_t air_set_hostname(char hostname[100]) {
  strcpy(air_hostname, hostname);
  return HSA_STATUS_SUCCESS;
}

// Used to get the hostname of the host
hsa_status_t air_get_hostname(char hostname[100]) {
  strcpy(hostname, air_hostname);
  return HSA_STATUS_SUCCESS;
}

// This is a part of the bootstraping process, where we have a file that shows
// us all the AIR instances, and we want to initialize the ERNIC with our own
// information, and create QPs for every remote AIR instance
hsa_status_t
air_explore_world(uint32_t ernic_id, uint64_t dev_mem_offset,
                  uint64_t bar_offset,
                  std::map<std::string, world_view_entry *> pass_world_view,
                  std::map<std::string, std::string> pass_data_placement) {

  // Storing the representation of the distributed system
  world_view = pass_world_view;
  data_placement = pass_data_placement;

  // Reading the world view to get our own IP and MAC address
  uint32_t ip_addr =
      std::stoul(std::string(world_view[air_hostname]->ip), nullptr, 16);
  uint64_t mac_addr =
      std::stoul(std::string(world_view[air_hostname]->mac), nullptr, 16);
  uint32_t mac_addr_msb = mac_addr >> 32;
  uint32_t mac_addr_lsb = mac_addr & 0xffffffff;
  uint32_t our_rank = world_view[air_hostname]->rank;
  uint32_t src_qps[128];
  memcpy(src_qps, world_view[air_hostname]->qps, sizeof(src_qps));

#ifdef VERBOSE_DEBUG
  printf("[INFO] Initializing ERNIC:\n");
  printf("\tip_addr: 0x%x\n", ip_addr);
  printf("\tmac_addr_msb: 0x%x\n", mac_addr_msb);
  printf("\tmac_addr_lsb: 0x%x\n", mac_addr_lsb);
#endif

  // Initializing the ERNIC
  air_ernic_dev = pcie_ernic_open_dev(
      "" /*air_get_bram_bar(0).c_str()*/, // axil_bar_filename // TODO: Fix this
      2097152,                            // axil_bar_size
      bar_offset,                         // axil_bar_offset
      "" /*air_get_ddr_bar(0).c_str()*/,  // dev_mem_bar_filename // TODO: Fix
                                          // this
      67108864,                           // dev_mem_bar_size
      0x0000000800000000,                 // dev_mem_global_offset
      dev_mem_offset,                     // dev_mem_segment_offset
      0x00100000,                         // mrmac_reset_offset
      0x00110000,                         // mac_0_csr_offset
      0x00120000,                         // mac_1_csr_offset
      ernic_id,                           // ernic_id
      ip_addr,                            // ipv4 addr
      mac_addr_lsb,                       // mac_addr_lsb
      mac_addr_msb,                       // mac_addr_msb
      true,                               // configure_cmac
      false,                              // configure_bdf
      true,                               // is_versal
      true);                              // dual_reset

  if (air_ernic_dev == NULL) {
    printf("[ERROR] Failed to create pcie_ernic_dev structure\n");
    return HSA_STATUS_ERROR;
  }

  // QP 1 is specifically used for management so need to use QP 2
  for (std::map<std::string, world_view_entry *>::iterator it =
           world_view.begin();
       it != world_view.end(); it++) {
    if (it->first != air_hostname) {
      // Getting the information of the remote host
      ip_addr = std::stoul(std::string(world_view[it->first]->ip), nullptr, 16);
      mac_addr =
          std::stoul(std::string(world_view[it->first]->mac), nullptr, 16);
      mac_addr_msb = mac_addr >> 32;
      mac_addr_lsb = mac_addr & 0xffffffff;
      uint32_t remote_rank = world_view[it->first]->rank;
      uint32_t dst_qps[128];
      memcpy(dst_qps, world_view[it->first]->qps, sizeof(dst_qps));

#if defined(VERBOSE_DEBUG) && defined(DEBUG_NETWORK)
      printf("[INFO] Creating a QP:\n");
      printf("\tqpid: %d\n", src_qps[remote_rank]);
      printf("\tdest qpid: %d\n", dst_qps[our_rank]);
      printf("\tip_addr: 0x%x\n", ip_addr);
      printf("\tmac_addr_msb: 0x%x\n", mac_addr_msb);
      printf("\tmac_addr_lsb: 0x%x\n", mac_addr_lsb);
#endif

      // Allocate a pd for every QP with the QPID - 2 (First QP users can use is
      // 2)
      struct pcie_ernic_pd *pd =
          pcie_ernic_alloc_pd(air_ernic_dev, src_qps[remote_rank] - 2);

      // Creating a QP
      struct pcie_ernic_qp *qp2 =
          pcie_ernic_create_qp(air_ernic_dev,        // pcie_ernic_dev
                               pd,                   // pcie_ernic_pd
                               src_qps[remote_rank], // qpid
                               dst_qps[our_rank],    // dest qpid
                               QP_DEPTH,             // queue depth
                               ip_addr,              // dest ip
                               mac_addr_msb,         // dest mac
                               mac_addr_lsb,         // dest mac
                               false,                // enable cq
                               true);                // on_device

      if (qp2 == NULL) {
        printf("[ERROR] Failed to create QP2\n");
        return HSA_STATUS_ERROR;
      }

      hostname_to_qp_map.insert(
          std::pair<std::string, int>(it->first, src_qps[remote_rank]));
    }
  }

  return HSA_STATUS_SUCCESS;
}

/* Conditionally allocates and/or registers memory depending on
the location memory. This function should be called to initialize
all remote or local buffers that are accessed remotely. If the
buffer is local, the memory is allocated and the virtual address
and key are advertised to all AIR instances. If the buffer is
remote, we poll on receiving the virtual address and key
from the remote instance who is hosting the memory */
hsa_status_t air_ernic_mem_alloc(char buff_name[100], uint32_t size, void *t,
                                 bool register_mem) {

  std::string buff_location = data_placement[buff_name];

  std::cout << "[INFO] Allocating buffer: " << buff_name << std::endl;
  std::cout << "\tBuffer location: " << buff_location << std::endl;
  std::cout << "\tBuffer size: " << size << std::endl;

  tensor_t<uint32_t, 2> *tt = (tensor_t<uint32_t, 2> *)t;

  if (buff_location == air_hostname) {

    // Allocating the memory on the device and keeping track of the PA
    // then pointing the tensor to it
    struct pcie_ernic_buff *reg_mem =
        pcie_ernic_malloc(air_ernic_dev, size, true);
    tt->data = tt->alloc = (uint32_t *)reg_mem->buff;

    // Creating a map from the tensor to this
    struct tensor_to_qp_map_entry *entry =
        (struct tensor_to_qp_map_entry *)(malloc(
            sizeof(tensor_to_qp_map_entry)));
    entry->qp = 0;
    entry->rkey = 0;
    entry->vaddr = 0;
    entry->local_buff = reg_mem;

    tensor_to_qp_map.insert(
        std::pair<void *, tensor_to_qp_map_entry *>(tt->alloc, entry));

    // Registering the memory, and advertising it to every QP
    if (register_mem) {
      for (std::map<std::string, int>::iterator it = hostname_to_qp_map.begin();
           it != hostname_to_qp_map.end(); it++) {
        int remote_qp = it->second;
        struct pcie_ernic_qp *qp = air_ernic_dev->qps[remote_qp];
#if defined(VERBOSE_DEBUG) && defined(DEBUG_MEMORY)
        std::cout << "[INFO] Advertising buffer to QP" << remote_qp
                  << std::endl;
#endif
        pcie_ernic_reg_adv(air_ernic_dev, reg_mem, qp, 0x10, 0x00200000,
                           PD_READ_WRITE);
      }
    }
  } else {
    int remote_qp = hostname_to_qp_map[buff_location];
    if (remote_qp == 0) {
      std::cout << "[ERROR] Cannot find QP that points to hostname: "
                << buff_location << std::endl;
      return HSA_STATUS_ERROR;
    }

#if defined(VERBOSE_DEBUG) && defined(DEBUG_MEMORY)
    std::cout << "Allocating a remote buffer at QP" << remote_qp << std::endl;
#endif

    uint32_t rkey = 0;
    uint64_t vaddr = 0;
    struct pcie_ernic_qp *qp = air_ernic_dev->qps[remote_qp];
    pcie_ernic_recv_buff(air_ernic_dev, qp, &rkey, &vaddr);

#ifdef VERBOSE_DEBUG
    printf("[INFO] Received information about a remote buffer\n");
    printf("\tQP: 0x%x\n", remote_qp);
    printf("\trkey: 0x%x\n", rkey);
    printf("\tvaddr: 0x%lx\n", vaddr);
#endif

    struct tensor_to_qp_map_entry *entry =
        (struct tensor_to_qp_map_entry *)(malloc(
            sizeof(tensor_to_qp_map_entry)));
    struct pcie_ernic_buff *reg_mem =
        pcie_ernic_malloc(air_ernic_dev, size, true);
    entry->qp = remote_qp;
    entry->rkey = rkey;
    entry->vaddr = vaddr;
    entry->local_buff = reg_mem;

    // Can't use the address of the tensor to uniquely identify it because it is
    // copied internally before getting passed to the runtime. So need to
    // allocate a little bit of memory locally for the remote buffer so I can
    // identify it.
    // TODO: There must be a better way to do this
    tt->data = tt->alloc = (uint32_t *)reg_mem->buff;

    tensor_to_qp_map.insert(
        std::pair<void *, tensor_to_qp_map_entry *>(tt->alloc, entry));
  }

  return HSA_STATUS_SUCCESS;
}

// Should be called at the end of the application
hsa_status_t air_ernic_free() {

  pcie_ernic_free_dev(air_ernic_dev);

  return HSA_STATUS_SUCCESS;
}

/* Performs a message passing receive. We first poll on receiving an
RDMA SEND which contains the data, which is then copied to the provided
tensor t. We then send a synchronizing SEND back to remote agent. This
function is capable of performing a non-blocking SEND by passing an
HSA signal as a handle*/
void air_recv(hsa_signal_t *s, tensor_t<uint32_t, 1> *t, uint32_t size,
              uint32_t offset, uint32_t src_rank, hsa_agent_t *agent,
              hsa_queue_t *q, uint8_t ernic_sel) {

#ifdef VERBOSE_DEBUG
  printf("Called air_recv(%p, %p, 0x%x, 0x%x, 0x%x\n", s, t, size, offset,
         src_rank);
#endif

  if (q == NULL) {
    printf("[ERROR] air_recv given NULL queue\n");
    return;
  }

  if (world_view[air_hostname] == NULL) {
    printf("[ERROR] Called air_recv but hostname %s is not in world view\n",
           air_hostname);
    return;
  }

  // Unforunately we don't have any way to get the physical address for a buffer
  // not registered for RDMA, so for now the memory needs to be local registered
  // memory
  struct tensor_to_qp_map_entry *rdma_entry = tensor_to_qp_map[t->alloc];
  if (rdma_entry == NULL) {
    printf("[ERROR] air_recv given tensor not currently mapped\n");
    return;
  }

  // Checking if the buffer is remote or not
  if (rdma_entry->qp != 0) {
    printf("[ERROR] air_recv given remote tensor\n");
    return;
  }

  // Using the world_view to determine the QP that the SEND will come on
  uint32_t qpid = world_view[air_hostname]->qps[src_rank];
  if (qpid <= 1) {
    printf("[ERROR] in air_recv given src_rank %d which illegaly to qp %d\n",
           src_rank, qpid);
    return;
  }

  // If we are provided a signal we can implement a non-blocking SEND and
  // some other packets can wait on that signal
  if (s) {
    printf("TODO: Implement non-blocking air_recv\n");
  } else {

    // Calculating the number of RQEs that will be received
    uint32_t num_rqes = ceil((float)size / RQE_SIZE);
    uint32_t amount_data_left = size;

    uint64_t wr_idx, packet_id;
    uint32_t rqe_offset = 0;
    while (amount_data_left > 0) {

      // Calculating how much data we will recieve this iteration
      uint32_t amount_data_to_recv = 0;
      if (amount_data_left > RQE_SIZE) {
        amount_data_to_recv = RQE_SIZE;
      } else {
        amount_data_to_recv = amount_data_left;
      }

      wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
      packet_id = wr_idx % q->size;
      hsa_agent_dispatch_packet_t recv_pkt;

      air_packet_post_rdma_recv(&recv_pkt, // HSA Packet
                                rdma_entry->local_buff->pa + rqe_offset +
                                    offset,          // Local PADDR
                                amount_data_to_recv, // Length
                                (uint8_t)qpid,       // QPID
                                ernic_sel);          // ERNIC select

      // Need to do this because otherwise the runtime will complain the packet
      // is timing out but it is supposed to be blocking
      // air_write_pkt<hsa_agent_dispatch_packet_t>(q, packet_id, &recv_pkt);
      air_queue_dispatch(q, packet_id, wr_idx, &recv_pkt);

      while (hsa_signal_wait_scacquire(recv_pkt.completion_signal,
                                       HSA_SIGNAL_CONDITION_EQ, 0, 0x80000,
                                       HSA_WAIT_STATE_ACTIVE) != 0)
        ;

      // Calculating how much data we have to receive now and the new offset
      amount_data_left -= amount_data_to_recv;
      rqe_offset += amount_data_to_recv;
    }

    // Sending a synchronizing SEND so the corresponding air_send() can complete
    wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
    packet_id = wr_idx % q->size;
    hsa_agent_dispatch_packet_t sync_send_pkt;
    air_packet_post_rdma_wqe(
        &sync_send_pkt,             // HSA Packet
        0,                          // Remote VADDR
        rdma_entry->local_buff->pa, // Local PADDR -- Once 0 length SENDs are
                                    // working we can just make this 0
        0x00000100, // Length -- For some reason 0 length is not working so need
                    // to do a single RQE SEND
        (uint8_t)OP_SEND, // op
        0,                // Key
        (uint8_t)qpid,    // QPID
        ernic_sel);       // ERNIC select

    // air_write_pkt<hsa_agent_dispatch_packet_t>(q, packet_id, &sync_send_pkt);
    air_queue_dispatch_and_wait(agent, q, packet_id, wr_idx, &sync_send_pkt);
  }
}

/* Performs an SEND operation of the data in the provided tensor t.
It first performs an RDMA SEND of the data, and then must poll
on a synchronizing SEND which reports the data was received.
We then send a synchronizing SEND back to remote agent. This
function is capable of performing a non-blocking SEND by
passing an HSA signal as a handle */
void air_send(hsa_signal_t *s, tensor_t<uint32_t, 1> *t, uint32_t size,
              uint32_t offset, uint32_t dst_rank, hsa_agent_t *agent,
              hsa_queue_t *q, uint8_t ernic_sel) {

#ifdef VERBOSE_DEBUG
  printf("Called air_send(%p, %p, 0x%x, 0x%x\n", s, t, size, dst_rank);
#endif

  if (q == NULL) {
    printf("[ERROR] air_send given NULL queue\n");
    return;
  }

  if (world_view[air_hostname] == NULL) {
    printf("[ERROR] Called air_send but hostname %s is not in world view\n",
           air_hostname);
    return;
  }

  // Unforunately we don't have any way to get the physical address for a buffer
  // not registered for RDMA, so for now the memory needs to be local registered
  // memory
  struct tensor_to_qp_map_entry *rdma_entry = tensor_to_qp_map[t->alloc];
  if (rdma_entry == NULL) {
    printf("[ERROR] air_send given tensor not currently mapped\n");
    return;
  }

  // Checking if the buffer is remote or not
  if (rdma_entry->qp != 0) {
    printf("[ERROR] air_send given remote tensor\n");
    return;
  }

  // Using the world_view to determine the QP that the SEND will come on
  uint32_t qpid = world_view[air_hostname]->qps[dst_rank];
  if (qpid <= 1) {
    printf(
        "[ERROR] in air_send given dst_rank %d which illegaly maps to qp %d\n",
        dst_rank, qpid);
    return;
  }

  // If we are provided a signal we can implement a non-blocking SEND and
  // some other packets can wait on that signal
  if (s) {
    printf("TODO: Implement non-blocking air_send\n");
  } else {

    // Calculating the number of RQEs that will be received
    uint32_t num_rqes = ceil((float)size / RQE_SIZE);

    uint64_t wr_idx, packet_id;
    uint32_t rqe_offset = 0;
    for (int i = 0; i < num_rqes; i++) {
      wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
      packet_id = wr_idx % q->size;
      hsa_agent_dispatch_packet_t send_pkt;
      air_packet_post_rdma_wqe(
          &send_pkt,                                        // HSA Packet
          0,                                                // Remote VADDR
          rdma_entry->local_buff->pa + rqe_offset + offset, // Local PADDR
          0x00000100, // Length -- Need to send RQE size elements, receive side
                      // will only copy the valid data
          (uint8_t)OP_SEND, // op
          0,                // Key
          (uint8_t)qpid,    // QPID
          ernic_sel);       // ERNIC select

      // air_write_pkt<hsa_agent_dispatch_packet_t>(q, packet_id, &send_pkt);
      air_queue_dispatch_and_wait(agent, q, packet_id, wr_idx, &send_pkt);
    }

    wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
    packet_id = wr_idx % q->size;
    hsa_agent_dispatch_packet_t recv_pkt;

    air_packet_post_rdma_recv(
        &recv_pkt,                  // HSA Packet
        rdma_entry->local_buff->pa, // Local PADDR
        0, // Length - Synchronizing so don't want to copy any of the data over
        (uint8_t)qpid, // QPID
        ernic_sel);    // Ernic select

    // Need to do this because otherwise the runtime will complain the packet is
    // timing out but it is supposed to be blocking
    // air_write_pkt<hsa_agent_dispatch_packet_t>(q, packet_id, &recv_pkt);
    air_queue_dispatch(q, packet_id, wr_idx, &recv_pkt);
    while (hsa_signal_wait_scacquire(recv_pkt.completion_signal,
                                     HSA_SIGNAL_CONDITION_EQ, 0, 0x80000,
                                     HSA_WAIT_STATE_ACTIVE) != 0)
      ;
  }
}

/* Provides a very simplistic barrier for remote AIR instances.
This barrier uses rank 0 as the coordinater, which receives an
incoming send from every non-zero rank AIR instance. Then,
it perform an air_send to every non-zero rank AIR instance
which allows them to proceed. */
void air_barrier(tensor_t<uint32_t, 1> *dummy_tensor, hsa_agent_t *agent,
                 hsa_queue_t *q, uint8_t ernic_sel) {

  // Get my own rank
  // TODO: Do this when we explore world and then store it in some AIR state
  int my_rank = -1;
  int world_size = 0;
  for (std::map<std::string, world_view_entry *>::iterator it =
           world_view.begin();
       it != world_view.end(); it++) {
    if (it->first == air_hostname) {
      my_rank = world_view[it->first]->rank;
    }
    world_size++;
  }
  if (my_rank == -1) {
    std::cout << "Can't find hostname " << air_hostname << " in world view"
              << std::endl;
    return;
  }

#ifdef VERBOSE_DEBUG
  std::cout << "I am rank " << my_rank << " and I am in air_barrier()"
            << std::endl;
#endif

  // Because ERNIC doesn't support 0B SENDs have to send some dummy data
  if (my_rank == 0) {
    // Waiting for everyone to hit the barrier
    for (int i = 1; i < world_size; i++) {
      air_recv(NULL, dummy_tensor, RQE_SIZE, 0, i, agent, q, ernic_sel);
    }

    // Notifying everyone that they can continue
    for (int i = 1; i < world_size; i++) {
      air_send(NULL, dummy_tensor, RQE_SIZE, 0, i, agent, q, ernic_sel);
    }
  } else {
    // Perform a send() then recv() to rank 0
    air_send(NULL, dummy_tensor, RQE_SIZE, 0, 0, agent, q, ernic_sel);
    air_recv(NULL, dummy_tensor, RQE_SIZE, 0, 0, agent, q, ernic_sel);
  }
}

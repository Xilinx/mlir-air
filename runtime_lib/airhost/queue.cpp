#include <fcntl.h>
#include <sys/mman.h>
#include <cstdlib>
#include <cstring>

#include <cstdio>
#include <vector>
#include <iostream>

#include "air_host.h"
#include "acdc_queue.h"

hsa_status_t air_get_agents(void *data) {
  std::vector<air_agent_t> *pAgents = nullptr;

  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  } else {
    pAgents = static_cast<std::vector<air_agent_t> *>(data);
  }

  uint64_t total_controllers = 0;

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR;

  uint64_t *bram_base =
      reinterpret_cast<uint64_t *>(mmap(NULL, 0x1000, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd, AIR_VCK190_SHMEM_BASE));

  total_controllers = bram_base[65];
  if (total_controllers < 1) {
    std::cerr << "No agents found" << std::endl;
    return HSA_STATUS_ERROR;
  }

  uint64_t *base_addr = reinterpret_cast<uint64_t *>(AIR_VCK190_SHMEM_BASE);
  for (int i = 0; i < total_controllers; i++) {
    air_agent_t a;
    a.handle = reinterpret_cast<uintptr_t>(&base_addr[i]);
    pAgents->push_back(a);
  }

  auto res = munmap(bram_base, 0x1000);
  if (res) {
    std::cerr << "Could not munmap" << std::endl;
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_get_agent_info(queue_t *queue, air_agent_info_t attribute, void* data) {
  if ((data == nullptr) || (queue == nullptr)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  } 
 
  uint64_t wr_idx = queue_add_write_index(queue, 1);
  uint64_t packet_id = wr_idx % queue->size;

  dispatch_packet_t *pkt = (dispatch_packet_t*)(queue->base_address_vaddr) + packet_id;
  initialize_packet(pkt);
  pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
  pkt->arg[0] = AIR_PKT_TYPE_GET_INFO;
  pkt->arg[1] = attribute;
  air_queue_dispatch_and_wait(queue, wr_idx, pkt);
  
  if (attribute <= AIR_AGENT_INFO_VENDOR_NAME) {
    std::memcpy(data,&pkt->arg[2],8);
  } else {
    uint64_t *p = static_cast<uint64_t *>(data);
    *p = pkt->arg[2];
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_create(uint32_t size, uint32_t type, queue_t **queue, uint64_t paddr)
{
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;

  uint64_t paddr_aligned = paddr & 0xfffffffffffff000;
  uint64_t paddr_offset = paddr & 0x0000000000000fff;

  uint64_t *bram_ptr = (uint64_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, paddr_aligned);

  //printf("Opened shared memory paddr: %p vaddr: %p\n", paddr, bram_ptr);
  uint64_t q_paddr = bram_ptr[paddr_offset/sizeof(uint64_t)];
  uint64_t q_offset = q_paddr - paddr;
  queue_t *q = (queue_t*)( ((size_t)bram_ptr) + q_offset + paddr_offset );
  //printf("Queue location at paddr: %p vaddr: %p\n", bram_ptr[paddr_offset/sizeof(uint64_t)], q);

  if (q->id !=  0xacdc) {
    //printf("%s error invalid id %x\n", __func__, q->id);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->size != size) {
    //printf("%s error size mismatch %d\n", __func__, q->size);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->type != type) {
    //printf("%s error type mismatch %d\n", __func__, q->type);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  uint64_t base_address_offset = q->base_address - paddr_aligned;
  q->base_address_vaddr = ((size_t)bram_ptr) + base_address_offset;

  *queue = q;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_dispatch(queue_t *q, uint64_t doorbell, dispatch_packet_t *pkt) {
  // dispatch packet
  signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t*)&q->doorbell);
  signal_store_release((signal_t*)&q->doorbell, doorbell);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_wait(queue_t *q, dispatch_packet_t *pkt) {
  // wait for packet completion
  while (signal_wait_aquire((signal_t*)&pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout!\n");
    printf("%x\n", pkt->header);
    printf("%x\n", pkt->type);
    printf("%x\n", (unsigned)pkt->completion_signal);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_queue_dispatch_and_wait(queue_t *q, uint64_t doorbell, dispatch_packet_t *pkt) {
  // dispatch packet
  signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t*)&q->doorbell);
  signal_store_release((signal_t*)&q->doorbell, doorbell);

  // wait for packet completion
  while (signal_wait_aquire((signal_t*)&pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
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
                                  //uint8_t start_row, uint8_t num_rows,
                                  //uint16_t dma0, uint16_t dma1) {
  initialize_packet(pkt);

  pkt->arg[0]  = AIR_PKT_TYPE_HERD_INITIALIZE;
  pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  pkt->arg[0] |= ((uint64_t)num_cols) << 40;
  pkt->arg[0] |= ((uint64_t)start_col) << 32;
  pkt->arg[0] |= ((uint64_t)num_rows) << 24;
  pkt->arg[0] |= ((uint64_t)start_row) << 16;

  pkt->arg[1]  = herd_id;  // Herd ID
  //pkt->arg[1] |= ((char)dma0) << 16;
  //pkt->arg[1] |= ((char)dma1) << 24;
  pkt->arg[2]  = 0;        // unused
  pkt->arg[3]  = 0;        // unused

  pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_device_init(dispatch_packet_t *pkt, uint32_t num_cols) {
  initialize_packet(pkt);

  pkt->arg[0]  = AIR_PKT_TYPE_DEVICE_INITIALIZE;
  pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  pkt->arg[0] |= ((uint64_t)num_cols << 40);

  pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_aie_lock_range(dispatch_packet_t *pkt, uint16_t herd_id,
                                 uint64_t lock_id, uint64_t acq_rel, uint64_t value,
                                 uint8_t start_col, uint8_t num_cols,
                                 uint8_t start_row, uint8_t num_rows) {
  initialize_packet(pkt);

  pkt->arg[0]  = AIR_PKT_TYPE_XAIE_LOCK;
  pkt->arg[0] |= (AIR_ADDRESS_HERD_RELATIVE_RANGE << 48);
  pkt->arg[0] |= ((uint64_t)num_cols) << 40;
  pkt->arg[0] |= ((uint64_t)start_col) << 32;
  pkt->arg[0] |= ((uint64_t)num_rows) << 24;
  pkt->arg[0] |= ((uint64_t)start_row) << 16;
  pkt->arg[1]  = lock_id;
  pkt->arg[2]  = acq_rel;
  pkt->arg[3]  = value;

  pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_packet_nd_memcpy(dispatch_packet_t *pkt, uint16_t herd_id,
                                 uint8_t col, uint8_t direction, uint8_t channel,
                                 uint8_t burst_len, uint8_t memory_space,
                                 uint64_t phys_addr, uint32_t transfer_length1d,
                                 uint32_t transfer_length2d, uint32_t transfer_stride2d,
                                 uint32_t transfer_length3d, uint32_t transfer_stride3d,
                                 uint32_t transfer_length4d, uint32_t transfer_stride4d) {

  initialize_packet(pkt);

  pkt->arg[0]  = AIR_PKT_TYPE_ND_MEMCPY;
  pkt->arg[0] |= ((uint64_t)memory_space) << 16;
  pkt->arg[0] |= ((uint64_t)channel)      << 24;
  pkt->arg[0] |= ((uint64_t)col)          << 32;
  pkt->arg[0] |= ((uint64_t)burst_len)    << 52;
  pkt->arg[0] |= ((uint64_t)direction)    << 60;

  pkt->arg[1]  = phys_addr;
  pkt->arg[2]  = transfer_length1d;
  pkt->arg[2] |= ((uint64_t)transfer_length2d) <<32;
  pkt->arg[2] |= ((uint64_t)transfer_stride2d) <<48;
  pkt->arg[3]  = transfer_length3d;
  pkt->arg[3] |= ((uint64_t)transfer_stride3d) <<16;
  pkt->arg[3] |= ((uint64_t)transfer_length4d) <<32;
  pkt->arg[3] |= ((uint64_t)transfer_stride4d) <<48;

  pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  return HSA_STATUS_SUCCESS;
}



hsa_status_t air_packet_aie_lock(dispatch_packet_t *pkt, uint16_t herd_id,
                                 uint64_t lock_id, uint64_t acq_rel, uint64_t value,
                                 uint8_t col, uint8_t row) {
  return air_packet_aie_lock_range(pkt, herd_id, lock_id, acq_rel,
                                   value, col, 1, row, 1);
}

//air_packet_unlock

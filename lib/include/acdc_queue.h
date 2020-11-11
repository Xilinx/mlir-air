#include <stdint.h>
#include "hsa_defs.h"

#define MB_QUEUE_SIZE 64

typedef struct dispatch_packet_s {
  
  // HSA-like interface
  uint16_t header;
  uint16_t type;
  uint32_t reserved0;
  uint64_t return_address;
  uint64_t arg[4];
  uint64_t reserved1;
  uint64_t completion_signal;

} __attribute__((packed)) dispatch_packet_t;

typedef struct queue_s {

  // HSA-like interface
  uint32_t type;
  uint32_t features;
  uint64_t base_address;
  uint64_t doorbell;
  uint32_t size;
  uint32_t reserved0;
  uint64_t id;

  // implementation detail
  uint64_t read_index;
  uint64_t write_index;
  uint64_t last_doorbell;

  uint64_t base_address_paddr;
  uint64_t base_address_vaddr;

} __attribute__((packed)) queue_t;

typedef struct signal_s {
  uint64_t handle;
} signal_t;

typedef uint64_t signal_value_t;

namespace {

inline uint64_t queue_add_write_index(queue_t *q, uint64_t v)
{
  auto r = q->write_index;
  q->write_index = r + v;
  return r;
}

inline uint64_t queue_add_read_index(queue_t *q, uint64_t v)
{
  auto r = q->read_index;
  q->read_index = r + v;
  return r;
}

inline uint64_t queue_load_read_index(queue_t *q)
{
  return q->read_index;
}

inline uint64_t queue_load_write_index(queue_t *q)
{
  return q->write_index;
}

inline bool packet_get_active(dispatch_packet_t *pkt)
{
  return pkt->reserved1 & 0x1;
}

inline void packet_set_active(dispatch_packet_t *pkt, bool b)
{
  pkt->reserved1 = (pkt->reserved1 & ~0x1) | b;
}

inline void initialize_packet(dispatch_packet_t *pkt)
{
  pkt->header = 0;
  pkt->type = HSA_PACKET_TYPE_INVALID;
}

inline hsa_status_t signal_create(signal_value_t initial_value, uint32_t num_consumers, void *consumers, signal_t *signal)
{
  //auto s = (signal_value_t*)malloc(sizeof(signal_value_t));
  //*s = initial_value;
  signal->handle = (uint64_t)initial_value;
  return HSA_STATUS_SUCCESS;
}

inline hsa_status_t signal_destroy(signal_t signal)
{
  //free((void*)signal.handle);
}

inline void signal_store_release(signal_t *signal, signal_value_t value)
{
  signal->handle = (uint64_t)value;
}

inline signal_value_t signal_wait_aquire(volatile signal_t *signal,
                                         hsa_signal_condition_t condition,
                                         signal_value_t compare_value,
                                         uint64_t timeout_hint,
                                         hsa_wait_state_t wait_state_hint)
{
  signal_value_t ret = 0;
  uint64_t timeout = timeout_hint;
  do {
    ret = signal->handle;
    if (ret == compare_value)
      return compare_value;
  } while (timeout--);
  return ret;
}

inline void signal_subtract_acq_rel(signal_t *signal, signal_value_t value)
{
  signal->handle = signal->handle - (uint64_t)value;
  // uint64_t i;
  // memcpy((void*)&i, (void*)signal, sizeof(signal_value_t));
  // i = i - (uint64_t)value;
  // memcpy((void*)signal, (void*)&i, sizeof(signal_value_t));
}

}
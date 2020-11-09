#include <stdint.h>

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
    volatile uint64_t doorbell;
    uint32_t size;
    uint32_t reserved0;
    uint64_t id;

    // implementation detail
    uint64_t read_index;
    uint64_t write_index;
    uint64_t last_doorbell;

} __attribute__((packed)) queue_t;

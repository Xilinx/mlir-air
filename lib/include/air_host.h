#ifndef AIR_HOST_H
#define AIR_HOST_H

#include "acdc_queue.h"
#include "hsa_defs.h"

#include <cstdlib>

// queue operations
//

hsa_status_t air_queue_create(uint32_t size, uint32_t type, queue_t **queue, uint64_t paddr);

// memory operations
//

void* air_mem_alloc(size_t size);

void* air_mem_get_paddr(void *vaddr);
void* air_mem_get_vaddr(void *paddr);

void air_mem_dealloc(void *vaddr);

#endif // AIR_HOST_H
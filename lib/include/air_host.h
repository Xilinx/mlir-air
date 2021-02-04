#ifndef AIR_HOST_H
#define AIR_HOST_H

#include "acdc_queue.h"
#include "hsa_defs.h"

#include <cstdlib>

#ifdef AIR_LIBXAIE_ENABLE
#include <xaiengine.h>
#define air_mem_t XAieLib_MemInst
#else
#define air_mem_t void
#endif

// queue operations
//

hsa_status_t air_queue_create(uint32_t size, uint32_t type, queue_t **queue, uint64_t paddr);

// memory operations
//

air_mem_t* air_malloc(size_t size);
void air_dealloc(air_mem_t *mem);

#endif // AIR_HOST_H
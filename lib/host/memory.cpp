#include "air_host.h"

#include <cassert>
#include <vector>

#ifdef AIR_LIBXAIE_ENABLE
#include <xaiengine.h>
#endif

namespace {

static std::vector<XAieLib_MemInst*> AirAllocations;

}

void* air_mem_alloc(size_t size) {
#ifdef AIR_LIBXAIE_ENABLE
  XAieLib_MemInst *mem = XAieLib_MemAllocate(size, 0);
  if (!mem)
    return nullptr;
  AirAllocations.push_back(mem);
  void *vaddr = (void*)XAieLib_MemGetVaddr(mem);
  return vaddr;
#else
  // TODO
  assert(0 && "not implemented");
  return nullptr;
#endif
}

void air_mem_dealloc(void *vaddr) {
#ifdef AIR_LIBXAIE_ENABLE
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
#else
  // TODO
  assert(0 && "not implemented");
#endif
}

void *air_mem_get_paddr(void *vaddr) {
#ifdef AIR_LIBXAIE_ENABLE
  for (auto *m : AirAllocations) {
    void *p = (void*)XAieLib_MemGetVaddr(m);
    if (p == vaddr)
      return (void*)XAieLib_MemGetPaddr(m);
  }
#else
  // TODO
  assert(0 && "not implemented");
#endif
  return nullptr;
}

void *air_mem_get_vaddr(void *paddr) {
#ifdef AIR_LIBXAIE_ENABLE
  for (auto *m : AirAllocations) {
    void *p = (void*)XAieLib_MemGetPaddr(m);
    if (p == paddr)
      return (void*)XAieLib_MemGetVaddr(m);
  }
#else
  // TODO
  assert(0 && "not implemented");
#endif
  return nullptr;
}

#include "air_host.h"

#include <cstdio>
#include <cassert>

int
main (int argc, char *argv[])
{
  void *ptr = air_mem_alloc(1024);
  printf("alloc returned vaddr: %p\n", ptr);

  void *paddr = air_mem_get_paddr(ptr);
  printf("               paddr: %p\n", paddr);

  void *vaddr = air_mem_get_vaddr(paddr);
  printf("               vaddr: %p\n", vaddr);

  int *iptr = (int*)ptr;
  iptr[512] = 0xdeadbeef;

  if (vaddr != ptr) {
    printf("failed.\n");
    return -1;
  }

  air_mem_dealloc(ptr);

  printf("PASS!\n");
  return 0;
}
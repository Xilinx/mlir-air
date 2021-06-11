
#include <cstdio>
#include <cassert>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "acdc_queue.h"
#include "hsa_defs.h"


#define L2_DMA_BASE 0x020240000000LL

int main(int argc, char *argv[])
{

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1) {
    printf("failed to open /dev/mem\n");
    return -1;
  }

  uint64_t *bank0_ptr = (uint64_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE);

  // I have no idea if this does anything
  __clear_cache((void*)bank0_ptr, (void*)(((size_t)bank0_ptr)+0x20000));

  // Write a value
  bank0_ptr[0] = 0xf001ba11deadbeefL;
  bank0_ptr[1] = 0x00defaceL;

  // Read back the value above it

  uint64_t word0 = bank0_ptr[0];
  uint64_t word1 = bank0_ptr[1];

  unsigned word0_lo = (unsigned)word0;
  unsigned word1_lo = (unsigned)word1;
  unsigned word0_hi = (unsigned)(word0 >> 32);
  unsigned word1_hi = (unsigned)(word1 >> 32);
  printf("I read back %08X%08X\r\n", word0_hi, word0_lo);
  printf("I read back %08X%08X\r\n", word1_hi, word1_lo);
  printf("PASS!\n");

  return 0;
}

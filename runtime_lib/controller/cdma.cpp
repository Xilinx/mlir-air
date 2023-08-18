//===- cdma.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// https://docs.xilinx.com/r/en-US/pg034-axi-cdma/Programming-Sequence

#include "platform.h"
#include "xil_printf.h"

#ifdef DEBUG_CDMA
#define PRINTF xil_printf
#else
#define PRINTF(...)
#endif // DEBUG_CDMA

#define CDMA_MAX_SG_DESCS 32
#define CDMA_SG_NO_PREV (uint32_t)(-1)
#define CDMA_MAX_TRANSFER_LEN (1 << 26) // 64MB
#define CDMA_ALIGNMENT_FACTOR 8 // must be aligned to 0x40 bytes

#define REG_CDMA_CR (CDMA_BASE + 0x00UL)        // control
#define REG_CDMA_SR (CDMA_BASE + 0x04UL)        // status
#define REG_CDMA_CURR_DESC (CDMA_BASE + 0x08UL) // current descriptor
#define REG_CDMA_CURR_DESC_MSB (CDMA_BASE + 0x0CUL)
#define REG_CDMA_TAIL_DESC (CDMA_BASE + 0x10UL) // tail descriptor
#define REG_CDMA_TAIL_DESC_MSB (CDMA_BASE + 0x14UL)
#define REG_CDMA_SA (CDMA_BASE + 0x18UL) // source address
#define REG_CDMA_SA_MSB (CDMA_BASE + 0x1CUL)
#define REG_CDMA_DA (CDMA_BASE + 0x20UL) // destination address
#define REG_CDMA_DA_MSB (CDMA_BASE + 0x24UL)
#define REG_CDMA_BTT (CDMA_BASE + 0x28UL) // bytes to transfer

#define REG_CDMA_CR_SG (1U << 3)
#define REG_CDMA_CR_RESET (1U << 2)

#define REG_CDMA_SR_SG_DEC_ERR (1U << 10)
#define REG_CDMA_SR_SG_SLV_ERR (1U << 9)
#define REG_CDMA_SR_SG_INT_ERR (1U << 8)
#define REG_CDMA_SR_DMA_DEC_ERR (1U << 6)
#define REG_CDMA_SR_DMA_SLV_ERR (1U << 5)
#define REG_CDMA_SR_DMA_INT_ERR (1U << 4)
#define REG_CDMA_SR_IDLE (1U << 1)

struct cdma_sg_descriptor {
  uint32_t next; // pointer to next descriptor in the chain
  uint32_t next_msb;
  uint32_t src; // source address
  uint32_t src_msb;
  uint32_t dest; // destination address
  uint32_t dest_msb;
  uint32_t ctrl;   // control (length of data transfer up to 64MB)
  uint32_t status; // completion status
  uint32_t padding[CDMA_ALIGNMENT_FACTOR];
};

// CDMA descriptors must be aligned to 16-byte addresses
static cdma_sg_descriptor descs[CDMA_MAX_SG_DESCS] __attribute__((aligned (64)));

/*
  reset DMA engine
*/
void cdma_reset(void) {
  IO_WRITE32(REG_CDMA_CR, IO_READ32(REG_CDMA_CR) | REG_CDMA_CR_RESET);
  while (IO_READ32(REG_CDMA_CR) & REG_CDMA_CR_RESET)
    ;
}

/*
  Initialize the static descriptor chain.

  This assumes all descriptors are in a contiguous memory block so they can be
  indexed directly. Thus, each descriptor has a linear index from 0 to
  CDMA_MAX_SG_DESCS-1. All descriptors point to the next index.
*/
void cdma_sg_init(void) {
  cdma_reset();

  for (uint32_t idx = 0; idx < CDMA_MAX_SG_DESCS; idx++) {
    uint64_t addr = (uint64_t)&descs[idx + 1];
    descs[idx].next = addr & 0xFFFFFFFF;
    descs[idx].next_msb = addr >> 32;
    descs[idx].status = 0;
  }
}

/*
  Set up a descriptor in a scatter/gather chain
*/
int cdma_sg_set(uint32_t idx, uint64_t dest, uint64_t src, uint32_t len) {
  if (idx >= CDMA_MAX_SG_DESCS)
    return -1;

  if (len > CDMA_MAX_TRANSFER_LEN)
    return -2;

  descs[idx].src = src & 0xFFFFFFFF;
  descs[idx].src_msb = src >> 32;
  descs[idx].dest = dest & 0xFFFFFFFF;
  descs[idx].dest_msb = dest >> 32;
  descs[idx].ctrl = len;
  descs[idx].status = 0;

  return 0;
}

/*
  Start processing a descriptor chain
*/
void cdma_sg_start(uint32_t head, uint32_t tail) {
  uint64_t addr;

  PRINTF("%s: head=%u tail=%u\r\n", __func__, head, tail);
  if (head >= CDMA_MAX_SG_DESCS)
    return;
  if (tail <= head)
    return;

  // clear SG mode
  IO_WRITE32(REG_CDMA_CR, 0);

  // enable SG mode
  IO_WRITE32(REG_CDMA_CR, REG_CDMA_CR_SG);

  // write the head (i.e. first descriptor to process)
  addr = (uint64_t)&descs[head];
  IO_WRITE32(REG_CDMA_CURR_DESC, addr & 0xFFFFFFFF);
  IO_WRITE32(REG_CDMA_CURR_DESC_MSB, addr >> 32);

  // write the tail (i.e. last descriptor to process)
  addr = (uint64_t)&descs[tail];
  IO_WRITE32(REG_CDMA_TAIL_DESC, addr & 0xFFFFFFFF);
  IO_WRITE32(REG_CDMA_TAIL_DESC_MSB, addr >> 32);
}

/*
  Start processing a descriptor chain and wait until it completes
*/
uint32_t cdma_sg_start_sync(uint32_t head, uint32_t tail) {
#ifdef USE_MEMCPY
  for (uint32_t idx = head; idx <= tail; idx++) {
    uint64_t src = (uint64_t)descs[idx].src | ((uint64_t)descs[idx].src_msb) << 32;
    uint64_t dest = (uint64_t)descs[idx].dest | ((uint64_t)descs[idx].dest_msb) << 32;
    uint32_t length = descs[idx].ctrl;
    memcpy((void*)dest, (void*)src, length);
  }
  return 0;
#else
  cdma_sg_start(head, tail);

  // block until the engine goes idle
  while (!(IO_READ32(REG_CDMA_SR) & REG_CDMA_SR_IDLE))
    ;
  return (IO_READ32(REG_CDMA_SR) &
    (REG_CDMA_SR_SG_DEC_ERR
    |REG_CDMA_SR_SG_SLV_ERR
    |REG_CDMA_SR_SG_INT_ERR
    |REG_CDMA_SR_DMA_DEC_ERR
    |REG_CDMA_SR_DMA_SLV_ERR
    |REG_CDMA_SR_DMA_INT_ERR));
#endif // USE_MEMCPY
}

void cdma_print_status(void) {
  xil_printf("Control: 0x%08x Status: 0x%08x\r\n",
    IO_READ32(REG_CDMA_CR), IO_READ32(REG_CDMA_SR));

  xil_printf("Current descriptor: 0x%lx\r\n",
    (uint64_t)IO_READ32(REG_CDMA_CURR_DESC) | (uint64_t)IO_READ32(REG_CDMA_CURR_DESC_MSB) << 32);

  xil_printf("Tail descriptor: 0x%lx\r\n",
    (uint64_t)IO_READ32(REG_CDMA_TAIL_DESC) | (uint64_t)IO_READ32(REG_CDMA_TAIL_DESC_MSB) << 32);

  for (uint32_t idx=0; idx < CDMA_MAX_SG_DESCS; idx++) {
    xil_printf("[%06lx] src=0x%08lx dest=0x%08lx length=0x%x status=0x%x next=0x%08lx\r\n",
      (uint64_t)&descs[idx], (uint64_t)descs[idx].src | ((uint64_t)descs[idx].src_msb) << 32,
      (uint64_t)descs[idx].dest | ((uint64_t)descs[idx].dest_msb) << 32,
      descs[idx].ctrl, descs[idx].status,
      (uint64_t)descs[idx].next | ((uint64_t)descs[idx].next_msb) << 32);
  }
}

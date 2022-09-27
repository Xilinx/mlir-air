//===- Module.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>

#include "LibAirHostModule.h"

#ifdef AIE_LIBXAIE_ENABLE
#include <xaiengine.h>
#endif

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

#ifdef AIE_LIBXAIE_ENABLE
#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_NUM_ROWS 9
#define XAIE_NUM_COLS 50
#define XAIE_COL_SHIFT 23
#define XAIE_ROW_SHIFT 18
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 0
#define XAIE_RES_TILE_NUM_ROWS 0
#define XAIE_AIE_TILE_ROW_START 1
#define XAIE_AIE_TILE_NUM_ROWS 8

#define HIGH_ADDR(addr)   ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)    (addr & 0x00000000ffffffff)
#endif

namespace {

#ifdef AIE_LIBXAIE_ENABLE
XAie_Config *AieConfigPtr;                           /**< AIE configuration pointer */
XAie_DevInst AieInst;                                /**< AIE global instance */

void printDMAStatus(int col, int row) {
  u64 tileAddr = _XAie_GetTileAddr(&(AieInst), row, col);

  u32 dma_mm2s_status;
  XAie_Read32(&(AieInst), tileAddr + 0x0001DF10, &dma_mm2s_status);
  u32 dma_s2mm_status;
  XAie_Read32(&(AieInst), tileAddr + 0x0001DF00, &dma_s2mm_status);
  u32 dma_mm2s0_control;
  XAie_Read32(&(AieInst), tileAddr + 0x0001DE10, &dma_mm2s0_control);
  u32 dma_mm2s1_control;
  XAie_Read32(&(AieInst), tileAddr + 0x0001DE18, &dma_mm2s1_control);
  u32 dma_s2mm0_control;
  XAie_Read32(&(AieInst), tileAddr + 0x0001DE00, &dma_s2mm0_control);
  u32 dma_s2mm1_control;
  XAie_Read32(&(AieInst), tileAddr + 0x0001DE08, &dma_s2mm1_control);
  u32 dma_bd0_a;
  XAie_Read32(&(AieInst), tileAddr + 0x0001D000, &dma_bd0_a);
  u32 dma_bd0_control;
  XAie_Read32(&(AieInst), tileAddr + 0x0001D018, &dma_bd0_control);
  u32 dma_bd1_a;
  XAie_Read32(&(AieInst), tileAddr + 0x0001D020, &dma_bd1_a);
  u32 dma_bd1_control;
  XAie_Read32(&(AieInst), tileAddr + 0x0001D038, &dma_bd1_control);

  u32 s2mm_ch0_running = dma_s2mm_status & 0x3;
  u32 s2mm_ch1_running = (dma_s2mm_status >> 2) & 0x3;
  u32 mm2s_ch0_running = dma_mm2s_status & 0x3;
  u32 mm2s_ch1_running = (dma_mm2s_status >> 2) & 0x3;

  printf("DMA [%d, %d] mm2s_status/0ctrl/1ctrl is %08X %02X %02X, "
         "s2mm_status/0ctrl/1ctrl is %08X %02X %02X, BD0_Addr_A is %08X, "
         "BD0_control is %08X, BD1_Addr_A is %08X, BD1_control is %08X\n",
         col, row, dma_mm2s_status, dma_mm2s0_control, dma_mm2s1_control,
         dma_s2mm_status, dma_s2mm0_control, dma_s2mm1_control, dma_bd0_a,
         dma_bd0_control, dma_bd1_a, dma_bd1_control);
  for (int bd = 0; bd < 8; bd++) {
    u32 dma_bd_addr_a;
    XAie_Read32(&(AieInst), tileAddr + 0x0001D000 + (0x20 * bd),
                &dma_bd_addr_a);
    u32 dma_bd_control;
    XAie_Read32(&(AieInst), tileAddr + 0x0001D018 + (0x20 * bd),
                &dma_bd_control);
    if (dma_bd_control & 0x80000000) {
      printf("BD %d valid\n", bd);
      int current_s2mm_ch0 = (dma_s2mm_status >> 16) & 0xf;
      int current_s2mm_ch1 = (dma_s2mm_status >> 20) & 0xf;
      int current_mm2s_ch0 = (dma_mm2s_status >> 16) & 0xf;
      int current_mm2s_ch1 = (dma_mm2s_status >> 20) & 0xf;

      if (s2mm_ch0_running && bd == current_s2mm_ch0) {
        printf(" * Current BD for s2mm channel 0\n");
      }
      if (s2mm_ch1_running && bd == current_s2mm_ch1) {
        printf(" * Current BD for s2mm channel 1\n");
      }
      if (mm2s_ch0_running && bd == current_mm2s_ch0) {
        printf(" * Current BD for mm2s channel 0\n");
      }
      if (mm2s_ch1_running && bd == current_mm2s_ch1) {
        printf(" * Current BD for mm2s channel 1\n");
      }

      if (dma_bd_control & 0x08000000) {
        u32 dma_packet;
        XAie_Read32(&(AieInst), tileAddr + 0x0001D010 + (0x20 * bd),
                    &dma_packet);
        printf("   Packet mode: %02X\n", dma_packet & 0x1F);
      }
      int words_to_transfer = 1 + (dma_bd_control & 0x1FFF);
      int base_address = dma_bd_addr_a & 0x1FFF;
      printf("   Transfering %d 32 bit words to/from %06X\n", words_to_transfer,
             base_address);

      printf("   ");
      for (int w = 0; w < 7; w++) {
        u32 tmpd;
        XAie_DataMemRdWord(&(AieInst), XAie_TileLoc(col, row),
                           (base_address + w) * 4, &tmpd);
        printf("%08X ", tmpd);
      }
      printf("\n");
      if (dma_bd_addr_a & 0x40000) {
        u32 lock_id = (dma_bd_addr_a >> 22) & 0xf;
        printf("   Acquires lock %d ", lock_id);
        if (dma_bd_addr_a & 0x10000)
          printf("with value %d ", (dma_bd_addr_a >> 17) & 0x1);

        printf("currently ");
        u32 locks;
        XAie_Read32(&(AieInst), tileAddr + 0x0001EF00, &locks);
        u32 two_bits = (locks >> (lock_id * 2)) & 0x3;
        if (two_bits) {
          u32 acquired = two_bits & 0x1;
          u32 value = two_bits & 0x2;
          if (acquired)
            printf("Acquired ");
          printf(value ? "1" : "0");
        } else
          printf("0");
        printf("\n");
      }
      if (dma_bd_control & 0x30000000) { // FIFO MODE
        int FIFO = (dma_bd_control >> 28) & 0x3;
        u32 dma_fifo_counter;
        XAie_Read32(&(AieInst), tileAddr + 0x0001DF20, &dma_fifo_counter);
        printf("   Using FIFO Cnt%d : %08X\n", FIFO, dma_fifo_counter);
      }
      u32 nextBd = ((dma_bd_control >> 13) & 0xF);
      u32 useNextBd = ((dma_bd_control >> 17) & 0x1);
      printf("   Next BD: %d, Use next BD: %d\n", nextBd, useNextBd);
    }
  }
}

void aie_memory_module_DMA_S2MM_Status(int col, int row)
{
  uint32_t dma_s2mm_status;
  XAie_Read32(&AieInst, _XAie_GetTileAddr(&AieInst, row, col) + 0x0001DF00, &dma_s2mm_status);
}
#endif

} // namespace

PYBIND11_MODULE(_airRt, m) {
    m.doc() = R"pbdoc(
        Xilinx AIR Python bindings
        --------------------------

        .. currentmodule:: AIR_

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    m.def("dma_status", [](int col, int row) {
#ifdef AIE_LIBXAIE_ENABLE
        AieConfigPtr = (XAie_Config*)malloc(sizeof(XAie_Config));
        AieConfigPtr->AieGen = XAIE_DEV_GEN_AIE;
        AieConfigPtr->BaseAddr = XAIE_BASE_ADDR;
        AieConfigPtr->ColShift = XAIE_COL_SHIFT;
        AieConfigPtr->RowShift = XAIE_ROW_SHIFT;
        AieConfigPtr->NumRows = XAIE_NUM_ROWS;
        AieConfigPtr->NumCols = XAIE_NUM_COLS;
        AieConfigPtr->ShimRowNum = XAIE_SHIM_ROW;
        AieConfigPtr->MemTileRowStart = XAIE_RES_TILE_ROW_START;
        AieConfigPtr->MemTileNumRows = XAIE_RES_TILE_NUM_ROWS;
        AieConfigPtr->AieTileRowStart = XAIE_AIE_TILE_ROW_START;
        AieConfigPtr->AieTileNumRows = XAIE_AIE_TILE_NUM_ROWS;
        AieConfigPtr->PartProp = {0};

        XAie_CfgInitialize(&AieInst, AieConfigPtr);
        XAie_PmRequestTiles(&AieInst, NULL, 0);
        XAie_Finish(&AieInst);
        XAie_CfgInitialize(&AieInst, AieConfigPtr);
        XAie_PmRequestTiles(&AieInst, NULL, 0);

        printDMAStatus(col, row);
#else
        printf("ERROR: LIBXAIE is not enabled\n");
#endif
    });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

  auto airhost = m.def_submodule("host", "libairhost bindings");
  xilinx::air::defineAIRHostModule(airhost);
}

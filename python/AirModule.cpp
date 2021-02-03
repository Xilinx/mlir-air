#include <pybind11/pybind11.h>

#ifdef AIR_LIBXAIE_ENABLE
#include <xaiengine.h>
#endif


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
  return i + j;
}

namespace py = pybind11;

#ifdef AIR_LIBXAIE_ENABLE
#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)   ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)    (addr & 0x00000000ffffffff)
#endif

namespace {

#ifdef AIR_LIBXAIE_ENABLE
XAieGbl_Config *AieConfigPtr;                              /**< AIE configuration pointer */
XAieGbl AieInst;                                          /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

void printDMAStatus(int col, int row) {
  u32 dma_mm2s_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF10);
  u32 dma_s2mm_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF00);
  u32 dma_mm2s_control = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DE10);
  u32 dma_s2mm_control = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DE00);
  u32 dma_bd0_a       = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D000); 
  u32 dma_bd0_control = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D018);

  u32 s2mm_ch0_running = dma_s2mm_status & 0x3;
  u32 s2mm_ch1_running = (dma_s2mm_status >> 2) & 0x3;
  u32 mm2s_ch0_running = dma_mm2s_status & 0x3;
  u32 mm2s_ch1_running = (dma_mm2s_status >> 2) & 0x3;

  printf("DMA [%d, %d] mm2s_status/ctrl is %08X %08X, s2mm_status is %08X %08X, BD0_Addr_A is %08X, BD0_control is %08X\n",col, row, dma_mm2s_status, dma_mm2s_control, dma_s2mm_status, dma_s2mm_control, dma_bd0_a, dma_bd0_control);
  for (int bd=0;bd<8;bd++) {
      u32 dma_bd_addr_a        = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D000 + (0x20*bd));
      u32 dma_bd_control       = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D018 + (0x20*bd));
    if (dma_bd_control & 0x80000000) {
      printf("BD %d valid\n",bd);
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
        u32 dma_packet = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D010 + (0x20*bd));
        printf("   Packet mode: %02X\n",dma_packet & 0x1F);
      }
      int words_to_transfer = 1+(dma_bd_control & 0x1FFF);
      int base_address = dma_bd_addr_a  & 0x1FFF;
      printf("   Transfering %d 32 bit words to/from %05X\n",words_to_transfer, base_address);

      printf("   ");
      for (int w=0;w<4; w++) {
        printf("%08X ",XAieTile_DmReadWord(&(TileInst[col][row]), (base_address+w) * 4));
      }
      printf("\n");
      if (dma_bd_addr_a & 0x40000) {
        u32 lock_id = (dma_bd_addr_a >> 22) & 0xf;
        printf("   Acquires lock %d ",lock_id);
        if (dma_bd_addr_a & 0x10000) 
          printf("with value %d ",(dma_bd_addr_a >> 17) & 0x1);

        printf("currently ");
        u32 locks = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001EF00);
        u32 two_bits = (locks >> (lock_id*2)) & 0x3;
        if (two_bits) {
          u32 acquired = two_bits & 0x1;
          u32 value = two_bits & 0x2;
          if (acquired)
            printf("Acquired ");
          printf(value?"1":"0");
        }
        else printf("0");
        printf("\n");

      }
      if (dma_bd_control & 0x30000000) { // FIFO MODE
        int FIFO = (dma_bd_control >> 28) & 0x3;
          u32 dma_fifo_counter = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF20);                
        printf("   Using FIFO Cnt%d : %08X\n",FIFO, dma_fifo_counter);
      }
    }

  }
}

void aie_memory_module_DMA_S2MM_Status(int col, int row)
{
  uint32_t dma_s2mm_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF00);
}
#endif

} // namespace

PYBIND11_MODULE(_air, m) {
    m.doc() = R"pbdoc(
        Xilinx AIR Python bindings
        --------------------------

        .. currentmodule:: AIR_

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    m.def("dma_status", [](int col, int row) {
#ifdef AIR_LIBXAIE_ENABLE
        size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
        XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
        XAieGbl_HwInit(&AieConfig);
        AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
        XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);
        printDMAStatus(col, row);
#else
        printf("ERROR: LIBXAIE is not enabled\n");
#endif
    });

    m.def("_add", [](uint64_t i, uint64_t j) {
      return add(i,j);
    });
    
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

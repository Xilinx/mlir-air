#include "platform.h"

extern "C" {
#include "xil_printf.h"
#include "xaiengine.h"
#include "xil_cache.h"
#include "air_queue.h"
#include "hsa_defs.h"
}

#define VCK5000_CDMA_BASE 0x0000A4000000UL
#define XAIE_NUM_ROWS 8
#define XAIE_NUM_COLS 50

const char vck5000_platform_name[] = "vck5000";

struct aie_libxaie_ctx_t {
  XAie_Config AieConfigPtr;
  XAie_DevInst DevInst;
};

aie_libxaie_ctx_t *_xaie;

u32 in32(u64 Addr)
{
  u32 Value = 0;
  XAie_Read32(&(_xaie->DevInst), Addr, &Value);
  return Value;
}

void out32(u64 Addr, u32 Value)
{
  XAie_Write32(&(_xaie->DevInst), Addr, Value);
}

u64 getTileAddr(u16 ColIdx, u16 RowIdx)
{
  return _XAie_GetTileAddr(&(_xaie->DevInst), RowIdx, ColIdx);
}

void mlir_aie_init_libxaie(aie_libxaie_ctx_t *ctx) {
  if (!ctx)
    return;

  ctx->AieConfigPtr.AieGen = XAIE_DEV_GEN_AIE;
  ctx->AieConfigPtr.BaseAddr = 0x20000000000; // XAIE_BASE_ADDR;
  ctx->AieConfigPtr.ColShift = XAIEGBL_TILE_ADDR_COL_SHIFT;
  ctx->AieConfigPtr.RowShift = XAIEGBL_TILE_ADDR_ROW_SHIFT;
  ctx->AieConfigPtr.NumRows = XAIE_NUM_ROWS + 1;
  ctx->AieConfigPtr.NumCols = XAIE_NUM_COLS;
  ctx->AieConfigPtr.ShimRowNum = 0;      // XAIE_SHIM_ROW;
  ctx->AieConfigPtr.MemTileRowStart = 0; // XAIE_RES_TILE_ROW_START;
  ctx->AieConfigPtr.MemTileNumRows = 0;  // XAIE_RES_TILE_NUM_ROWS;
  ctx->AieConfigPtr.AieTileRowStart = 1; // XAIE_AIE_TILE_ROW_START;
  ctx->AieConfigPtr.AieTileNumRows = XAIE_NUM_ROWS;
  ctx->AieConfigPtr.PartProp = {0};
  ctx->DevInst = {0};
}

int mlir_aie_init_device(aie_libxaie_ctx_t *ctx) {
  AieRC RC = XAIE_OK;

  RC = XAie_CfgInitialize(&(ctx->DevInst), &(ctx->AieConfigPtr));
  if (RC != XAIE_OK) {
    xil_printf("Driver initialization failed.\n\r");
    return -1;
  }

  RC = XAie_PmRequestTiles(&(ctx->DevInst), NULL, 0);
  if (RC != XAIE_OK) {
    xil_printf("Failed to request tiles.\n\r");
    return -1;
  }

  // TODO Extra code to really teardown the partitions
  RC = XAie_Finish(&(ctx->DevInst));
  if (RC != XAIE_OK) {
    xil_printf("Failed to finish tiles.\n\r");
    return -1;
  }
  RC = XAie_CfgInitialize(&(ctx->DevInst), &(ctx->AieConfigPtr));
  if (RC != XAIE_OK) {
    xil_printf("Driver initialization failed.\n\r");
    return -1;
  }
  RC = XAie_PmRequestTiles(&(ctx->DevInst), NULL, 0);
  if (RC != XAIE_OK) {
    xil_printf("Failed to request tiles.\n\r");
    return -1;
  }

  return 0;
}

int mlir_aie_reinit_device(aie_libxaie_ctx_t *ctx) {
  AieRC RC = XAIE_OK;

  RC = XAie_Finish(&(ctx->DevInst));
  if (RC != XAIE_OK) {
    xil_printf("Failed to finish tiles.\n\r");
    return -1;
  }
  RC = XAie_CfgInitialize(&(ctx->DevInst), &(ctx->AieConfigPtr));
  if (RC != XAIE_OK) {
    xil_printf("Driver initialization failed.\n\r");
    return -1;
  }
  RC = XAie_PmRequestTiles(&(ctx->DevInst), NULL, 0);
  if (RC != XAIE_OK) {
    xil_printf("Failed to request tiles.\n\r");
    return -1;
  }

  return 0;
}

int plat_device_init(void)
{
	return mlir_aie_reinit_device(_xaie);
}

int init_platform(struct platform_cfg *cfg)
{
#ifdef STDOUT_IS_16550
    XUartNs550_SetBaud(STDOUT_BASEADDR, XPAR_XUARTNS550_CLOCK_HZ, UART_BAUD);
    XUartNs550_SetLineControlReg(STDOUT_BASEADDR, XUN_LCR_8_DATA_BITS);
#endif

  Xil_DCacheDisable();

  aie_libxaie_ctx_t ctx;
  _xaie = &ctx;
  mlir_aie_init_libxaie(_xaie);
  int err = mlir_aie_init_device(_xaie);
	if (err)
		return err;

  cfg->mb_count = 1;
  cfg->version = ENCODE_VERSION(1, 0, 0);
  cfg->cdma_base = VCK5000_CDMA_BASE;
  cfg->platform_name = vck5000_platform_name;

  return 0;
}


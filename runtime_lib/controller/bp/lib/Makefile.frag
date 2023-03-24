# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

XIL_BSP_DIR = $(XIL_EMB_SW_DIR)/lib/bsp/standalone/src/common
XIL_UARTLITE_DIR = $(XIL_EMB_SW_DIR)/XilinxProcessorIPLib/drivers/uartlite/src
XIL_MUTEX_DIR = $(XIL_EMB_SW_DIR)/XilinxProcessorIPLib/drivers/mutex/src

XIL_LIB_SRC = \
  $(XIL_BSP_DIR)/xil_assert.c \
  $(XIL_MUTEX_DIR)/xmutex_g.c \
  $(XIL_MUTEX_DIR)/xmutex_sinit.c \
  $(XIL_UARTLITE_DIR)/xuartlite_l.c \

BP_FW_LIB_OVERRIDE_SRC = \
  $(BP_FW_LIB_SRC_DIR)/outbyte.c \
  $(BP_FW_LIB_SRC_DIR)/xil_printf.c \
  $(BP_FW_LIB_SRC_DIR)/xmutex.c \


// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef XPARAMETERS_H   /* prevent circular inclusions */
#define XPARAMETERS_H   /* by using protection macros */

/******************************************************************/

#define STDIN_BASEADDRESS 0x40600000
#define STDOUT_BASEADDRESS 0x40600000

/******************************************************************/

/* Definitions for driver UARTLITE */
#define XPAR_XUARTLITE_NUM_INSTANCES 1U

/* Definitions for peripheral AXI_UARTLITE_0 */
#define XPAR_AXI_UARTLITE_0_DEVICE_ID 0U
#define XPAR_AXI_UARTLITE_0_BASEADDR 0x40600000U
#define XPAR_AXI_UARTLITE_0_HIGHADDR 0x4060FFFFU
#define XPAR_AXI_UARTLITE_0_BAUDRATE 115200U
#define XPAR_AXI_UARTLITE_0_USE_PARITY 0U
#define XPAR_AXI_UARTLITE_0_ODD_PARITY 0U
#define XPAR_AXI_UARTLITE_0_DATA_BITS 8U

/* Canonical definitions for peripheral AXI_UARTLITE_0 */
#define XPAR_UARTLITE_0_DEVICE_ID 0U
#define XPAR_UARTLITE_0_BASEADDR 0x40600000U
#define XPAR_UARTLITE_0_HIGHADDR 0x4060FFFFU
#define XPAR_UARTLITE_0_BAUDRATE 115200U
#define XPAR_UARTLITE_0_USE_PARITY 0U
#define XPAR_UARTLITE_0_ODD_PARITY 0U
#define XPAR_UARTLITE_0_DATA_BITS 8U

/* Definitions for Mutex */
#define XPAR_XMUTEX_NUM_INSTANCES 1U
#define XPAR_MUTEX_0_DEVICE_ID 0U
#define XPAR_MUTEX_0_BASEADDR 0x020200000000ULL
#define XPAR_MUTEX_0_NUM_MUTEX 32U
#define XPAR_MUTEX_0_ENABLE_USER 0U
/* currently allocated locks */
#define XPAR_MUTEX_0_UART_LOCK 0U
#define XPAR_MUTEX_0_NUM_MB_LOCK 1U

/******************************************************************/
#endif  /* end of protection macro */

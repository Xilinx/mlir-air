/******************************************************************************
* Copyright (c) 2020 - 2021 Xilinx, Inc.  All rights reserved.
* Copyright (C) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "xparameters.h"
#include "xuartlite_l.h"

#ifdef __cplusplus
extern "C" {
#endif
void outbyte(char c);

#ifdef __cplusplus
}
#endif

void outbyte(char c) {
	 XUartLite_SendByte(STDOUT_BASEADDRESS, c);
}

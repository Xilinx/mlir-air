//===- shell.h --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef __SHELL_H_
#define __SHELL_H_

/*
        Poll for incoming commands on the UART

        If there are no waiting characters, return
        If there are waiting characters, add them to the command buffer
        A command is terminated by the \r or \n character
        When a full command is received, parse it and try to do what it says
*/
void shell(void);

#endif // __SHELL_H_

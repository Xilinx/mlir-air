//===- shell.cpp ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "shell.h"
#include "platform.h"
#include "xuartpsv_hw.h"
#include <stdint.h>
#include <stdlib.h>

#define MAX_LINE_LENGTH 256
#define MAX_CMD_LENGTH 15
#define NUM_CMDS (sizeof(command_tbl) / sizeof(command_entry))

/*
        Each command handler has this format
*/
typedef struct command_entry {
  char cmd[MAX_CMD_LENGTH + 1]; // the command string
  void (*fn)(char *cmd);        // the handler function
} command_entry;

// forward declarations of handlers
static void command_cp(char *line);
static void command_help(char *line);
static void command_x(char *line);

static int new_cmd = 1;
static int cmd_len;
static char line[MAX_LINE_LENGTH];
static command_entry command_tbl[] = {
    {"cp", command_cp},
    {"help", command_help},
    {"x", command_x},
};

static char to_ascii(uint8_t byte) {
  if ((byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') ||
      (byte >= '0' && byte <= '9'))
    return byte;

  return '.';
}

/*
        Print information about the command processors
*/
static void command_cp(char *line) {
  uint32_t cp_count;
  uint64_t base = get_base_address();

  cp_count = *(uint32_t *)(base + 0x208);
  xil_printf("Number of command processors: %u\r\n", cp_count);

  uint64_t *queue_base = (uint64_t *)base;
  for (uint32_t idx = 0; idx < cp_count; idx++) {
    xil_printf("[%02u]\t0x%llx\r\n", idx, queue_base[idx]);
  }
}

/*
        List all supported shell commands
*/
static void command_help(char *line) {
  uint32_t ll = 0; // line length
  uint32_t len;    // string length

  xil_printf("Supported commands:\r\n");

  for (uint32_t idx = 0; idx < NUM_CMDS; idx++) {
    len = strlen(command_tbl[idx].cmd) + 1;
    if (ll + len > 80) {
      ll = len;
      xil_printf("\r\n");
    }
    xil_printf("%s ", command_tbl[idx].cmd);
  }
  xil_printf("\r\n");
}

/*
        Print out a memory range

        parameter 1: ARM virtual address (in hex or decimal)
        parameter 2: number of bytes to read (should be aligned to 8)
*/
static void command_x(char *line) {
  char *cmd;
  char *address_str;
  char *range_str;
  uint64_t address = 0x400000;
  uint32_t range = 8;

  cmd = strtok(line, " ");
  address_str = strtok(NULL, " ");
  range_str = strtok(NULL, " ");

  if (address_str)
    address = strtoul(address_str, NULL, 16);
  if (range_str)
    range = strtoul(range_str, NULL, 16);

  xil_printf("%s: addr=0x%llx range=0x%llx\r\n", cmd, address, range);

  uint8_t *w = (uint8_t *)address;

  for (uint32_t i = 0; i < range; i += 8)
    xil_printf("[%16llx]: %02x %02x %02x %02x %02x %02x %02x %02x | "
               "%c%c%c%c%c%c%c%c\r\n",
               address + i * sizeof(*w), w[i], w[i + 1], w[i + 2], w[i + 3],
               w[i + 4], w[i + 5], w[i + 6], w[i + 7], to_ascii(w[i]),
               to_ascii(w[i + 1]), to_ascii(w[i + 2]), to_ascii(w[i + 3]),
               to_ascii(w[i + 4]), to_ascii(w[i + 5]), to_ascii(w[i + 6]),
               to_ascii(w[i + 7]));
}

static void handle_command(void) {
  // look up the command and call the handler
  for (uint32_t idx = 0; idx < NUM_CMDS; idx++) {
    if (strncmp(line, command_tbl[idx].cmd, strlen(command_tbl[idx].cmd)) ==
        0) {
      command_tbl[idx].fn(line);
      break;
    }
  }
}

void shell(void) {
  uint8_t in;

  if (new_cmd) {
    new_cmd = 0;
    cmd_len = 0;

    // good old DOS prompt
    xil_printf("C:\\> ");
  }

  // handle all characters from the UART (which is a slow interface) so the UI
  // is responsive. If no character is waiting, go back to processing queues.
  while ((in = XUartPsv_RecvByte(STDOUT_BASEADDRESS))) {
    // make sure character will fit in the command buffer
    if (cmd_len >= MAX_LINE_LENGTH) {
      xil_printf("Line too long\r\n");
      new_cmd = 1;
      break;
    }

    // if it's a return character, handle the command
    if (in == '\r' || in == '\n') {
      XUartPsv_SendByte(STDOUT_BASEADDRESS, '\r');
      XUartPsv_SendByte(STDOUT_BASEADDRESS, '\n');
      handle_command();
      new_cmd = 1;
      break;
    }

    line[cmd_len++] = in;
    line[cmd_len] = 0;

    // reflect the typed character back for confirmation
    XUartPsv_SendByte(STDOUT_BASEADDRESS, in);
  }
}

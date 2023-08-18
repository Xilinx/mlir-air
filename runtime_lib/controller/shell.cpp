//===- shell.cpp ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "shell.h"
#include "cdma.h"
#include "platform.h"
#include "xuartpsv_hw.h"
#include <stdint.h>
#include <stdlib.h>

#define MAX_LINE_LENGTH 256
#define MAX_CMD_LENGTH 15
#define MAX_DIFF_LENGTH 0x8000
#define NUM_CMDS (sizeof(command_tbl) / sizeof(command_entry))

/*
        Each command handler has this format
*/
typedef struct command_entry {
  char cmd[MAX_CMD_LENGTH + 1]; // the command string
  void (*fn)(char *cmd);        // the handler function
} command_entry;

// forward declarations of handlers
static void command_aie(char *line);
static void command_cdma(char *line);
static void command_cp(char *line);
static void command_diff(char *line);
static void command_dma(char *line);
static void command_help(char *line);
static void command_write(char *line);
static void command_read(char *line);
static void command_reset(char *line);

static int new_cmd = 1;
static int cmd_len;
static char line[MAX_LINE_LENGTH];
static command_entry command_tbl[] = {
    {"aie", command_aie},   {"cdma", command_cdma}, {"cp", command_cp},
    {"diff", command_diff}, {"dma", command_dma},   {"help", command_help},
    {"w", command_write},   {"x", command_read},    {"reset", command_reset},
};

static char to_ascii(uint8_t byte) {
  if ((byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') ||
      (byte >= '0' && byte <= '9'))
    return byte;

  return '.';
}

static void command_aie(char *line) {
  char *col_str, *row_str, *op_str;
  int col = 6, row = 2;

  strtok(line, " ");
  col_str = strtok(NULL, " ");
  row_str = strtok(NULL, " ");
  op_str = strtok(NULL, " ");

  if (col_str)
    col = strtol(col_str, NULL, 0);
  if (row_str)
    row = strtol(row_str, NULL, 0);

  if (col < 0 || col > 49)
    return;
  if (row < 1 || row > 8)
    return;

  if (!op_str || strcmp(op_str, "status") == 0) {
    xil_printf("aie status col=%d row=%d\r\n", col, row);
    mlir_aie_print_tile_status(col, row);
  } else if (strcmp(op_str, "reset") == 0) {
    xil_printf("aie reset col=%d row=%d\r\n", col, row);
    aie_tile_reset(col, row);
  } else if (strcmp(op_str, "enable") == 0) {
    xil_printf("aie enable col=%d row=%d\r\n", col, row);
    aie_tile_enable(col, row);
  }
}

static void command_cdma(char *line) { cdma_print_status(); }

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
  Compare two regions of memory
*/
static void command_diff(char *line) {
  char *first_str, *second_str, *length_str;
  uint64_t first, second;
  uint32_t length = 4;
  uint32_t w1, w2;
  uint8_t identical = 1;

  strtok(line, " ");
  first_str = strtok(NULL, " ");
  second_str = strtok(NULL, " ");
  length_str = strtok(NULL, " ");

  if (!first_str || !second_str)
    return;

  first = strtoul(first_str, NULL, 0);
  second = strtoul(second_str, NULL, 0);

  if (length_str)
    length = strtoul(length_str, NULL, 0);

  // limit the length to prevent mistakes
  if (length > MAX_DIFF_LENGTH) {
    xil_printf("Limiting to 0x%x\r\n", MAX_DIFF_LENGTH);
    length = MAX_DIFF_LENGTH;
  }

  // force the addresses to be 4-byte aligned
  first &= ~(4ULL - 1);
  second &= ~(4ULL - 1);

  for (uint32_t idx = 0; idx < length; idx += 4) {
    w1 = *(uint32_t *)first;
    w2 = *(uint32_t *)second;
    if (w1 != w2) {
      identical = 0;
      xil_printf("[0x%lx] 0x%x != 0x%x\r\n", second, w1, w2);
    }

    first += 4;
    second += 4;
  }

  if (identical)
    xil_printf("Identical!\r\n");
}

/*
  Tile DMA status
*/
static void command_dma(char *line) {
  char *col_str, *row_str;
  uint32_t col = 6, row = 2;

  strtok(line, " ");
  col_str = strtok(NULL, " ");
  row_str = strtok(NULL, " ");

  if (col_str)
    col = strtoul(col_str, NULL, 0);
  if (row_str)
    row = strtoul(row_str, NULL, 0);

  if (col < 0 || col > 49)
    return;
  if (row > 8)
    return;

  if (row == 0)
    mlir_aie_print_shimdma_status(col);
  else
    mlir_aie_print_dma_status(col, row);
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
static void command_read(char *line) {
  char *address_str;
  char *range_str;
  uint64_t address = 0x400000;
  uint32_t range = 8;
  uint32_t w[2];
  uint8_t *bytes = (uint8_t *)&w;

  strtok(line, " ");
  address_str = strtok(NULL, " ");
  range_str = strtok(NULL, " ");

  if (address_str)
    address = strtoul(address_str, NULL, 0);
  if (range_str)
    range = strtoul(range_str, NULL, 0);

  // force the address to be 4-byte aligned
  address &= ~(4ULL - 1);

  for (uint32_t i = 0; i < range; i += 8) {
    w[0] = IO_READ32(address + i);
    w[1] = IO_READ32(address + i + 4);

    xil_printf("[%16llx]: %02x %02x %02x %02x %02x %02x %02x %02x | "
               "%c%c%c%c%c%c%c%c\r\n",
               address + i, bytes[0], bytes[1], bytes[2], bytes[3], bytes[4],
               bytes[5], bytes[6], bytes[7], to_ascii(bytes[0]),
               to_ascii(bytes[1]), to_ascii(bytes[2]), to_ascii(bytes[3]),
               to_ascii(bytes[4]), to_ascii(bytes[5]), to_ascii(bytes[6]),
               to_ascii(bytes[7]));
  }
}

static void command_write(char *line) {
  char *address_str;
  char *val_str;
  uint64_t address = 0x400000;
  uint32_t val = 0;

  strtok(line, " ");
  address_str = strtok(NULL, " ");
  val_str = strtok(NULL, " ");

  if (address_str) {
    xil_printf("address_str: %s\r\n", address_str);
    address = strtoul(address_str, NULL, 0);
  }
  if (val_str)
    val = strtoul(val_str, NULL, 0);

  xil_printf("write 0x%x to 0x%lx\r\n", val, address);
  uint32_t *w = (uint32_t *)address;
  *w = val;
}

static void command_reset(char *line) {
  char *block_str;

  strtok(line, " ");
  block_str = strtok(NULL, " ");

  if (!block_str) {
    xil_printf("Invalid block name\r\n");
    xil_printf("options: array device\r\n");
    return;
  }

  if (strcmp(block_str, "array") == 0) {
    xaie_array_reset();
  } else if (strcmp(block_str, "device") == 0) {
    xaie_device_init();
  }
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
  while (XUartPsv_IsReceiveData(STDOUT_BASEADDRESS)) {

    // When we know that we have data, read it from the UART
    in = XUartPsv_RecvByte(STDOUT_BASEADDRESS);

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

# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Description:
#   This file reads a .mem file produced by a tool such as objcopy and reformats it for
#   consumption by the Vivado tool to initialized instantiated memories.
#
#   The Vivado format uses word addresses and each data value (output data line) represents
#   a single word in the memory.
#

import os
import sys
import argparse
import math


def format_mem(f, width, base_addr):
  addr_offset = int(base_addr[2:],16)
  width_bytes = width//8
  with open(os.path.abspath(f)) as memfile:
    for line in memfile:
      l = line.strip()
      if line.startswith('@'):
        a = int(line[1:],16)
        # check address alignment
        # it is possible to handle this by padding the data bytes, but that may require processing
        # multiple lines at a time
        if (a % width_bytes != 0):
          print('ERROR: address {0} is not aligned to {1}'.format(l, width_bytes))
          return
        # convert address to an offset relative to 0
        # then, convert to a word address instead of byte address
        a = (a - addr_offset) // width_bytes
        print('@{0:012x}'.format(a))
      else:
        # strip internal whitespace and convert to byte array
        l = l.replace(' ', '')
        # compute number of bytes
        bytes_remaining = len(l)//2

        # starting index into byte array
        idx = 0
        while (bytes_remaining > 0):
          val = 0

          # compute number of bytes to write
          # largest power of 2 up to maximum bytes specified by width function argument
          log2_write_bytes = math.floor(math.log2(bytes_remaining))
          # built-in pow more accurate than math.pow()
          write_bytes = pow(2, log2_write_bytes)
          if (write_bytes > width_bytes):
            write_bytes = width_bytes;

          word = l[(idx*2):((idx+write_bytes)*2)]
          r = bytearray.fromhex(word)
          r.reverse()
          print(r.hex())

          bytes_remaining = bytes_remaining - write_bytes
          idx = idx + write_bytes


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mem', dest='mem_file', metavar='main.mem', default='main.mem', help='BP program .mem file', required=True)
  parser.add_argument('--width', dest='width', metavar='N', default=64, help='memory word width', type=int)
  parser.add_argument('--base-addr', dest='base_addr', metavar='0x80000000', default='0x80000000', help='base address of mem file', type=str)
  args = parser.parse_args()

  # verify args
  if (args.width < 8 or (math.ceil(math.log2(args.width)) != math.floor(math.log2(args.width)))):
    print('Width of {0} is invalid. Must be a power of two and at least 8'.format(args.width))
    exit(-1)

  # run on the input file
  format_mem(args.mem_file, args.width, args.base_addr)

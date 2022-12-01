//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <string>

//#include "air_host.h"
struct air_herd_shim_desc_t {
  int64_t *location_data;
  int64_t *channel_data;
};

struct air_herd_desc_t {
  int64_t name_length;
  char *name;
  air_herd_shim_desc_t *shim_desc;
};

struct air_partition_desc_t {
  int64_t name_length;
  char *name;
  uint64_t herd_length;
  air_herd_desc_t **herd_descs;
};

struct air_module_desc_t {
  uint64_t partition_length;
  air_partition_desc_t **partition_descs;
};

extern air_module_desc_t __airrt_module_descriptor;

int
main(int argc, char *argv[])
{
  int num_partitions = __airrt_module_descriptor.partition_length;
  printf("Num Partitions: %d\n", (int)num_partitions);
  for (int j = 0; j < num_partitions; j++) {

    auto partition_desc = __airrt_module_descriptor.partition_descs[j];
    std::string partition_name(partition_desc->name,
                               partition_desc->name_length);
    printf("\tPartition %d: %s\n", j, partition_name.c_str());

    int num_herds = partition_desc->herd_length;
    printf("\tNum Herds: %d\n", num_herds);
    for (int i = 0; i < num_herds; i++) {
      auto herd_desc = partition_desc->herd_descs[i];

      std::string herd_name(herd_desc->name, herd_desc->name_length);
      printf("\t\tHerd %d: %s\n", i, herd_name.c_str());

      for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
          for (int k = 0; k < 8; k++) {
            if (int d =
                    herd_desc->shim_desc->channel_data[i * 8 * 8 + j * 8 + k])
              printf("\t\t\tShim Channel : id %d, row %d, col %d, channel %d\n",
                     i, j, k, d);
            if (int d =
                    herd_desc->shim_desc->location_data[i * 8 * 8 + j * 8 + k])
              printf("\t\t\tShim Location : id %d, row %d, col %d, column %d\n",
                     i, j, k, d);
          }
        }
      }
    }
  }
  return 0;
}
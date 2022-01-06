// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include <cstdio>
#include <string>

//#include "air_host.h"
struct air_herd_shim_desc_t {
  int64_t *location_data;
  int64_t *channel_data;
};

struct air_herd_desc_t {
  int32_t name_length;
  char *name;
  air_herd_shim_desc_t *shim_desc;
};

struct air_module_desc_t {
  uint64_t length;
  air_herd_desc_t **herd_descs;
};

extern air_module_desc_t __air_module_descriptor;

int
main(int argc, char *argv[])
{
  auto num_herds = __air_module_descriptor.length;
  printf("Num Herds: %d\n", (int)num_herds);
  for (int i=0; i<num_herds; i++) {
    auto herd_desc = __air_module_descriptor.herd_descs[i];

    std::string herd_name(herd_desc->name, herd_desc->name_length);
    printf("\tHerd %d: %s\n", i, herd_name.c_str());

    for (int i=0; i<16; i++) {
      for (int j=0; j<8; j++) {
        for (int k=0; k<8; k++) {
          if (int d = herd_desc->shim_desc->channel_data[i*8*8 + j*8 + k])
            printf("\t\tShim Channel : id %d, row %d, col %d, channel %d\n", i,j,k,d);
          if (int d = herd_desc->shim_desc->location_data[i*8*8 + j*8 + k])
            printf("\t\tShim Location : id %d, row %d, col %d, column %d\n", i,j,k,d);
        }
      }
    }
  }

  return 0;
}
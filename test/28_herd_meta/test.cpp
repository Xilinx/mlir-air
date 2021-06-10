#include <cstdio>
#include <string>

#include "air_host.h"

extern air_module_desc_t __air_module_descriptor;

int
main(int argc, char *argv[])
{
  auto num_herds = __air_module_descriptor.length;
  printf("Num Herds: %d\n", num_herds);
  for (int i=0; i<num_herds; i++) {
    auto herd_desc = __air_module_descriptor.herd_descs[i];
  
    std::string herd_name(herd_desc->name, herd_desc->name_length);
    printf("\tHerd %d: %s\n", i, herd_name.c_str());

    for (int i=0; i<16; i++) {
      for (int j=0; j<4; j++) {
        for (int k=0; k<4; k++) {
          if (int d = herd_desc->shim_desc->channel_data[i*4*4 + j*4 + k])
            printf("\t\tShim Channel : id %d, row %d, col %d, channel %d\n", i,j,k,d);
          if (int d = herd_desc->shim_desc->location_data[i*4*4 + j*4 + k])
            printf("\t\tShim Location : id %d, row %d, col %d, column %d\n", i,j,k,d);
        }
      }
    }
  }

  return 0;
}
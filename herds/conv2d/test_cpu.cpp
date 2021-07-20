
#include <iostream>

#include "air_tensor.h"

extern "C" {

void _mlir_ciface_graph(tensor_t<float,4> *in, tensor_t<float,4> *out);

void air_herd_load(char *name) {
  std::cout << "Load herd: " << name << std::endl;
}

void air_memcpy4d_4F32_4F32(uint32_t id, uint64_t x, uint64_t y, tensor_t<float,4> *dst, tensor_t<float,4> *src,
                                 uint64_t dst_offset_3, uint64_t dst_offset_2, uint64_t dst_offset_1, uint64_t dst_offset_0,
                                 uint64_t src_offset_3, uint64_t src_offset_2, uint64_t src_offset_1, uint64_t src_offset_0,
                                 uint64_t length, uint64_t stride, uint64_t elem_per_stride) {

  std::cout << "called memcpy4d";
  std::cout << " id: " << id;
  std::cout << " x: " << x;
  std::cout << " y: " << y;
  std::cout << " dst_offset: [" << dst_offset_3 << "," << dst_offset_2  << "," << dst_offset_1 << "," << dst_offset_0 << "]";
  std::cout << " src_offset: [" << src_offset_3 << "," << src_offset_2  << "," << src_offset_1 << "," << src_offset_0 << "]";
  std::cout << " length: " << length;
  std::cout << " stride: " << stride;
  std::cout << " elem_per_stride: " << elem_per_stride;  
  std::cout << "\n";
}

};

int main(int argc, char *argv[])
{
  tensor_t<float,4> in;
  tensor_t<float,4> out;
  
  in.shape[0] = 1;
  in.shape[1] = 48;
  in.shape[2] = 416;
  in.shape[3] = 416;
  in.d = in.aligned = (float*)malloc(sizeof(float)*in.shape[0]*in.shape[1]*in.shape[2]*in.shape[3]);

  out.shape[0] = 1;
  out.shape[1] = 64;
  out.shape[2] = 416;
  out.shape[3] = 416;
  out.d = out.aligned = (float*)malloc(sizeof(float)*out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]);

  _mlir_ciface_graph(&in, &out);
  return EXIT_SUCCESS;
}
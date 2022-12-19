//===- memory.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air_tensor.h"

#include <cstdint>
#include <cstdio>

#define VERBOSE false

namespace {

template <typename T, int R>
void air_memcpy_nd_dst(tensor_t<T, R> *dst, tensor_t<T, R> *src, size_t *offset,
                       size_t *size, size_t *stride) {
  if (VERBOSE)
    printf("dst offset %lu, %lu, size %lu, %lu, stride %lu, %lu\n", offset[1],
           offset[0], size[1], size[0], stride[1], stride[0]);
  size_t src_offset = 0;
  for (size_t j = 0; j < size[1]; j++)
    for (size_t i = 0; i < size[0]; i++) {
      size_t idx =
          ((offset[1] + j) * stride[1]) + ((offset[0] + i) * stride[0]);
      dst->data[idx] = src->data[src_offset++];
    }
}

template <typename T, int R>
void air_memcpy_nd_src(tensor_t<T, R> *dst, tensor_t<T, R> *src, size_t *offset,
                       size_t *size, size_t *stride) {
  if (VERBOSE)
    printf("src offset %lu, %lu, size %lu, %lu, stride %lu, %lu\n", offset[1],
           offset[0], size[1], size[0], stride[1], stride[0]);
  size_t dst_offset = 0;
  for (size_t j = 0; j < size[1]; j++)
    for (size_t i = 0; i < size[0]; i++) {
      size_t idx =
          ((offset[1] + j) * stride[1]) + ((offset[0] + i) * stride[0]);
      dst->data[dst_offset++] = src->data[idx];
    }
}

} // namespace

extern "C" void
_mlir_ciface_air_memcpy_nd_I32_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64_M0D2I32(
    uint32_t id, uint64_t x, uint64_t y, void *d, uint64_t offset1,
    uint64_t offset0, uint64_t size1, uint64_t size0, uint64_t stride1,
    uint64_t stride0, void *s) {
  tensor_t<int32_t, 2> *dst = (tensor_t<int32_t, 2> *)d;
  tensor_t<int32_t, 2> *src = (tensor_t<int32_t, 2> *)s;
  size_t offset[2] = {offset0, offset1};
  size_t size[2] = {size0, size1};
  size_t stride[2] = {stride0, stride1};

  if (VERBOSE)
    printf("id: %d, %lu, %lu ", id, x, y);
  air_memcpy_nd_dst(dst, src, offset, size, stride);
}

extern "C" void
_mlir_ciface_air_memcpy_nd_I32_I64_I64_M0D2I32_M0D2I32_I64_I64_I64_I64_I64_I64(
    uint32_t id, uint64_t x, uint64_t y, void *d, void *s, uint64_t offset1,
    uint64_t offset0, uint64_t size1, uint64_t size0, uint64_t stride1,
    uint64_t stride0) {
  tensor_t<int32_t, 2> *dst = (tensor_t<int32_t, 2> *)d;
  tensor_t<int32_t, 2> *src = (tensor_t<int32_t, 2> *)s;
  size_t offset[2] = {offset0, offset1};
  size_t size[2] = {size0, size1};
  size_t stride[2] = {stride0, stride1};

  if (VERBOSE)
    printf("id: %d, %lu, %lu", id, x, y);
  air_memcpy_nd_src(dst, src, offset, size, stride);
}

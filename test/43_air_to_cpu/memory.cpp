// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "air_tensor.h"

#ifndef VERBOSE
#define VERBOSE 0
#endif

namespace {

template <typename T, int R>
void air_alloc_tensor(tensor_t<T, R> *t, size_t *shape) {
  size_t n = 1;
  for (int i = 0; i < R; i++) {
    t->shape[i] = shape[i];
    t->stride[R - i - 1] = n;
    n = n * shape[i];
  }
  t->d = t->aligned = (T *)malloc(sizeof(T) * n);
  t->offset = 0;
}

template <typename T, int R> void air_dealloc_tensor(tensor_t<T, R> *t) {
  if (t->d)
    free(t->d);
  t->d = 0;
}

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
      dst->d[idx] = src->d[src_offset++];
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
      dst->d[dst_offset++] = src->d[idx];
    }
}

} // namespace

extern "C" {

void _mlir_ciface_air_alloc_rM1D2I32(void *t) {
  tensor_t<int32_t, 2> *tt = (tensor_t<int32_t, 2> *)t;
  size_t shape[2] = {64, 64};
  air_alloc_tensor(tt, shape);
}

void _mlir_ciface_air_alloc_rM2D2I32_I64_I64(void *t, uint64_t x, uint64_t y) {
  tensor_t<int32_t, 2> *tt = (tensor_t<int32_t, 2> *)t;
  size_t shape[2] = {32, 32};
  air_alloc_tensor(tt, shape);
}

void _mlir_ciface_air_dealloc_M1D2I32(void *t) {
  tensor_t<int32_t, 2> *tt = (tensor_t<int32_t, 2> *)t;
  air_dealloc_tensor(tt);
}

void _mlir_ciface_air_dealloc_I64_I64_M2D2I32(uint64_t x, uint64_t y, void *t) {
  _mlir_ciface_air_dealloc_M1D2I32(t);
}

void _mlir_ciface_air_memcpy_nd_I32_M1D2I32_M0D2I32_I64_I64_I64_I64_I64_I64(
    uint32_t id, void *d, void *s, uint64_t offset1, uint64_t offset0,
    uint64_t size1, uint64_t size0, uint64_t stride1, uint64_t stride0) {
  tensor_t<int32_t, 2> *dst = (tensor_t<int32_t, 2> *)d;
  tensor_t<int32_t, 2> *src = (tensor_t<int32_t, 2> *)s;
  size_t offset[2] = {offset0, offset1};
  size_t size[2] = {size0, size1};
  size_t stride[2] = {stride0, stride1};
  if (VERBOSE)
    printf("id: %d, ", id);
  air_memcpy_nd_src(dst, src, offset, size, stride);
}

void _mlir_ciface_air_memcpy_nd_I32_M0D2I32_I64_I64_I64_I64_I64_I64_M1D2I32(
    uint32_t id, void *d, uint64_t offset1, uint64_t offset0, uint64_t size1,
    uint64_t size0, uint64_t stride1, uint64_t stride0, void *s) {
  tensor_t<int32_t, 2> *dst = (tensor_t<int32_t, 2> *)d;
  tensor_t<int32_t, 2> *src = (tensor_t<int32_t, 2> *)s;
  size_t offset[2] = {offset0, offset1};
  size_t size[2] = {size0, size1};
  size_t stride[2] = {stride0, stride1};
  if (VERBOSE)
    printf("id: %d, ", id);
  air_memcpy_nd_dst(dst, src, offset, size, stride);
}

void _mlir_ciface_air_memcpy_nd_I32_I64_I64_M2D2I32_M1D2I32_I64_I64_I64_I64_I64_I64(
    uint32_t id, uint64_t x, uint64_t y, void *d, void *s, uint64_t offset1,
    uint64_t offset0, uint64_t size1, uint64_t size0, uint64_t stride1,
    uint64_t stride0) {
  if (VERBOSE)
    printf("x: %lu, y: %lu, ", x, y);
  _mlir_ciface_air_memcpy_nd_I32_M1D2I32_M0D2I32_I64_I64_I64_I64_I64_I64(
      id, d, s, offset1, offset0, size1, size0, stride1, stride0);
}

void _mlir_ciface_air_memcpy_nd_I32_I64_I64_M1D2I32_I64_I64_I64_I64_I64_I64_M2D2I32(
    uint32_t id, uint64_t x, uint64_t y, void *d, uint64_t offset1,
    uint64_t offset0, uint64_t size1, uint64_t size0, uint64_t stride1,
    uint64_t stride0, void *s) {
  if (VERBOSE)
    printf("x: %lu, y: %lu, ", x, y);
  _mlir_ciface_air_memcpy_nd_I32_M0D2I32_I64_I64_I64_I64_I64_I64_M1D2I32(
      id, d, offset1, offset0, size1, size0, stride1, stride0, s);
}

void _mlir_ciface_air_memcpy_nd_I32_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64_M2D2I32(
    uint32_t id, uint64_t x, uint64_t y, void *d, uint64_t offset1,
    uint64_t offset0, uint64_t size1, uint64_t size0, uint64_t stride1,
    uint64_t stride0, void *s) {
  if (VERBOSE)
    printf("x: %lu, y: %lu, ", x, y);
  _mlir_ciface_air_memcpy_nd_I32_M0D2I32_I64_I64_I64_I64_I64_I64_M1D2I32(
      id, d, offset1, offset0, size1, size0, stride1, stride0, s);
}

void _mlir_ciface_air_memcpy_nd_I32_I64_I64_M2D2I32_M0D2I32_I64_I64_I64_I64_I64_I64(
    uint32_t id, uint64_t x, uint64_t y, void *d, void *s, uint64_t offset1,
    uint64_t offset0, uint64_t size1, uint64_t size0, uint64_t stride1,
    uint64_t stride0) {
  if (VERBOSE)
    printf("x: %lu, y: %lu, ", x, y);
  _mlir_ciface_air_memcpy_nd_I32_M1D2I32_M0D2I32_I64_I64_I64_I64_I64_I64(
      id, d, s, offset1, offset0, size1, size0, stride1, stride0);
}
}
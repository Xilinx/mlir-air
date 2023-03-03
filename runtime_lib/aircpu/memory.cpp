// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "air_tensor.h"

#include <cstdint>
#include <cstdio>

#define VERBOSE 0

template <typename T, int R>
static void air_memcpy_nd_dst(tensor_t<T, R> *dst, tensor_t<T, R> *src,
                              size_t *_offset, size_t *_size, size_t *_stride) {
  size_t offset[4] = {0, 0, 0, 0};
  size_t size[4] = {1, 1, 1, 1};
  size_t stride[4] = {1, 1, 1, 1};
  for (int i = 0; i < R; i++) {
    offset[i] = _offset[i];
    size[i] = _size[i];
    stride[i] = _stride[i];
  }
  if (VERBOSE)
    printf("dst offset %lu, %lu, size %lu, %lu, stride %lu, %lu\n", offset[1],
           offset[0], size[1], size[0], stride[1], stride[0]);
  size_t src_offset = 0;
  for (size_t l = 0; l < size[3]; l++)
    for (size_t k = 0; k < size[2]; k++)
      for (size_t j = 0; j < size[1]; j++)
        for (size_t i = 0; i < size[0]; i++) {
          size_t idx =
              ((offset[3] + l) * stride[3]) + ((offset[2] + k) * stride[2]) +
              ((offset[1] + j) * stride[1]) + ((offset[0] + i) * stride[0]);
          dst->data[idx] = src->data[src_offset++];
        }
}

template <typename T, int R>
static void air_memcpy_nd_src(tensor_t<T, R> *dst, tensor_t<T, R> *src,
                              size_t *_offset, size_t *_size, size_t *_stride) {
  size_t offset[4] = {0, 0, 0, 0};
  size_t size[4] = {1, 1, 1, 1};
  size_t stride[4] = {1, 1, 1, 1};
  for (int i = 0; i < R; i++) {
    offset[i] = _offset[i];
    size[i] = _size[i];
    stride[i] = _stride[i];
  }
  if (VERBOSE)
    printf("src offset %lu, %lu, size %lu, %lu, stride %lu, %lu\n", offset[1],
           offset[0], size[1], size[0], stride[1], stride[0]);
  size_t dst_offset = 0;
  for (size_t l = 0; l < size[3]; l++)
    for (size_t k = 0; k < size[2]; k++)
      for (size_t j = 0; j < size[1]; j++)
        for (size_t i = 0; i < size[0]; i++) {
          size_t idx =
              ((offset[3] + l) * stride[3]) + ((offset[2] + k) * stride[2]) +
              ((offset[1] + j) * stride[1]) + ((offset[0] + i) * stride[0]);
          dst->data[dst_offset++] = src->data[idx];
        }
}

// 4D

template <typename T, int R>
static void
air_memcpy_nd_4d_src(uint32_t id, void *d, void *s, uint64_t offset3,
                     uint64_t offset2, uint64_t offset1, uint64_t offset0,
                     uint64_t size3, uint64_t size2, uint64_t size1,
                     uint64_t size0, uint64_t stride3, uint64_t stride2,
                     uint64_t stride1, uint64_t stride0) {
  tensor_t<T, R> *dst = (tensor_t<T, R> *)d;
  tensor_t<T, R> *src = (tensor_t<T, R> *)s;
  size_t offset[4] = {offset0, offset1, offset2, offset3};
  size_t size[4] = {size0, size1, size2, size3};
  size_t stride[4] = {stride0, stride1, stride2, stride3};
  if (VERBOSE)
    printf("id: %d, ", id);
  air_memcpy_nd_src(dst, src, offset, size, stride);
}

template <typename T, int R>
static void air_memcpy_nd_4d_dst(uint32_t id, void *d, uint64_t offset3,
                                 uint64_t offset2, uint64_t offset1,
                                 uint64_t offset0, uint64_t size3,
                                 uint64_t size2, uint64_t size1, uint64_t size0,
                                 uint64_t stride3, uint64_t stride2,
                                 uint64_t stride1, uint64_t stride0, void *s) {
  tensor_t<T, R> *dst = (tensor_t<T, R> *)d;
  tensor_t<T, R> *src = (tensor_t<T, R> *)s;
  size_t offset[4] = {offset0, offset1, offset2, offset3};
  size_t size[4] = {size0, size1, size2, size3};
  size_t stride[4] = {stride0, stride1, stride2, stride3};
  if (VERBOSE)
    printf("id: %d, ", id);
  air_memcpy_nd_dst(dst, src, offset, size, stride);
}

#define mlir_air_dma_nd_memcpy_4d_src(mangle, type)                            \
  void _mlir_ciface_air_memcpy_nd_##mangle(                                    \
      uint32_t id, void *d, void *s, uint64_t offset3, uint64_t offset2,       \
      uint64_t offset1, uint64_t offset0, uint64_t size3, uint64_t size2,      \
      uint64_t size1, uint64_t size0, uint64_t stride3, uint64_t stride2,      \
      uint64_t stride1, uint64_t stride0) {                                    \
    air_memcpy_nd_4d_src<type, 4>(id, d, s, offset3, offset2, offset1,         \
                                  offset0, size3, size2, size1, size0,         \
                                  stride3, stride2, stride1, stride0);         \
  }

#define mlir_air_dma_nd_memcpy_4d_dst(mangle, type)                            \
  void _mlir_ciface_air_memcpy_nd_##mangle(                                    \
      uint32_t id, void *d, uint64_t offset3, uint64_t offset2,                \
      uint64_t offset1, uint64_t offset0, uint64_t size3, uint64_t size2,      \
      uint64_t size1, uint64_t size0, uint64_t stride3, uint64_t stride2,      \
      uint64_t stride1, uint64_t stride0, void *s) {                           \
    air_memcpy_nd_4d_dst<type, 4>(id, d, offset3, offset2, offset1, offset0,   \
                                  size3, size2, size1, size0, stride3,         \
                                  stride2, stride1, stride0, s);               \
  }

#define mlir_air_dma_nd_memcpy_3d_src(mangle, type)                            \
  void _mlir_ciface_air_memcpy_nd_##mangle(                                    \
      uint32_t id, void *d, void *s, uint64_t offset2, uint64_t offset1,       \
      uint64_t offset0, uint64_t size2, uint64_t size1, uint64_t size0,        \
      uint64_t stride2, uint64_t stride1, uint64_t stride0) {                  \
    air_memcpy_nd_4d_src<type, 3>(id, d, s, 0, offset2, offset1, offset0, 1,   \
                                  size2, size1, size0, 1, stride2, stride1,    \
                                  stride0);                                    \
  }

#define mlir_air_dma_nd_memcpy_3d_dst(mangle, type)                            \
  void _mlir_ciface_air_memcpy_nd_##mangle(                                    \
      uint32_t id, void *d, uint64_t offset2, uint64_t offset1,                \
      uint64_t offset0, uint64_t size2, uint64_t size1, uint64_t size0,        \
      uint64_t stride2, uint64_t stride1, uint64_t stride0, void *s) {         \
    air_memcpy_nd_4d_dst<type, 3>(id, d, 0, offset2, offset1, offset0, 1,      \
                                  size2, size1, size0, 1, stride2, stride1,    \
                                  stride0, s);                                 \
  }

#define mlir_air_dma_nd_memcpy_2d_src(mangle, type)                            \
  void _mlir_ciface_air_memcpy_nd_##mangle(                                    \
      uint32_t id, void *d, void *s, uint64_t offset1, uint64_t offset0,       \
      uint64_t size1, uint64_t size0, uint64_t stride1, uint64_t stride0) {    \
    air_memcpy_nd_4d_src<type, 2>(id, d, s, 0, 0, offset1, offset0, 1, 1,      \
                                  size1, size0, 1, 1, stride1, stride0);       \
  }

#define mlir_air_dma_nd_memcpy_2d_dst(mangle, type)                            \
  void _mlir_ciface_air_memcpy_nd_##mangle(                                    \
      uint32_t id, void *d, uint64_t offset1, uint64_t offset0,                \
      uint64_t size1, uint64_t size0, uint64_t stride1, uint64_t stride0,      \
      void *s) {                                                               \
    air_memcpy_nd_4d_dst<type, 2>(id, d, 0, 0, offset1, offset0, 1, 1, size1,  \
                                  size0, 1, 1, stride1, stride0, s);           \
  }

#define mlir_air_dma_nd_memcpy_1d_src(mangle, type)                            \
  void _mlir_ciface_air_memcpy_nd_##mangle(uint32_t id, void *d, void *s,      \
                                           uint64_t offset0, uint64_t size0,   \
                                           uint64_t stride0) {                 \
    air_memcpy_nd_4d_src<type, 1>(id, d, s, 0, 0, 0, offset0, 1, 1, 1, size0,  \
                                  1, 1, 1, stride0);                           \
  }

#define mlir_air_dma_nd_memcpy_1d_dst(mangle, type)                            \
  void _mlir_ciface_air_memcpy_nd_##mangle(uint32_t id, void *d,               \
                                           uint64_t offset0, uint64_t size0,   \
                                           uint64_t stride0, void *s) {        \
    air_memcpy_nd_4d_dst<type, 1>(id, d, 0, 0, 0, offset0, 1, 1, 1, size0, 1,  \
                                  1, 1, stride0, s);                           \
  }
extern "C" {

// 4D

mlir_air_dma_nd_memcpy_4d_src(
    I32_M0D4I32_M0D4I32_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64,
    int32_t);
mlir_air_dma_nd_memcpy_4d_dst(
    I32_M0D4I32_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_M0D4I32,
    int32_t);

mlir_air_dma_nd_memcpy_4d_src(
    I32_M0D4F32_M0D4F32_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64, float);
mlir_air_dma_nd_memcpy_4d_dst(
    I32_M0D4F32_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_M0D4F32, float);

// 3D

mlir_air_dma_nd_memcpy_3d_dst(
    I32_M0D3I32_I64_I64_I64_I64_I64_I64_I64_I64_I64_M0D3I32, int32_t);
mlir_air_dma_nd_memcpy_3d_src(
    I32_M0D3I32_M0D3I32_I64_I64_I64_I64_I64_I64_I64_I64_I64, int32_t);

mlir_air_dma_nd_memcpy_3d_dst(
    I32_M0D3F32_I64_I64_I64_I64_I64_I64_I64_I64_I64_M0D3F32, float);
mlir_air_dma_nd_memcpy_3d_src(
    I32_M0D3F32_M0D3F32_I64_I64_I64_I64_I64_I64_I64_I64_I64, float);

// 2D

mlir_air_dma_nd_memcpy_2d_src(I32_M0D2I32_M0D2I32_I64_I64_I64_I64_I64_I64,
                              int32_t);
mlir_air_dma_nd_memcpy_2d_dst(I32_M0D2I32_I64_I64_I64_I64_I64_I64_M0D2I32,
                              int32_t);

mlir_air_dma_nd_memcpy_2d_src(I32_M0D2F32_M0D2F32_I64_I64_I64_I64_I64_I64,
                              float);
mlir_air_dma_nd_memcpy_2d_dst(I32_M0D2F32_I64_I64_I64_I64_I64_I64_M0D2F32,
                              float);

// 1D

mlir_air_dma_nd_memcpy_1d_src(I32_M0D1I32_M0D1I32_I64_I64_I64, int32_t);
mlir_air_dma_nd_memcpy_1d_dst(I32_M0D1I32_I64_I64_I64_M0D1I32, int32_t);

mlir_air_dma_nd_memcpy_1d_dst(I32_M0D1F32_I64_I64_I64_M0D1F32, float);
mlir_air_dma_nd_memcpy_1d_src(I32_M0D1F32_M0D1F32_I64_I64_I64, float);
}

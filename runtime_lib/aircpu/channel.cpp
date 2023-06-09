// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "air_channel.h"
#include "air_tensor.h"

#include <iostream>

#define VERBOSE 0

template <typename T, int R>
static void _air_channel_put(tensor_t<uint64_t, 0> *channel,
                             tensor_t<T, R> *src, size_t *_offset,
                             size_t *_size, size_t *_stride) {
  size_t offset[4] = {0, 0, 0, 0};
  size_t size[4] = {1, 1, 1, 1};
  size_t stride[4] = {1, 1, 1, 1};
  for (int i = 0; i < R; i++) {
    offset[i] = _offset[i];
    size[i] = _size[i];
    stride[i] = _stride[i];
  }

  // channel->data is an array of pointers to channel_t objects
  // if the channel is valid, channel->data[0] should be a valid memory address
  // otherwise, allocate a new channel
  if (channel->data[0] == 0) {
    channel_t<T> *new_channel = (channel_t<T> *)malloc(sizeof(channel_t<T>));
    new_channel->data =
        (T *)malloc(sizeof(T) * size[3] * size[2] * size[0] * size[1]);
    new_channel->_is_full = false;
    channel->data[0] = (uint64_t)new_channel;
  }

  channel_t<T> *chan = (channel_t<T> *)channel->data[0];

  // wait until the channel is empty
  while (chan->_is_full)
    ;

  if (VERBOSE)
    std::cerr << "dst offset " << offset[1] << ", " << offset[0] << ", size "
              << size[1] << ", " << size[0] << ", stride " << stride[1] << ", "
              << stride[0] << std::endl;
  size_t src_offset = 0;
  for (size_t l = 0; l < size[3]; l++)
    for (size_t k = 0; k < size[2]; k++)
      for (size_t j = 0; j < size[1]; j++)
        for (size_t i = 0; i < size[0]; i++) {
          size_t idx =
              ((offset[3] + l) * stride[3]) + ((offset[2] + k) * stride[2]) +
              ((offset[1] + j) * stride[1]) + ((offset[0] + i) * stride[0]);
          chan->data[src_offset++] = src->data[idx];
        }

  // mark the channel as full
  chan->_is_full = true;
}

template <typename T, int R>
static void _air_channel_get(tensor_t<uint64_t, 0> *channel,
                             tensor_t<T, R> *dst, size_t *_offset,
                             size_t *_size, size_t *_stride) {
  // get the buffer address from src
  channel_t<T> *chan = (channel_t<T> *)channel->data[0];
  // test if buffer points to a valid address
  if (chan == nullptr) {
    std::cerr << "channel has an invalid memory address " << chan << std::endl;
    exit(1);
  }
  size_t offset[4] = {0, 0, 0, 0};
  size_t size[4] = {1, 1, 1, 1};
  size_t stride[4] = {1, 1, 1, 1};
  for (int i = 0; i < R; i++) {
    offset[i] = _offset[i];
    size[i] = _size[i];
    stride[i] = _stride[i];
  }
  if (VERBOSE)
    std::cerr << "dst offset " << offset[1] << ", " << offset[0] << ", size "
              << size[1] << ", " << size[0] << ", stride " << stride[1] << ", "
              << stride[0] << std::endl;

  // wait until the channel is full
  while (!chan->_is_full)
    ;

  // copy data from buffer to dst
  size_t dst_offset = 0;
  for (size_t l = 0; l < size[3]; l++)
    for (size_t k = 0; k < size[2]; k++)
      for (size_t j = 0; j < size[1]; j++)
        for (size_t i = 0; i < size[0]; i++) {
          size_t idx =
              ((offset[3] + l) * stride[3]) + ((offset[2] + k) * stride[2]) +
              ((offset[1] + j) * stride[1]) + ((offset[0] + i) * stride[0]);
          dst->data[idx] = chan->data[dst_offset++];
        }

  // mark the channel as empty
  chan->_is_full = false;
}

template <typename T, int R>
static void air_channel_get(void *c, void *d, uint64_t offset3,
                            uint64_t offset2, uint64_t offset1,
                            uint64_t offset0, uint64_t size3, uint64_t size2,
                            uint64_t size1, uint64_t size0, uint64_t stride3,
                            uint64_t stride2, uint64_t stride1,
                            uint64_t stride0) {
  tensor_t<uint64_t, 0> *channel = (tensor_t<uint64_t, 0> *)c;
  tensor_t<T, R> *dst = (tensor_t<T, R> *)d;
  size_t offset[4] = {offset0, offset1, offset2, offset3};
  size_t size[4] = {size0, size1, size2, size3};
  size_t stride[4] = {stride0, stride1, stride2, stride3};
  _air_channel_get<T, R>(channel, dst, offset, size, stride);
}

template <typename T, int R>
static void air_channel_put(void *c, void *s, uint64_t offset3,
                            uint64_t offset2, uint64_t offset1,
                            uint64_t offset0, uint64_t size3, uint64_t size2,
                            uint64_t size1, uint64_t size0, uint64_t stride3,
                            uint64_t stride2, uint64_t stride1,
                            uint64_t stride0) {
  tensor_t<uint64_t, 0> *channel = (tensor_t<uint64_t, 0> *)c;
  tensor_t<T, R> *src = (tensor_t<T, R> *)s;
  size_t offset[4] = {offset0, offset1, offset2, offset3};
  size_t size[4] = {size0, size1, size2, size3};
  size_t stride[4] = {stride0, stride1, stride2, stride3};
  _air_channel_put<T, R>(channel, src, offset, size, stride);
}

// 4D
#define mlir_air_channel_get_4d(mangle, type)                                  \
  void _mlir_ciface_air_channel_get_##mangle(                                  \
      void *c, void *d, uint64_t offset3, uint64_t offset2, uint64_t offset1,  \
      uint64_t offset0, uint64_t size3, uint64_t size2, uint64_t size1,        \
      uint64_t size0, uint64_t stride3, uint64_t stride2, uint64_t stride1,    \
      uint64_t stride0) {                                                      \
    air_channel_get<type, 4>(c, d, offset3, offset2, offset1, offset0, size3,  \
                             size2, size1, size0, stride3, stride2, stride1,   \
                             stride0);                                         \
  }

#define mlir_air_channel_put_4d(mangle, type)                                  \
  void _mlir_ciface_air_channel_put_##mangle(                                  \
      void *c, void *s, uint64_t offset3, uint64_t offset2, uint64_t offset1,  \
      uint64_t offset0, uint64_t size3, uint64_t size2, uint64_t size1,        \
      uint64_t size0, uint64_t stride3, uint64_t stride2, uint64_t stride1,    \
      uint64_t stride0) {                                                      \
    air_channel_put<type, 4>(c, s, offset3, offset2, offset1, offset0, size3,  \
                             size2, size1, size0, stride3, stride2, stride1,   \
                             stride0);                                         \
  }

// 3D
#define mlir_air_channel_get_3d(mangle, type)                                  \
  void _mlir_ciface_air_channel_get_##mangle(                                  \
      void *c, void *d, uint64_t offset2, uint64_t offset1, uint64_t offset0,  \
      uint64_t size2, uint64_t size1, uint64_t size0, uint64_t stride2,        \
      uint64_t stride1, uint64_t stride0) {                                    \
    air_channel_get<type, 3>(c, d, 0, offset2, offset1, offset0, 1, size2,     \
                             size1, size0, 1, stride2, stride1, stride0);      \
  }

#define mlir_air_channel_put_3d(mangle, type)                                  \
  void _mlir_ciface_air_channel_put_##mangle(                                  \
      void *c, void *s, uint64_t offset2, uint64_t offset1, uint64_t offset0,  \
      uint64_t size2, uint64_t size1, uint64_t size0, uint64_t stride2,        \
      uint64_t stride1, uint64_t stride0) {                                    \
    air_channel_put<type, 3>(c, s, 0, offset2, offset1, offset0, 1, size2,     \
                             size1, size0, 1, stride2, stride1, stride0);      \
  }

// 2D
#define mlir_air_channel_get_2d(mangle, type)                                  \
  void _mlir_ciface_air_channel_get_##mangle(                                  \
      void *c, void *d, uint64_t offset1, uint64_t offset0, uint64_t size1,    \
      uint64_t size0, uint64_t stride1, uint64_t stride0) {                    \
    air_channel_get<type, 2>(c, d, 0, 0, offset1, offset0, 1, 1, size1, size0, \
                             1, 1, stride1, stride0);                          \
  }

#define mlir_air_channel_put_2d(mangle, type)                                  \
  void _mlir_ciface_air_channel_put_##mangle(                                  \
      void *c, void *s, uint64_t offset1, uint64_t offset0, uint64_t size1,    \
      uint64_t size0, uint64_t stride1, uint64_t stride0) {                    \
    air_channel_put<type, 2>(c, s, 0, 0, offset1, offset0, 1, 1, size1, size0, \
                             1, 1, stride1, stride0);                          \
  }

// 1D
#define mlir_air_channel_get_1d(mangle, type)                                  \
  void _mlir_ciface_air_channel_get_##mangle(                                  \
      void *c, void *d, uint64_t offset0, uint64_t size0, uint64_t stride0) {  \
    air_channel_get<type, 1>(c, d, 0, 0, 0, offset0, 1, 1, 1, size0, 1, 1, 1,  \
                             stride0);                                         \
  }

#define mlir_air_channel_put_1d(mangle, type)                                  \
  void _mlir_ciface_air_channel_put_##mangle(                                  \
      void *c, void *s, uint64_t offset0, uint64_t size0, uint64_t stride0) {  \
    air_channel_put<type, 1>(c, s, 0, 0, 0, offset0, 1, 1, 1, size0, 1, 1, 1,  \
                             stride0);                                         \
  }

extern "C" {
// 4D
mlir_air_channel_get_4d(
    M0I64_M0D4I32_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64, int32_t);
mlir_air_channel_put_4d(
    M0I64_M0D4I32_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64, int32_t);
mlir_air_channel_get_4d(
    M0I64_M0D4F32_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64, float);
mlir_air_channel_put_4d(
    M0I64_M0D4F32_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64_I64, float);

// 3D
mlir_air_channel_get_3d(M0I64_M0D3I32_I64_I64_I64_I64_I64_I64_I64_I64_I64,
                        int32_t);
mlir_air_channel_put_3d(M0I64_M0D3I32_I64_I64_I64_I64_I64_I64_I64_I64_I64,
                        int32_t);
mlir_air_channel_get_3d(M0I64_M0D3F32_I64_I64_I64_I64_I64_I64_I64_I64_I64,
                        float);
mlir_air_channel_put_3d(M0I64_M0D3F32_I64_I64_I64_I64_I64_I64_I64_I64_I64,
                        float);

// 2D
mlir_air_channel_get_2d(M0I64_M0D2I32_I64_I64_I64_I64_I64_I64, int32_t);
mlir_air_channel_put_2d(M0I64_M0D2I32_I64_I64_I64_I64_I64_I64, int32_t);
mlir_air_channel_get_2d(M0I64_M0D2F32_I64_I64_I64_I64_I64_I64, float);
mlir_air_channel_put_2d(M0I64_M0D2F32_I64_I64_I64_I64_I64_I64, float);

// 1D
mlir_air_channel_get_1d(M0I64_M0D1I32_I64_I64_I64, int32_t);
mlir_air_channel_put_1d(M0I64_M0D1I32_I64_I64_I64, int32_t);
mlir_air_channel_get_1d(M0I64_M0D1F32_I64_I64_I64, float);
mlir_air_channel_put_1d(M0I64_M0D1F32_I64_I64_I64, float);
}

// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "air_tensor.h"

#ifndef VERBOSE
#define VERBOSE 0
#endif

using namespace boost;

namespace {

typedef size_t air_signal_t;
typedef int64_t air_signal_value_t;

using GraphvizAttributes = 
    std::map<std::string, std::string>;
using Graph =
    adjacency_list<vecS, vecS, directedS,
      property<vertex_attribute_t, int, GraphvizAttributes>,
      property<edge_index_t, int, property<edge_attribute_t, GraphvizAttributes> >,
      property<graph_name_t, std::string,
        property<graph_graph_attribute_t,  GraphvizAttributes,
        property<graph_vertex_attribute_t, GraphvizAttributes,
        property<graph_edge_attribute_t,   GraphvizAttributes>
      > > >
    >;

typedef std::pair<air_signal_t, air_signal_t> Edge;
std::vector<Edge> deps;
std::vector<air_signal_t> nodes;

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

void air_signal_create(air_signal_value_t initial_value, air_signal_t *out) {
  air_signal_value_t *value = (air_signal_value_t *)malloc(sizeof(int64_t));
  *value = initial_value;
  air_signal_t v = (air_signal_t)value;
  nodes.push_back(v);
  *out = v;
}

air_signal_value_t air_signal_load(air_signal_t signal)
{
  return *((air_signal_value_t*)signal);
}

void air_signal_subtract(air_signal_t signal, air_signal_value_t value) {
  air_signal_value_t *s = (air_signal_value_t *)signal;
  *s = *s - value;
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

void dump_graph(char *filename)
{
  std::ofstream ofs (filename, std::ofstream::out); 
  Graph g(deps.begin(), deps.end(), nodes.size());
  write_graphviz(ofs, g);
}

uint64_t _mlir_ciface_air_wait_all_rE(void) {
  air_signal_t evt;
  air_signal_create(0, &evt);
  return evt;
}

void _mlir_ciface_air_wait_all_E_E(uint64_t e0, uint64_t e1) {
  air_signal_t evt;
  air_signal_create(0, &evt);
  auto ep = find(nodes.begin(), nodes.end(), evt) - nodes.begin();
  auto e0p = find(nodes.begin(), nodes.end(), e0) - nodes.begin();
  auto e1p = find(nodes.begin(), nodes.end(), e1) - nodes.begin();
  deps.push_back({ep, e0p});
  deps.push_back({ep, e1p});
}

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

uint64_t _mlir_ciface_air_memcpy_nd_rE_I32_E_M1D2I32_M0D2I32_I64_I64_I64_I64_I64_I64(
    uint32_t id, uint64_t e0, void *d, void *s, uint64_t offset1, uint64_t offset0,
    uint64_t size1, uint64_t size0, uint64_t stride1, uint64_t stride0) {
  tensor_t<int32_t, 2> *dst = (tensor_t<int32_t, 2> *)d;
  tensor_t<int32_t, 2> *src = (tensor_t<int32_t, 2> *)s;
  size_t offset[2] = {offset0, offset1};
  size_t size[2] = {size0, size1};
  size_t stride[2] = {stride0, stride1};
  
  air_signal_t evt;
  air_signal_create(1, &evt);
  
  auto ep = find(nodes.begin(), nodes.end(), evt) - nodes.begin();
  auto e0p = find(nodes.begin(), nodes.end(), e0) - nodes.begin();
  deps.push_back({ep, e0p});
  while(air_signal_load(e0)) {
    ;// blocked
    // should not happen in sequential impl
    exit(1);
  }

  if (VERBOSE)
    printf("id: %d, ", id);
  air_memcpy_nd_src(dst, src, offset, size, stride);

  air_signal_subtract(evt, 1);
  return evt;
}

uint64_t _mlir_ciface_air_memcpy_nd_rE_I32_M1D2I32_M0D2I32_I64_I64_I64_I64_I64_I64(
    uint32_t id, void *d, void *s, uint64_t offset1, uint64_t offset0,
    uint64_t size1, uint64_t size0, uint64_t stride1, uint64_t stride0) {
  tensor_t<int32_t, 2> *dst = (tensor_t<int32_t, 2> *)d;
  tensor_t<int32_t, 2> *src = (tensor_t<int32_t, 2> *)s;
  size_t offset[2] = {offset0, offset1};
  size_t size[2] = {size0, size1};
  size_t stride[2] = {stride0, stride1};
  air_signal_t evt;
  if (VERBOSE)
    printf("id: %d, ", id);
  air_signal_create(1, &evt);
  air_memcpy_nd_src(dst, src, offset, size, stride);
  air_signal_subtract(evt, 1);
  return evt;
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

uint64_t _mlir_ciface_air_memcpy_nd_rE_I32_E_M0D2I32_I64_I64_I64_I64_I64_I64_M1D2I32(
    uint32_t id, uint64_t e0, void *d, uint64_t offset1, uint64_t offset0, uint64_t size1,
    uint64_t size0, uint64_t stride1, uint64_t stride0, void *s) {
  tensor_t<int32_t, 2> *dst = (tensor_t<int32_t, 2> *)d;
  tensor_t<int32_t, 2> *src = (tensor_t<int32_t, 2> *)s;
  size_t offset[2] = {offset0, offset1};
  size_t size[2] = {size0, size1};
  size_t stride[2] = {stride0, stride1};
  air_signal_t evt;
  if (VERBOSE)
    printf("id: %d, ", id);

  air_signal_create(1, &evt);
  auto ep = find(nodes.begin(), nodes.end(), evt) - nodes.begin();
  auto e0p = find(nodes.begin(), nodes.end(), e0) - nodes.begin();

  deps.push_back({ep, e0p});
  while(air_signal_load(e0)) {
    ;// blocked
    // should not happen in sequential impl
    exit(1);
  }

  air_memcpy_nd_dst(dst, src, offset, size, stride);
  air_signal_subtract(evt, 1);
  return evt;
}

uint64_t _mlir_ciface_air_memcpy_nd_rE_I32_E_E_E_M0D2I32_I64_I64_I64_I64_I64_I64_M1D2I32(
    uint32_t id, uint64_t e0, uint64_t e1, uint64_t e2, void *d, uint64_t offset1, uint64_t offset0, uint64_t size1,
    uint64_t size0, uint64_t stride1, uint64_t stride0, void *s) {
  tensor_t<int32_t, 2> *dst = (tensor_t<int32_t, 2> *)d;
  tensor_t<int32_t, 2> *src = (tensor_t<int32_t, 2> *)s;
  size_t offset[2] = {offset0, offset1};
  size_t size[2] = {size0, size1};
  size_t stride[2] = {stride0, stride1};
  air_signal_t evt;
  if (VERBOSE)
    printf("id: %d, ", id);

  air_signal_create(1, &evt);
  auto ep = find(nodes.begin(), nodes.end(), evt) - nodes.begin();
  auto e0p = find(nodes.begin(), nodes.end(), e0) - nodes.begin();
  auto e1p = find(nodes.begin(), nodes.end(), e1) - nodes.begin();
  auto e2p = find(nodes.begin(), nodes.end(), e2) - nodes.begin();
  deps.push_back({ep, e0p});
  deps.push_back({ep, e1p});
  deps.push_back({ep, e2p});
  while(air_signal_load(e0) || air_signal_load(e1) || air_signal_load(e2)) {
    ;// blocked
    // should not happen in sequential impl
    exit(1);
  }

  air_memcpy_nd_dst(dst, src, offset, size, stride);
  air_signal_subtract(evt, 1);
  return evt;
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
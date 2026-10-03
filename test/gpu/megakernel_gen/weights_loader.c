//===- weights_loader.c -----------------------------------------*- C -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// One extern the generated chain calls to fill a weight buffer from the blob
// weights.py wrote. mlir-runner resolves it out of a --shared-libs library, and
// the chain passes a raw pointer it already has (it uses
// memref.extract_aligned_pointer_as_index elsewhere), so there is no memref
// descriptor to agree about across the ABI.
//
// Every failure here is fatal rather than a short read left to look like a
// numerical difference later.
//
//   clang -O2 -shared -fPIC -o libairweights.so weights_loader.c
//
//===------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static FILE *blob = NULL;

static void air_weights_open(void) {
  const char *path = getenv("AIR_WEIGHTS");
  if (!path) {
    fprintf(stderr, "air_load_weights: AIR_WEIGHTS is not set\n");
    exit(1);
  }
  blob = fopen(path, "rb");
  if (!blob) {
    perror(path);
    exit(1);
  }
}

// Read `count` floats starting at float index `offset` into `dst`.
void air_load_weights(float *dst, int64_t offset, int64_t count) {
  if (!blob)
    air_weights_open();
  if (fseeko(blob, (off_t)offset * 4, SEEK_SET) != 0) {
    perror("air_load_weights: seek");
    exit(1);
  }
  size_t got = fread(dst, sizeof(float), (size_t)count, blob);
  if (got != (size_t)count) {
    fprintf(stderr, "air_load_weights: wanted %lld floats at %lld, got %zu\n",
            (long long)count, (long long)offset, got);
    exit(1);
  }
}

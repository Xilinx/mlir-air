# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Host-side KV cache buffers for the fused decode.
#
# A decode build may split its KV cache over several buffers because one buffer
# is only addressable over 4 GiB (see kv_layout in fused_decode.py). Drivers
# allocate, seed and bind the cache through KVBuffers so the host layout always
# comes from the builder.

import numpy as np


class KVBuffers:
    """The KV buffers of a decode build at `attn_maxl`.

    `fd` is the imported builder; its import-time ATTN_MAXL does not matter,
    the layout comes from fd.kv_layout(attn_maxl). `make_bo(nbytes, arg)`
    allocates one buffer, where `arg` is its index among the host buffer args
    (x=0 .. kvc=4); an xclbin kernel's group id is group_id(3 + arg).
    """

    def __init__(self, fd, attn_maxl, make_bo):
        self.layer, self.group, sizes, self.args = fd.kv_layout(attn_maxl)
        self.bos = [make_bo(n * 2, a) for n, a in zip(sizes, self.args)]

    @property
    def kvc(self):
        """Group 0, bound as the kvc arg."""
        return self.bos[0]

    @property
    def extra(self):
        """Groups 1 and up, bound after every other buffer arg."""
        return tuple(self.bos[1:])

    def locate(self, layer):
        """(buffer, byte offset) of `layer`'s slab."""
        g = layer // self.group
        return self.bos[g], (layer - g * self.group) * self.layer * 2

    def write_layer(self, layer, arr, to_dev):
        """Write `arr` (int16 view of bf16) at the start of `layer`'s slab and
        sync just that range."""
        bo, off = self.locate(layer)
        bo.write(arr, off)
        bo.sync(to_dev, arr.nbytes, off)

    def write_layers(self, kv, to_dev, conv=None):
        """Write a [n_layers, KV_LAYER] int16 array, one buffer at a time, and a
        hybrid's ShortConv state (`conv`, int16), which follows group 0's slabs."""
        for g, bo in enumerate(self.bos):
            rows = np.ascontiguousarray(kv[g * self.group : (g + 1) * self.group])
            bo.write(rows.reshape(-1), 0)
            bo.sync(to_dev, rows.nbytes, 0)
        if conv is not None:
            conv = np.ascontiguousarray(conv).reshape(-1)
            off = self.group * self.layer * 2
            self.bos[0].write(conv, off)
            self.bos[0].sync(to_dev, conv.nbytes, off)

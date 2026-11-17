# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# KV cache buffers of a decode build.
#
# A decode build may split its KV cache over several buffers because one buffer
# is only addressable over 4 GiB. Builders size the buffers with kv_groups, in
# their kv_layout; drivers allocate, seed and bind the cache through KVBuffers
# so the host layout always comes from the builder.

import numpy as np

# A shim BD offset is an i32 count of bf16 elements (AIRRtToNpuPass,
# aie.dma_bd), so it reaches 4 GiB into a buffer. Past that it goes negative: an
# xclbin dispatch then reads and writes outside the buffer without any error,
# and a full ELF fails to link (aiebu rejects relocation addends wider than 32
# bits).
BO_REACH_ELEMS = (1 << 32) // 2


def kv_groups(n_slabs, slab, extra0=0):
    """Split `n_slabs` KV slabs of `slab` elements evenly over the fewest
    buffers that a shim BD reaches, buffer 0 also holding `extra0` elements.

    Returns (slabs per buffer, KV elements of each buffer).
    """
    if slab > BO_REACH_ELEMS - extra0:
        raise SystemExit(
            f"one layer's KV slab is {slab * 2 / 2**30:.2f} GiB, past the 4 GiB "
            "a shim BD reaches; lower the context"
        )
    n = -(-n_slabs // ((BO_REACH_ELEMS - extra0) // slab))
    group = -(-n_slabs // n)
    return group, [min(group, n_slabs - g * group) * slab for g in range(n)]


class KVBuffers:
    """The KV buffers of a decode build at `attn_maxl`.

    `fd` is the imported builder; its import-time ATTN_MAXL does not matter,
    the layout comes from fd.kv_layout(attn_maxl). `make_bo(nbytes, arg)`
    allocates one buffer, where `arg` is its index among the host buffer args
    (x=0 .. kvc=4); an xclbin kernel's group id is group_id(3 + arg).
    """

    def __init__(self, fd, attn_maxl, make_bo):
        self.layer, self.group, self.sizes, self.args = fd.kv_layout(attn_maxl)
        self.bos = [make_bo(n * 2, a) for n, a in zip(self.sizes, self.args)]

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

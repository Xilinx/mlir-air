# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Physical argument layout and bounded logical views for grouped decoders."""

import numpy as np


def compact_abi(interface):
    """Keep state in one BO and append the LM head to the last weight group.

    All offsets and sizes in this interface are bytes. The AIR builder emits
    matching BF16 subviews before compilation; no transaction/header rewriting
    is needed. Logical argument numbers retain x/w/r/y/KV, then extra weights.
    """
    sizes = interface["elements"]
    views = {}
    state_bytes = 0
    for name, argument in (("x", 0), ("r", 2), ("y", 3), ("kv", 4)):
        state_bytes = (state_bytes + 4095) // 4096 * 4096
        views[str(argument)] = dict(
            buffer=0, offset_bytes=state_bytes, bytes=sizes[name] * 2, name=name
        )
        state_bytes += sizes[name] * 2
    groups = interface["weight_buffers"]
    body = groups[:-1] if len(groups) > 1 else groups
    buffers = [dict(argument=0, bytes=state_bytes)]
    for index, item in enumerate(body, 1):
        count = item["elements"] * 2
        buffers.append(dict(argument=index, bytes=count))
        views[str(item["argument"])] = dict(
            buffer=index, offset_bytes=0, bytes=count, name=f"w{index - 1}"
        )
    if len(groups) > 1:
        head = groups[-1]
        views[str(head["argument"])] = dict(
            buffer=len(buffers) - 1,
            offset_bytes=buffers[-1]["bytes"],
            bytes=head["elements"] * 2,
            name="head",
        )
        buffers[-1]["bytes"] += head["elements"] * 2
    if any(item["bytes"] >= 2**32 for item in buffers):
        raise ValueError("A grouped decoder buffer exceeds 32-bit DMA addressing")
    return dict(version=1, buffers=buffers, logical_views=views, runtime_scalar=True)


class BufferView:
    """A logical tensor's disjoint byte range in a physically bound XRT BO."""

    def __init__(self, parent, offset, size):
        self.parent, self.offset, self.size = parent, int(offset), int(size)

    def map(self):
        return memoryview(self.parent.map()).cast("B")[
            self.offset : self.offset + self.size
        ]

    def write(self, data, offset=0):
        if offset < 0 or offset + np.asarray(data).nbytes > self.size:
            raise ValueError("Logical BO write exceeds its reserved span")
        self.parent.write(data, self.offset + offset)

    def sync(self, direction, size=None, offset=0):
        size = self.size - offset if size is None else size
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ValueError("Logical BO sync exceeds its reserved span")
        self.parent.sync(direction, size, self.offset + offset)

    def address(self):
        return self.parent.address() + self.offset

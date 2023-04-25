// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: GPL-2.0

#ifndef AMDAIR_HEADER_H_
#define AMDAIR_HEADER_H_

#define AMDAIR_MMAP_RANGE_BRAM 0x0
#define AMDAIR_MMAP_RANGE_DRAM 0x1
#define AMDAIR_MMAP_RANGE_AIE 0x8

#define AMDAIR_IOCTL_MAJOR_VERSION 1
#define AMDAIR_IOCTL_MINOR_VERSION 0

struct amdair_get_version_args {
	uint32_t major_version; /* from driver */
	uint32_t minor_version; /* from driver */
};

enum amdair_queue_type {
	AMDAIR_QUEUE_HOST, /* queue is in host memory */
	AMDAIR_QUEUE_DEVICE, /* queue is in device memory */
	AMDAIR_QUEUE_3RD_PARTY, /* queue is not in device or host memory */
};

struct amdair_create_queue_args {
	uint64_t ring_base_address; /* to driver */
	uint64_t write_pointer_address; /* from driver */
	uint64_t read_pointer_address; /* from driver */
	uint64_t doorbell_offset; /* from driver */

	uint32_t ring_size; /* to driver */
	uint32_t device_id; /* to driver */
	uint32_t queue_type; /* to driver */
	uint32_t queue_id; /* from driver */
};

struct amdair_destroy_queue_args {
	uint32_t queue_id; /* from driver */
};

#define AMDAIR_COMMAND_START 1
#define AMDAIR_COMMAND_END 3

#define AMDAIR_IOCTL_BASE 'K'
#define AMDAIR_IO(nr) _IO(AMDAIR_IOCTL_BASE, nr)
#define AMDAIR_IOR(nr, type) _IOR(AMDAIR_IOCTL_BASE, nr, type)
#define AMDAIR_IOW(nr, type) _IOW(AMDAIR_IOCTL_BASE, nr, type)
#define AMDAIR_IOWR(nr, type) _IOWR(AMDAIR_IOCTL_BASE, nr, type)

#define AMDAIR_IOC_GET_VERSION AMDAIR_IOR(0x01, struct amdair_get_version_args)

#define AMDAIR_IOC_CREATE_QUEUE                                                \
	AMDAIR_IOWR(0x02, struct amdair_create_queue_args)

#define AMDAIR_IOC_DESTROY_QUEUE                                               \
	AMDAIR_IOWR(0x03, struct amdair_destroy_queue_args)

/*
The mmap interface is not designed to handle multiple devices or address
ranges, so we must encode multiple fields into the offset.

[39:36] device ID
[35:32] range (BRAM, DRAM, etc.)
[31:12] offset
[11:0] empty
*/
#define AMDAIR_MMAP_OFFSET_MASK 0xFFFFF000UL
#define AMDAIR_MMAP_ENCODE_OFFSET(_range, _offset, _devid)                     \
	((_offset & AMDAIR_MMAP_OFFSET_MASK) |                                 \
	 (((uint64_t)AMDAIR_MMAP_RANGE_##_range & 0xF) << 32) |                \
	 (((uint64_t)_devid & 0xF) << 36))

#define AMDAIR_MMAP_DECODE_OFFSET_DEVID(_offset) ((_offset >> 36) & 0xF)
#define AMDAIR_MMAP_DECODE_OFFSET_RANGE(_offset) ((_offset >> 32) & 0xF)
#define AMDAIR_MMAP_DECODE_OFFSET(_offset) ((_offset)&AMDAIR_MMAP_OFFSET_MASK)

#endif /* AMDAIR_HEADER_H_ */

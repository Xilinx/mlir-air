// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: GPL-2.0

#ifndef DEVICE_H_
#define DEVICE_H_

#define MAX_HERD_CONTROLLERS 64

/* For now, there is a 1:1 relationship between queues and controllers */
#define MAX_QUEUES MAX_HERD_CONTROLLERS

/*
	This represents a single VCK5000 card.

	This does not use the Linux kernel 'device' infrastructure because we want
	to have only a single character device interface (/dev/amdair) regardless of
	how many physical devices are attached. This follows the AMDKFD driver
	design.
*/
struct vck5000_device {
	struct list_head list;
	uint32_t device_id;
	struct kobject kobj_aie;

	void __iomem *dram_bar;
	uint64_t dram_bar_len;

	void __iomem *aie_bar;
	uint64_t aie_bar_len;

	void __iomem *bram_bar;
	uint64_t bram_bar_len;

	uint32_t controller_count;
	uint64_t queue_used; /* bitmap to remember which queues are free */
	pid_t queue_owner[MAX_QUEUES]; /* which process owns this queue */

	/* AIE memory can be accessed indirectly through sysfs.
		It is a two-step protocol:
		(1) write the memory address to:
		/sys/class/amdair/amdair/<id>/address
		(2) Read (or write) the value from:
		/sys/class/amdair/amdair/<id>/value
	*/
	uint64_t mem_addr; /* address for indirect memory access */
};

struct vck5000_device *get_device_by_id(uint32_t device_id);
uint32_t get_controller_count(struct vck5000_device *dev);
uint64_t get_controller_base_address(struct vck5000_device *dev,
				     uint32_t ctrlr_idx);
uint32_t find_free_controller(struct vck5000_device *dev);
void mark_controller_busy(struct vck5000_device *dev, uint32_t ctrlr_idx,
			  pid_t pid);

#endif /* DEVICE_H_ */

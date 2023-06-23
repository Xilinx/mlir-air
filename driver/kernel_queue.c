// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stddef.h>
#include <linux/device.h>
#include <linux/export.h>
#include <linux/err.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/compat.h>
#include <linux/time.h>
#include <linux/mm.h>
#include <linux/mman.h>
#include <linux/ptrace.h>
#include <linux/dma-buf.h>
#include <linux/fdtable.h>
#include <linux/processor.h>
#include <linux/pci.h>

#include "chardev.h"
#include "kernel_queue.h"
#include "hsa_defs.h"

/*  
  Get the address to queue 0 and point the admin queue of the device to it.
*/
hsa_status_t air_queue_create(uint32_t size, uint32_t type,
			      struct vck5000_device *dev)
{
	// Get the base address of the queue, subtract the physical offset,
	// and point our admin queue pointer to it
	dev->admin_queue = dev->bram_bar +
			   HERD_CONTROLLER_BASE_ADDR(dev->bram_bar, 0) -
			   BRAM_PADDR;

	uint32_t queue_id = ioread32(dev->admin_queue + QUEUE_ID_OFFSET);
	if (queue_id != 0xacdc) {
		printk("[WARNING %s] error invalid id %lx\n", __func__,
		       queue_id);
		return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
	}

	uint32_t queue_size = ioread32(dev->admin_queue + QUEUE_SIZE_OFFSET);
	if (queue_size != size) {
		printk("[WARNING %s] error size mismatch %d\n", __func__,
		       queue_size);
		return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
	}

	uint32_t queue_type = ioread32(dev->admin_queue + QUEUE_TYPE_OFFSET);
	if (queue_type != type) {
		printk("[WARNING %s] error type mismatch %d\n", __func__,
		       queue_type);
		return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
	}

	return HSA_STATUS_SUCCESS;
}

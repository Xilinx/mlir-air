// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef VCK5000_CHARDEV_H_
#define VCK5000_CHARDEV_H_

/* The indices in config space (64-bit BARs) */
#define DRAM_BAR_INDEX 0
#define AIE_BAR_INDEX 2
#define BRAM_BAR_INDEX 4

#define MMAP_OFFSET_AIE 0x100000ULL
#define MMAP_OFFSET_BRAM 0x8000000ULL

/*
	This represents a single VCK5000 card.

	This does not use the Linux kernel 'device' infrastructure because we want
	to have only a single character device interface (/dev/amdair) regardless of
	how many physical devices are attached. This follows the AMDKFD driver
	design.
*/
struct vck5000_device {
	struct kobject kobj_aie;

	void __iomem *dram_bar;
	uint64_t dram_bar_len;

	void __iomem *aie_bar;
	uint64_t aie_bar_len;

	void __iomem *bram_bar;
	uint64_t bram_bar_len;

	uint64_t total_controllers;

	/* AIE memory can be accessed indirectly through sysfs.
		It is a two-step protocol:
		(1) write the memory address to:
		/sys/class/amdair/amdair/<id>/address
		(2) Read (or write) the value from:
		/sys/class/amdair/amdair/<id>/value
	*/
	uint64_t mem_addr; /* address for indirect memory access */

  /* Pointer to the admin queue which the kernel can submit 
    requests to.
  */
	void __iomem *admin_queue;
};

int vck5000_chardev_init(struct pci_dev *pdev);
void vck5000_chardev_exit(void);
int create_aie_mem_sysfs(struct vck5000_device *priv, uint32_t index);

const char *amdair_dev_name(void);

#endif /* VCK5000_CHARDEV_H_ */

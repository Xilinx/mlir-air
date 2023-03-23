/*
 * Copyright 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

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

	/* device memory can be accessed indirectly through sysfs.
		It is a two-step protocol:
		(1) write the memory address to:
		/sys/class/amdair/<id>/aie/address
		(2) Read (or write) the value from:
		/sys/class/amdair/<id>/aie/value
	*/
	uint64_t mem_addr; /* address for indirect memory access */
};

int vck5000_chardev_init(struct pci_dev *pdev);
void vck5000_chardev_exit(void);
int create_aie_mem_sysfs(struct vck5000_device *priv, uint32_t index);

const char *amdair_dev_name(void);

#endif /* VCK5000_CHARDEV_H_ */

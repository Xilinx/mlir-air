// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: GPL-2.0

#ifndef VCK5000_CHARDEV_H_
#define VCK5000_CHARDEV_H_

#include "device.h"

/* The indices in config space (64-bit BARs) */
#define DRAM_BAR_INDEX 0
#define AIE_BAR_INDEX 2
#define BRAM_BAR_INDEX 4

int vck5000_chardev_init(struct pci_dev *pdev);
void vck5000_chardev_exit(void);
int create_aie_mem_sysfs(struct vck5000_device *priv, uint32_t index);

const char *amdair_dev_name(void);

#endif /* VCK5000_CHARDEV_H_ */

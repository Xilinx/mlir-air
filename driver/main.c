// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: GPL-2.0

#include <linux/init.h>
#include <linux/module.h>
#include <linux/pci.h>
#include "chardev.h"
#include "device.h"

/* Physical address of the BRAM */
#define BRAM_PA 0x20100000000ULL

/* Offsets into BRAM BAR */
#define HERD_CONTROLLER_BASE_ADDR(_base, _x)                                   \
	((((uint64_t)ioread32(_base + (_x * sizeof(uint64_t) + 4))) << 32) |   \
	 ioread32(_base + (_x * sizeof(uint64_t) + 0)))
#define REG_HERD_CONTROLLER_COUNT 0x208

struct vck5000_device_list {
	struct vck5000_device *next;
	struct list_head list;
};

static const char air_dev_name[] = "amdair";
static int dev_idx; /* linear index of managed devices */
static struct vck5000_device_list device_list;
bool enable_aie;

static int vck5000_probe(struct pci_dev *pdev, const struct pci_device_id *ent);
static void vck5000_remove(struct pci_dev *pdev);

static struct pci_device_id vck5000_id_table[] = { { PCI_DEVICE(0x10EE,
								0xB034) },
						   {
							   0,
						   } };

MODULE_DEVICE_TABLE(pci, vck5000_id_table);

/* Driver registration structure */
static struct pci_driver vck5000 = { .name = air_dev_name,
				     .id_table = vck5000_id_table,
				     .probe = vck5000_probe,
				     .remove = vck5000_remove };

struct vck5000_device *get_device_by_id(uint32_t device_id)
{
	struct vck5000_device *dev;
	list_for_each_entry (dev, &device_list.list, list) {
		if (dev->device_id == device_id)
			return dev;
	}

	return NULL;
}

uint32_t get_controller_count(struct vck5000_device *dev)
{
	return dev->controller_count;
}

/*
	Read the address that the controller is polling on
	Subtract the base address of the physical memory to get an offset that is
	valid over PCIe. The offset can be passed to mmap()
*/
uint64_t get_controller_base_address(struct vck5000_device *dev,
				     uint32_t ctrlr_idx)
{
	return HERD_CONTROLLER_BASE_ADDR(dev->bram_bar, ctrlr_idx) - BRAM_PA;
}

/*
	Find a controller belonging to the specified device that does not have its
	queue allocated yet. Return the total number of controllers if there are
	none free.
*/
uint32_t find_free_controller(struct vck5000_device *dev)
{
	uint32_t idx;

	for (idx = 0; idx < dev->controller_count; idx++) {
		if (!(dev->queue_used & (1ULL << idx))) {
			printk("Controller %u has a free queue\n", idx);
			return idx;
		}
	}

	return dev->controller_count;
}

void mark_controller_busy(struct vck5000_device *dev, uint32_t ctrlr_idx,
			  pid_t pid)
{
	if (dev->queue_used & (1ULL << ctrlr_idx)) {
		printk("Controller %u is already busy!\n", ctrlr_idx);
	}

	dev->queue_used |= (1ULL << ctrlr_idx);
	dev->queue_owner[ctrlr_idx] = pid;
}

/*
	Register the driver with the PCI subsystem
*/
static int __init vck5000_init(void)
{
	if (enable_aie)
		printk("%s: AIE bar access enabled\n", air_dev_name);

	INIT_LIST_HEAD(&device_list.list);

	return pci_register_driver(&vck5000);
}

static void __exit vck5000_exit(void)
{
	/* Unregister */
	pci_unregister_driver(&vck5000);
}

static int vck5000_probe(struct pci_dev *pdev, const struct pci_device_id *ent)
{
	int bar_mask;
	int err;
	struct vck5000_device *dev_priv;
	uint32_t controller_count;
	uint64_t controller_base;
	uint32_t idx;

	/* Enable device memory */
	err = pcim_enable_device(pdev);
	if (err) {
		dev_err(&pdev->dev, "Can't enable device memory");
		return err;
	}

	/* Allocate memory for the device private data */
	dev_priv = kzalloc(sizeof(struct vck5000_device), GFP_KERNEL);
	if (!dev_priv) {
		dev_err(&pdev->dev, "Error allocating private data");
		pci_disable_device(pdev);
		return -ENOMEM;
	}
	dev_priv->mem_addr = 0xbadbeef;
	dev_priv->queue_used = 0;

	/* Find all memory BARs. We are expecting 3 64-bit BARs */
	bar_mask = pci_select_bars(pdev, IORESOURCE_MEM);
	if (bar_mask != 0x15) {
		dev_err(&pdev->dev,
			"These are not the bars we're looking for: 0x%x",
			bar_mask);
		pci_disable_device(pdev);
		return -ENOMEM;
	}

	err = pcim_iomap_regions_request_all(pdev, bar_mask, air_dev_name);
	if (err) {
		dev_err(&pdev->dev, "Can't get memory region for bars");
		// pci_disable_device(pdev);
		return err;
	}
	dev_priv->dram_bar = pcim_iomap_table(pdev)[DRAM_BAR_INDEX];
	dev_priv->aie_bar = pcim_iomap_table(pdev)[AIE_BAR_INDEX];
	dev_priv->bram_bar = pcim_iomap_table(pdev)[BRAM_BAR_INDEX];
	dev_priv->dram_bar_len = pci_resource_len(pdev, DRAM_BAR_INDEX);
	dev_priv->aie_bar_len = pci_resource_len(pdev, AIE_BAR_INDEX);
	dev_priv->bram_bar_len = pci_resource_len(pdev, BRAM_BAR_INDEX);
	dev_warn(&pdev->dev, "bar 0: 0x%lx (0x%llx)",
		 (unsigned long)dev_priv->dram_bar, dev_priv->dram_bar_len);
	dev_warn(&pdev->dev, "bar 2: 0x%lx (0x%llx)",
		 (unsigned long)dev_priv->aie_bar, dev_priv->aie_bar_len);
	dev_warn(&pdev->dev, "bar 4: 0x%lx (0x%llx)",
		 (unsigned long)dev_priv->bram_bar, dev_priv->bram_bar_len);

	/* Set driver private data */
	pci_set_drvdata(pdev, dev_priv);

	/* Request interrupt and set up handler */

	/* set up chardev interface */
	err = vck5000_chardev_init(pdev);
	if (err) {
		dev_err(&pdev->dev, "Error creating char device");
		return -EINVAL;
	}

	/* Query number of herd controllers */
	controller_count =
		ioread32(dev_priv->bram_bar + REG_HERD_CONTROLLER_COUNT);
	if (controller_count > MAX_HERD_CONTROLLERS) {
		dev_err(&pdev->dev,
			"Number of controllers: %u exceeds maximum expected %u",
			controller_count, MAX_HERD_CONTROLLERS);
		return -EINVAL;
	}

	dev_priv->controller_count = controller_count;

	/* Each herd controller has a private memory region */
	for (idx = 0; idx < controller_count; idx++) {
		controller_base =
			HERD_CONTROLLER_BASE_ADDR(dev_priv->bram_bar, idx);
		dev_warn(&pdev->dev, "Controller %u base address: 0x%llx", idx,
			 controller_base);
	}

	/* take queue 0 for exclusive use by driver */
	mark_controller_busy(dev_priv, 0, 0);

	dev_priv->device_id = dev_idx;
	list_add_tail(&dev_priv->list, &device_list.list);

	/* Create sysfs files for accessing AIE memory region */
	create_aie_mem_sysfs(dev_priv, dev_idx);
	dev_idx++;

	return 0;
}

/* Clean up */
static void vck5000_remove(struct pci_dev *pdev)
{
	struct vck5000_device *dev_priv = pci_get_drvdata(pdev);

	vck5000_chardev_exit();

	if (dev_priv) {
		kobject_put(&dev_priv->kobj_aie);
		kfree(dev_priv);
	}

	dev_warn(&pdev->dev, "removed");
}

const char *amdair_dev_name(void)
{
	return air_dev_name;
}

module_init(vck5000_init);
module_exit(vck5000_exit);

module_param(enable_aie, bool, 0644);
MODULE_PARM_DESC(enable_aie, "Enable debug access to AIE BAR");

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Joel Nider <joel.nider@amd.com>");
MODULE_DESCRIPTION("VCK5000 AIR driver");
MODULE_VERSION("1.0");

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

#include <linux/init.h>
#include <linux/module.h>
#include <linux/pci.h>
#include "chardev.h"

static const char air_dev_name[] = "amdair";
bool enable_aie;

static struct pci_device_id vck5000_id_table[] = { { PCI_DEVICE(0x10EE,
								0xB034) },
						   {
							   0,
						   } };

MODULE_DEVICE_TABLE(pci, vck5000_id_table);

static int vck5000_probe(struct pci_dev *pdev, const struct pci_device_id *ent);
static void vck5000_remove(struct pci_dev *pdev);

/* Driver registration structure */
static struct pci_driver vck5000 = { .name = air_dev_name,
				     .id_table = vck5000_id_table,
				     .probe = vck5000_probe,
				     .remove = vck5000_remove };

#define MAX_HERD_CONTROLLERS 64

/* Offsets into BRAM BAR */
#define HERD_CONTROLLER_BASE_ADDR(_base, _x)                                   \
	((((uint64_t)ioread32(_base + (_x * sizeof(uint64_t) + 4))) << 32) |   \
	 ioread32(_base + (_x * sizeof(uint64_t) + 0)))
#define REG_HERD_CONTROLLER_COUNT 0x208

/*
	Register the driver with the PCI subsystem
*/
static int __init vck5000_init(void)
{
	if (enable_aie)
		printk("%s: AIE bar access enabled\n", air_dev_name);

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
	struct vck5000_priv *drv_priv;
	uint32_t controller_count;
	uint64_t controller_base;
	uint32_t idx;
	//uint8_t *ptr;

	dev_warn(&pdev->dev, "vendor: 0x%X device: 0x%X\n", pdev->vendor,
		 pdev->device);

	/* Enable device memory */
	err = pcim_enable_device(pdev);
	if (err) {
		dev_err(&pdev->dev, "Can't enable device memory");
		return err;
	}

	/* Allocate memory for the driver private data */
	drv_priv = kzalloc(sizeof(struct vck5000_priv), GFP_KERNEL);
	if (!drv_priv) {
		dev_err(&pdev->dev, "Error allocating private data");
		pci_disable_device(pdev);
		return -ENOMEM;
	}

	/* Find all memory BARs. We are expecting 3 64-bit BARs */
	bar_mask = pci_select_bars(pdev, IORESOURCE_MEM);
	dev_warn(&pdev->dev, "bar mask: 0x%X", bar_mask);
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
	drv_priv->dram_bar = pcim_iomap_table(pdev)[DRAM_BAR_INDEX];
	drv_priv->aie_bar = pcim_iomap_table(pdev)[AIE_BAR_INDEX];
	drv_priv->bram_bar = pcim_iomap_table(pdev)[BRAM_BAR_INDEX];
	dev_warn(&pdev->dev, "bar 0: 0x%lx", (unsigned long)drv_priv->dram_bar);
	dev_warn(&pdev->dev, "bar 2: 0x%lx", (unsigned long)drv_priv->aie_bar);
	dev_warn(&pdev->dev, "bar 4: 0x%lx", (unsigned long)drv_priv->bram_bar);

	/* Set driver private data */
	pci_set_drvdata(pdev, drv_priv);

	/* Request interrupt and set up handler */

	/* set up chardev interface */
	vck5000_chardev_init(pdev);

	/* Query number of herd controllers */
	controller_count =
		ioread32(drv_priv->bram_bar + REG_HERD_CONTROLLER_COUNT);
	dev_warn(&pdev->dev, "Total controllers: 0x%x", controller_count);
	drv_priv->total_controllers = controller_count;

	/* Each herd controller has a private memory region */
	for (idx = 0; idx < controller_count; idx++) {
		controller_base =
			HERD_CONTROLLER_BASE_ADDR(drv_priv->bram_bar, idx);
		dev_warn(&pdev->dev, "Controller %u base address: 0x%llx", idx,
			 controller_base);
	}

	return 0;
}

/* Clean up */
static void vck5000_remove(struct pci_dev *pdev)
{
	struct vck5000_priv *drv_priv = pci_get_drvdata(pdev);

	vck5000_chardev_exit();

	if (drv_priv) {
		kfree(drv_priv);
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

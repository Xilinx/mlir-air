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
#include <uapi/linux/kfd_ioctl.h>
#include "chardev.h"

#define VCK5000_IOCTL_DEF(ioctl, _func, _flags)                                \
	[_IOC_NR(ioctl)] = { .cmd = ioctl,                                     \
			     .func = _func,                                    \
			     .flags = _flags,                                  \
			     .cmd_drv = 0,                                     \
			     .name = #ioctl }

#define HERD_CONTROLLER_BRAM_SIZE 0x1000

typedef int vck5000_ioctl_t(struct file *filep, void *data);

struct vck5000_ioctl_desc {
	unsigned int cmd;
	int flags;
	vck5000_ioctl_t *func;
	unsigned int cmd_drv;
	const char *name;
};

static long vck_ioctl(struct file *, unsigned int, unsigned long);
static int vck_open(struct inode *, struct file *);
static int vck_release(struct inode *, struct file *);
static int vck_mmap(struct file *, struct vm_area_struct *);

static int vck5000_ioctl_get_version(struct file *filep, void *data);
static int vck5000_ioctl_create_queue(struct file *filep, void *data);
static int vck5000_ioctl_destroy_queue(struct file *filp, void *data);

static const struct file_operations vck_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = vck_ioctl,
	.compat_ioctl = compat_ptr_ioctl,
	.open = vck_open,
	.release = vck_release,
	.mmap = vck_mmap,
};

extern bool enable_aie;
static int chardev_major = -1;
static struct class *vck5000_class;
static struct device *vck5000_device;

/** Ioctl table */
static const struct vck5000_ioctl_desc vck5000_ioctls[] = {
	VCK5000_IOCTL_DEF(AMDKFD_IOC_GET_VERSION, vck5000_ioctl_get_version, 0),

	VCK5000_IOCTL_DEF(AMDKFD_IOC_CREATE_QUEUE, vck5000_ioctl_create_queue,
			  0),

	VCK5000_IOCTL_DEF(AMDKFD_IOC_DESTROY_QUEUE, vck5000_ioctl_destroy_queue,
			  0),
};

int vck5000_chardev_init(struct pci_dev *pdev)
{
	int ret = 0;

	ret = register_chrdev(0, amdair_dev_name(), &vck_fops);
	if (ret < 0)
		goto err_register;

	chardev_major = ret;

	vck5000_class = class_create(THIS_MODULE, amdair_dev_name());
	ret = PTR_ERR(vck5000_class);
	if (IS_ERR(vck5000_class))
		goto err_class;

	vck5000_device =
		device_create(vck5000_class, NULL, MKDEV(chardev_major, 0),
			      NULL, amdair_dev_name());

	ret = PTR_ERR(vck5000_device);
	if (IS_ERR(vck5000_device))
		goto err_device;

	dev_set_drvdata(vck5000_device, pdev);

	return 0;

err_device:
	class_destroy(vck5000_class);

err_class:
	unregister_chrdev(chardev_major, amdair_dev_name());

err_register:
	return ret;
}

void vck5000_chardev_exit(void)
{
	device_destroy(vck5000_class, MKDEV(chardev_major, 0));
	class_destroy(vck5000_class);
	unregister_chrdev(chardev_major, amdair_dev_name());

	vck5000_device = NULL;
}

static long vck_ioctl(struct file *filep, unsigned int cmd, unsigned long arg)
{
	uint32_t amdkfd_size;
	uint32_t usize, asize;
	char stack_kdata[128];
	char *kdata = NULL;
	vck5000_ioctl_t *func;
	const struct vck5000_ioctl_desc *ioctl = NULL;
	unsigned int nr = _IOC_NR(cmd);
	int ret;

	dev_warn(vck5000_device, "%s", __func__);

	if ((nr < AMDKFD_COMMAND_START) || (nr >= AMDKFD_COMMAND_END)) {
		dev_warn(vck5000_device, "%s invalid %u", __func__, nr);
		return 0;
	}

	ioctl = &vck5000_ioctls[nr];

	amdkfd_size = _IOC_SIZE(ioctl->cmd);
	usize = asize = _IOC_SIZE(cmd);
	if (amdkfd_size > asize)
		asize = amdkfd_size;

	cmd = ioctl->cmd;
	func = ioctl->func;
	if (cmd & (IOC_IN | IOC_OUT)) {
		if (asize <= sizeof(stack_kdata)) {
			kdata = stack_kdata;
		} else {
			kdata = kmalloc(asize, GFP_KERNEL);
			if (!kdata) {
				return -ENOMEM;
			}
		}
		if (asize > usize)
			memset(kdata + usize, 0, asize - usize);
	}

	if (cmd & IOC_IN) {
		if (copy_from_user(kdata, (void __user *)arg, usize) != 0) {
			if (kdata != stack_kdata)
				kfree(kdata);
			return -EFAULT;
		}
	} else if (cmd & IOC_OUT) {
		memset(kdata, 0, usize);
	}
	ret = func(filep, kdata);

	/* copy any results back to userspace */
	if (cmd & IOC_OUT)
		if (copy_to_user((void __user *)arg, kdata, usize) != 0)
			ret = -EFAULT;

	if (kdata != stack_kdata)
		kfree(kdata);
	return ret;
}

static int vck_open(struct inode *node, struct file *f)
{
	dev_warn(vck5000_device, "%s", __func__);
	return 0;
}

static int vck_release(struct inode *node, struct file *f)
{
	dev_warn(vck5000_device, "%s", __func__);
	return 0;
}

static int vck_mmap(struct file *f, struct vm_area_struct *vma)
{
	struct pci_dev *pdev;
	unsigned long pgoff = 0;
	unsigned long bar;
	size_t size = vma->vm_end - vma->vm_start;
	loff_t offset = (loff_t)vma->vm_pgoff << PAGE_SHIFT;

	pdev = dev_get_drvdata(vck5000_device);

	dev_warn(vck5000_device, "%s start %lx end %lx offset %llx", __func__,
		 vma->vm_start, vma->vm_end, offset);

	if (offset >= 0x8000000ULL) {
		unsigned int herd_idx = 0;
		/* the herd private memory starts at 0x1000, and is 0x1000 long for each herd controller */
		unsigned long herd_ctlr_offset =
			HERD_CONTROLLER_BRAM_SIZE * (1 + herd_idx);
		bar = pci_resource_start(pdev, BRAM_BAR_INDEX) +
		      herd_ctlr_offset;
		size = HERD_CONTROLLER_BRAM_SIZE;
		dev_warn(vck5000_device, "mapping %lx BRAM at 0x%lx to 0x%lx",
			 size, bar, vma->vm_start);
	} else if (offset == 0x100000ULL) {
		if (!enable_aie) {
			dev_warn(vck5000_device,
				 "mapping AIE BAR is not enabled");
			return -EOPNOTSUPP;
		}
		bar = pci_resource_start(pdev, AIE_BAR_INDEX);
		dev_warn(vck5000_device, "mapping %lx AIE at 0x%lx to 0x%lx",
			 size, bar, vma->vm_start);
	} else {
		bar = pci_resource_start(pdev, DRAM_BAR_INDEX) + offset;
		dev_warn(vck5000_device, "mapping %lx DRAM at 0x%lx to 0x%lx",
			 size, bar, vma->vm_start);
	}
	pgoff = (bar >> PAGE_SHIFT);

	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

	/* Remap-pfn-range will mark the range VM_IO */
	if (remap_pfn_range(vma, vma->vm_start, pgoff, size,
			    vma->vm_page_prot)) {
		return -EAGAIN;
	}

	return 0;
}

static int vck5000_ioctl_get_version(struct file *filep, void *data)
{
	struct kfd_ioctl_get_version_args *args = data;

	dev_warn(vck5000_device, "%s %u.%u", __func__, KFD_IOCTL_MAJOR_VERSION,
		 KFD_IOCTL_MINOR_VERSION);
	args->major_version = KFD_IOCTL_MAJOR_VERSION;
	args->minor_version = KFD_IOCTL_MINOR_VERSION;

	return 0;
}

/*
	This shouldn't be used (yet) - use mmap instead
*/
static int vck5000_ioctl_create_queue(struct file *filep, void *data)
{
	dev_warn(vck5000_device, "%s", __func__);

	return 0;
}

/*
	This shouldn't be used either
*/
static int vck5000_ioctl_destroy_queue(struct file *filp, void *data)
{
	int retval = 0;
	dev_warn(vck5000_device, "%s", __func__);

	return retval;
}

// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: GPL-2.0

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
#include "amdair_ioctl.h"

/*
	Define an entry in the ioctl table
	The entry's index equals the ioctl number
*/
#define AMDAIR_IOCTL_DEF(ioctl, _func, _flags)                                 \
	[_IOC_NR(ioctl)] = { .cmd = ioctl,                                     \
			     .func = _func,                                    \
			     .flags = _flags,                                  \
			     .cmd_drv = 0,                                     \
			     .name = #ioctl }

#define HERD_CONTROLLER_BRAM_SIZE 0x1000

enum aie_address_validation {
	AIE_ADDR_OK,
	AIE_ADDR_ALIGNMENT,
	AIE_ADDR_RANGE,
};

typedef int vck5000_ioctl_t(struct file *filep, void *data);

struct vck5000_ioctl_desc {
	unsigned int cmd;
	int flags;
	vck5000_ioctl_t *func;
	unsigned int cmd_drv;
	const char *name;
};

struct amdair_attribute {
	struct attribute attr;
	ssize_t (*show)(struct kobject *kobj, struct attribute *attr,
			char *buf);
	ssize_t (*store)(struct kobject *kobj, struct attribute *attr,
			 const char *buf, size_t count);
};

/* Some forward declarations */
static ssize_t aie_show(struct kobject *kobj, struct attribute *attr,
			char *buf);
static ssize_t aie_store(struct kobject *kobj, struct attribute *attr,
			 const char *buf, size_t count);
static ssize_t address_show(struct kobject *kobj, struct attribute *attr,
			    char *buf);
static ssize_t address_store(struct kobject *kobj, struct attribute *attr,
			     const char *buf, size_t count);
static ssize_t value_show(struct kobject *kobj, struct attribute *attr,
			  char *buf);
static ssize_t value_store(struct kobject *kobj, struct attribute *attr,
			   const char *buf, size_t count);

static long vck_ioctl(struct file *, unsigned int, unsigned long);
static int vck_open(struct inode *, struct file *);
static int vck_release(struct inode *, struct file *);
static int vck_mmap(struct file *, struct vm_area_struct *);

static int amdair_ioctl_get_version(struct file *filep, void *data);
static int amdair_ioctl_create_queue(struct file *filep, void *data);
static int amdair_ioctl_destroy_queue(struct file *filp, void *data);

/* define sysfs attributes */
struct amdair_attribute aie_attr_address = __ATTR_RW(address);
struct amdair_attribute aie_attr_value = __ATTR_RW(value);

static const struct sysfs_ops aie_sysfs_ops = {
	.show = aie_show,
	.store = aie_store,
};

static struct kobj_type aie_sysfs_type = {
	.sysfs_ops = &aie_sysfs_ops,
};

struct attribute *aie_sysfs_attrs[] = { &aie_attr_address.attr,
					&aie_attr_value.attr, NULL };

ATTRIBUTE_GROUPS(aie_sysfs);

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
static struct device *vck5000_chardev;

static const struct vck5000_ioctl_desc amdair_ioctl_table[] = {
	AMDAIR_IOCTL_DEF(AMDAIR_IOC_GET_VERSION, amdair_ioctl_get_version, 0),

	AMDAIR_IOCTL_DEF(AMDAIR_IOC_CREATE_QUEUE, amdair_ioctl_create_queue, 0),

	AMDAIR_IOCTL_DEF(AMDAIR_IOC_DESTROY_QUEUE, amdair_ioctl_destroy_queue,
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

	vck5000_chardev =
		device_create(vck5000_class, &pdev->dev,
			      MKDEV(chardev_major, 0), pdev, "amdair");

	ret = PTR_ERR(vck5000_chardev);
	if (IS_ERR(vck5000_chardev))
		goto err_device;

	return 0;

err_device:
	class_destroy(vck5000_class);

err_class:
	unregister_chrdev(chardev_major, amdair_dev_name());

err_register:
	return -ENODEV;
}

void vck5000_chardev_exit(void)
{
	device_destroy(vck5000_class, MKDEV(chardev_major, 0));
	class_destroy(vck5000_class);
	unregister_chrdev(chardev_major, amdair_dev_name());

	vck5000_chardev = NULL;
}

/*
	Allocate a queue in device memory

	This allocates some BRAM or DRAM and returns the base address to the caller
	Each controller can have only one queue at the moment. Find a controller
	that is free, and return its statically allocated address.
*/
static int alloc_device_queue(uint32_t device_id, pid_t pid, uint32_t *queue_id,
			      uint64_t *base_address)
{
	uint32_t ctrlr_idx;
	struct vck5000_device *dev = get_device_by_id(device_id);

	if (!dev) {
		printk("Can't find device id %u\n", device_id);
		return -EINVAL;
	}

	ctrlr_idx = find_free_controller(dev);
	if (ctrlr_idx == get_controller_count(dev)) {
		printk("All controllers are busy\n");
		return -EBUSY;
	}

	mark_controller_busy(dev, ctrlr_idx, pid);

	/* queue id is global (i.e. across all controllers) and there is only one
		queue per controller at this time, so queue id == controller id
	*/
	*queue_id = ctrlr_idx;
	*base_address = get_controller_base_address(dev, ctrlr_idx);

	return 0;
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

	dev_warn(vck5000_chardev, "%s %u", __func__, nr);

	if ((nr < AMDAIR_COMMAND_START) || (nr > AMDAIR_COMMAND_END)) {
		dev_warn(vck5000_chardev, "%s invalid %u", __func__, nr);
		return 0;
	}

	ioctl = &amdair_ioctl_table[nr];

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
			dev_warn(vck5000_chardev, "Missing data in ioctl %u",
				 nr);
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
	dev_warn(vck5000_chardev, "%s", __func__);
	return 0;
}

/*
	Called when userspace closes the handle to the driver
	Release all queues and clean up
*/
static int vck_release(struct inode *node, struct file *f)
{
	dev_warn(vck5000_chardev, "%s", __func__);
	return 0;
}

static int vck_mmap(struct file *f, struct vm_area_struct *vma)
{
	struct pci_dev *pdev;
	unsigned long pgoff = 0;
	unsigned long start, end;
	uint64_t devid, range;
	size_t size = vma->vm_end - vma->vm_start;
	loff_t offset = (loff_t)vma->vm_pgoff << PAGE_SHIFT;

	pdev = dev_get_drvdata(vck5000_chardev);

	devid = AMDAIR_MMAP_DECODE_OFFSET_DEVID(offset);
	range = AMDAIR_MMAP_DECODE_OFFSET_RANGE(offset);
	offset = AMDAIR_MMAP_DECODE_OFFSET(offset);

	dev_warn(
		vck5000_chardev,
		"%s start 0x%lx end 0x%lx devid %llu range 0x%llx offset 0x%llx",
		__func__, vma->vm_start, vma->vm_end, devid, range, offset);

	switch (range) {
	case AMDAIR_MMAP_RANGE_AIE:
		if (!enable_aie) {
			dev_warn(vck5000_chardev,
				 "mapping AIE BAR is not enabled");
			return -EOPNOTSUPP;
		}
		start = pci_resource_start(pdev, AIE_BAR_INDEX) + offset;
		end = pci_resource_end(pdev, AIE_BAR_INDEX);
		dev_warn(vck5000_chardev, "mapping 0x%lx AIE at 0x%lx to 0x%lx",
			 size, start, vma->vm_start);
		break;

	case AMDAIR_MMAP_RANGE_DRAM:
		start = pci_resource_start(pdev, DRAM_BAR_INDEX) + offset;
		end = pci_resource_end(pdev, DRAM_BAR_INDEX);
		dev_warn(vck5000_chardev,
			 "mapping 0x%lx DRAM at 0x%lx to 0x%lx", size, start,
			 vma->vm_start);
		break;

	case AMDAIR_MMAP_RANGE_BRAM:
		start = pci_resource_start(pdev, BRAM_BAR_INDEX) + offset;
		end = pci_resource_end(pdev, BRAM_BAR_INDEX);
		size = HERD_CONTROLLER_BRAM_SIZE;
		dev_warn(vck5000_chardev,
			 "mapping 0x%lx BRAM at 0x%lx to 0x%lx", size, start,
			 vma->vm_start);
		break;

	default:
		dev_warn(vck5000_chardev, "Unrecognized mmap range 0x%llx",
			 range);
		return -EOPNOTSUPP;
	}

	if ((start + size) >= end) {
		dev_err(vck5000_chardev,
			"size 0x%lx starting at 0x%lx exceeds BAR", size,
			start);
		return -EINVAL;
	}

	pgoff = (start >> PAGE_SHIFT);

	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

	/* Remap-pfn-range will mark the range VM_IO */
	if (remap_pfn_range(vma, vma->vm_start, pgoff, size,
			    vma->vm_page_prot)) {
		return -EAGAIN;
	}

	return 0;
}

static int amdair_ioctl_get_version(struct file *filep, void *data)
{
	struct amdair_get_version_args *args = data;

	dev_warn(vck5000_chardev, "%s %u.%u", __func__,
		 AMDAIR_IOCTL_MAJOR_VERSION, AMDAIR_IOCTL_MINOR_VERSION);
	args->major_version = AMDAIR_IOCTL_MAJOR_VERSION;
	args->minor_version = AMDAIR_IOCTL_MINOR_VERSION;

	return 0;
}

/*
	Allocate a queue of a specified device to the pid of the caller
	It will create a new queue and map the device address range into
	the address space of the caller using mmap.
*/
static int amdair_ioctl_create_queue(struct file *filep, void *data)
{
	int ret;
	uint32_t queue_id;
	uint64_t offset, base;
	struct amdair_create_queue_args *args =
		(struct amdair_create_queue_args *)data;
	uint64_t size = HERD_CONTROLLER_BRAM_SIZE;

	dev_warn(vck5000_chardev, "%s from pid %u", __func__, current->pid);

	switch (args->queue_type) {
	case AMDAIR_QUEUE_HOST:
		dev_warn(vck5000_chardev, "Host queues not supported yet!");
		break;

	case AMDAIR_QUEUE_DEVICE:
		ret = alloc_device_queue(args->device_id, current->pid,
					 &queue_id, &offset);
		if (ret == 0) {
			/* encode offset to pass device ID and memory type */
			offset = AMDAIR_MMAP_ENCODE_OFFSET(BRAM, offset,
							   args->device_id);
			base = vm_mmap(filep, 0, size, PROT_READ | PROT_WRITE,
				       MAP_SHARED, offset);
			args->write_pointer_address = 0xbad0;
			args->read_pointer_address = 0xbad1;
			args->doorbell_offset = 0xbad2;
			args->ring_base_address = base;
			args->queue_id = queue_id;
		}
		break;

	case AMDAIR_QUEUE_3RD_PARTY:
		dev_warn(vck5000_chardev,
			 "3rd party queues not supported yet!");
		break;
	}

	return 0;
}

/*
	This shouldn't be used either
*/
static int amdair_ioctl_destroy_queue(struct file *filp, void *data)
{
	int retval = 0;
	dev_warn(vck5000_chardev, "%s from pid %u", __func__, current->pid);

	return retval;
}

static int validate_aie_address(uint64_t offset, struct vck5000_device *dev)
{
	/* alignment */
	if (offset & 0x3) {
		printk("%s: 0x%llx not aligned to 4 bytes\n", __func__, offset);
		return AIE_ADDR_ALIGNMENT;
	}

	/* range within the specified BAR */
	if (offset >= dev->aie_bar_len) {
		printk("%s: invalid offset 0x%llx (max 0x%llx)\n", __func__,
		       offset, dev->aie_bar_len);
		return AIE_ADDR_RANGE;
	}

	return AIE_ADDR_OK;
}

static ssize_t address_show(struct kobject *kobj, struct attribute *attr,
			    char *buf)
{
	struct vck5000_device *drv_priv =
		container_of(kobj, struct vck5000_device, kobj_aie);

	snprintf(buf, PAGE_SIZE, "0x%llx\n", drv_priv->mem_addr);
	return strlen(buf) + 1;
}

static ssize_t address_store(struct kobject *kobj, struct attribute *attr,
			     const char *buf, size_t count)
{
	unsigned long address;
	struct vck5000_device *drv_priv =
		container_of(kobj, struct vck5000_device, kobj_aie);

	kstrtoul(buf, 0, &address);
	drv_priv->mem_addr = address;
	return count;
}

static ssize_t value_show(struct kobject *kobj, struct attribute *attr,
			  char *buf)
{
	uint32_t value;
	struct vck5000_device *drv_priv =
		container_of(kobj, struct vck5000_device, kobj_aie);
	uint64_t offset = drv_priv->mem_addr;

	if (validate_aie_address(offset, drv_priv)) {
		snprintf(buf, PAGE_SIZE, "0xffffffff\n");
		return strlen(buf) + 1;
	}

	value = ioread32(drv_priv->aie_bar + offset);

	snprintf(buf, PAGE_SIZE, "0x%x\n", value);
	return strlen(buf) + 1;
}

static ssize_t value_store(struct kobject *kobj, struct attribute *attr,
			   const char *buf, size_t count)
{
	uint32_t value;
	struct vck5000_device *drv_priv =
		container_of(kobj, struct vck5000_device, kobj_aie);
	uint64_t offset = drv_priv->mem_addr;

	if (validate_aie_address(offset, drv_priv) == AIE_ADDR_OK) {
		kstrtouint(buf, 0, &value);
		iowrite32(value, drv_priv->aie_bar + offset);
	}

	return count;
}

static ssize_t aie_show(struct kobject *kobj, struct attribute *attr, char *buf)
{
	struct amdair_attribute *air_attr =
		container_of(attr, struct amdair_attribute, attr);

	if (!air_attr->show) {
		printk("Missing show method for %s\r\n", attr->name);
		return 0;
	}

	return air_attr->show(kobj, attr, buf);
}

static ssize_t aie_store(struct kobject *kobj, struct attribute *attr,
			 const char *buf, size_t count)
{
	struct amdair_attribute *air_attr =
		container_of(attr, struct amdair_attribute, attr);

	if (!air_attr->store) {
		printk("Missing store method for %s\r\n", attr->name);
		return 0;
	}

	return air_attr->store(kobj, attr, buf, count);
}

int create_aie_mem_sysfs(struct vck5000_device *priv, uint32_t index)
{
	int err;

	err = kobject_init_and_add(&priv->kobj_aie, &aie_sysfs_type,
				   &vck5000_chardev->kobj, "%02u", index);
	if (err) {
		dev_err(vck5000_chardev, "Error creating sysfs device");
		kobject_put(&priv->kobj_aie);
		return -1;
	}

	sysfs_create_groups(&priv->kobj_aie, aie_sysfs_groups);

	return 0;
}

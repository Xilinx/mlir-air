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
#include <uapi/linux/kfd_ioctl.h>
#include "chardev.h"

#define DEVICE_INDEX_STR_MAX 15

#define VCK5000_IOCTL_DEF(ioctl, _func, _flags)                                \
	[_IOC_NR(ioctl)] = { .cmd = ioctl,                                     \
			     .func = _func,                                    \
			     .flags = _flags,                                  \
			     .cmd_drv = 0,                                     \
			     .name = #ioctl }

#define HERD_CONTROLLER_BRAM_SIZE 0x1000
#define DOORBELL_OFFSET 0x50
#define SIZE_OFFSET 0x58
#define ID_OFFSET 0x60
#define RD_PTR_OFFSET 0x68
#define WR_PTR_OFFSET 0x70
#define QUEUE_OFFSET 0xC0
#define QUEUE_TIMEOUT_VAL 5000000

#define USE_RW32_PKTS

  /*

typedef struct dispatch_packet_s {

  // HSA-like interface
  volatile uint16_t header;
  volatile uint16_t type;
  uint32_t reserved0;
  uint64_t return_address;
  uint64_t arg[4];
  uint64_t reserved1;
  uint64_t completion_signal;

} __attribute__((packed, aligned(__alignof__(uint64_t)))) dispatch_packet_t;
*/

#define PKT_SIZE                0x40
#define PKT_HEADER_TYPE_OFFSET  0x00
#define PKT_RET_ADDR_OFFSET     0x08
#define PKT_ARG_ADDR_OFFSET     0x10
#define PKT_COMPL_OFFSET        0x38
#define NUM_PKTS                0x30

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

static int vck5000_ioctl_get_version(struct file *filep, void *data);
static int vck5000_ioctl_create_queue(struct file *filep, void *data);
static int vck5000_ioctl_destroy_queue(struct file *filp, void *data);

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

// Used for Eddie debugging read32/write32
uint32_t show_count = 0;
uint32_t store_count = 0;


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

	if ((nr < AMDKFD_COMMAND_START) || (nr >= AMDKFD_COMMAND_END)) {
		dev_warn(vck5000_chardev, "%s invalid %u", __func__, nr);
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
	dev_warn(vck5000_chardev, "%s", __func__);
	return 0;
}

static int vck_release(struct inode *node, struct file *f)
{
	dev_warn(vck5000_chardev, "%s", __func__);
	return 0;
}

static int vck_mmap(struct file *f, struct vm_area_struct *vma)
{
	struct pci_dev *pdev;
	unsigned long pgoff = 0;
	unsigned long bar;
	size_t size = vma->vm_end - vma->vm_start;
	loff_t offset = (loff_t)vma->vm_pgoff << PAGE_SHIFT;

	pdev = dev_get_drvdata(vck5000_chardev);

	dev_warn(vck5000_chardev, "%s start %lx end %lx offset %llx", __func__,
		 vma->vm_start, vma->vm_end, offset);

	if (offset >= MMAP_OFFSET_BRAM) {
		unsigned int herd_idx = 0;
		/* the herd private memory starts at 0x1000, and is 0x1000 long for each herd controller */
		unsigned long herd_ctlr_offset =
			HERD_CONTROLLER_BRAM_SIZE * (1 + herd_idx);
		bar = pci_resource_start(pdev, BRAM_BAR_INDEX) +
		      herd_ctlr_offset;
		size = HERD_CONTROLLER_BRAM_SIZE;
		dev_warn(vck5000_chardev, "mapping %lx BRAM at 0x%lx to 0x%lx",
			 size, bar, vma->vm_start);
	} else if (offset == MMAP_OFFSET_AIE) {
		if (!enable_aie) {
			dev_warn(vck5000_chardev,
				 "mapping AIE BAR is not enabled");
			return -EOPNOTSUPP;
		}
		bar = pci_resource_start(pdev, AIE_BAR_INDEX);
		dev_warn(vck5000_chardev, "mapping %lx AIE at 0x%lx to 0x%lx",
			 size, bar, vma->vm_start);
	} else {
		bar = pci_resource_start(pdev, DRAM_BAR_INDEX) + offset;
		dev_warn(vck5000_chardev, "mapping %lx DRAM at 0x%lx to 0x%lx",
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

	dev_warn(vck5000_chardev, "%s %u.%u", __func__, KFD_IOCTL_MAJOR_VERSION,
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
	dev_warn(vck5000_chardev, "%s", __func__);

	return 0;
}

/*
	This shouldn't be used either
*/
static int vck5000_ioctl_destroy_queue(struct file *filp, void *data)
{
	int retval = 0;
	dev_warn(vck5000_chardev, "%s", __func__);

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

  show_count++;
  printk("[EDDIE DEBUG %s] show_count: 0x%x\n", __func__, show_count);

#ifdef USE_RW32_PKTS

  // Step 1: Calculate the queue location
  uint32_t queue_id = ioread32(drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + ID_OFFSET);
  if(queue_id != 0xacdc) {
    dev_warn(vck5000_chardev, "%s Invalid queue ID of 0x%x", __func__, queue_id);
		return strlen(buf) + 1;
  }

  // Step 2: Read the write pointer
  uint32_t wr_idx = ioread32(drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + WR_PTR_OFFSET);

  // Step 3: Read the queue size
  uint32_t queue_size = ioread32(drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + SIZE_OFFSET);

  // Step 4: Calculate packet_id
  uint32_t packet_id = wr_idx % queue_size;
  printk("[EDDIE DEBUG %s] We are writing to a packet ID of 0x%x\n", __func__, packet_id);

  // Step 5.1: Write the new write index to the queue
  iowrite32(wr_idx + 1, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + WR_PTR_OFFSET);

  // Step 6.1: Intiialize_packet - Write HSA_PACKET_TYPE_INVALID (1) to header
  iowrite32(0x00000001, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_HEADER_TYPE_OFFSET);
  // Step 6.2: Write arguments
  iowrite32((uint32_t)(offset & 0xFFFFFFFF), drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_ARG_ADDR_OFFSET); // Writing the addr low bits
  iowrite32((uint32_t)(offset >> 32), drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_ARG_ADDR_OFFSET + 0x4);  // Writing the addr high bits
  iowrite32(0x00000000, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_ARG_ADDR_OFFSET + 0x8);                // Writing the value, doesn't matter because read
  iowrite32(0x00000000, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_ARG_ADDR_OFFSET + 0xC);                // Writing the is_write
  // Step 6.3: Write pkt->type and header
  iowrite32(0x00500004, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_HEADER_TYPE_OFFSET); // Writing dispatch header type and rw32 dude
  //printk("[EDDIE DEBUG %s] we are writing to 0x%lx\n", __func__, (uint64_t)(HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_HEADER_TYPE_OFFSET));
  // Step 6.4: Create the completion signal
  iowrite32(0x00000001, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_COMPL_OFFSET); // Writing a 1 to the signal which will mark the completion

  // Step 7: Ring the doorbell
  iowrite32(wr_idx, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + DOORBELL_OFFSET);
  iowrite32(0, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + DOORBELL_OFFSET + 0x4);

  // Step 8: Poll on the read pointer to make sure that we did the dude - Make sure that there is a timeout and if there is return 0
  uint64_t timeout_val = 0;
  uint32_t read_completion = ioread32(drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_COMPL_OFFSET);
  while(read_completion != 0x00000000) {
    if(timeout_val >= QUEUE_TIMEOUT_VAL) {
		  dev_warn(vck5000_chardev, "%s Timed out sending packet to the admin queue", __func__);
		  return strlen(buf) + 1;
    }
    read_completion = ioread32(drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_COMPL_OFFSET);
    timeout_val++;
  }

  // Step 9: Get the value from the packet and return it to the dude
  value = ioread32(drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_RET_ADDR_OFFSET);
#else
  value = ioread32(drv_priv->aie_bar + offset);
#endif

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

    store_count++;
    printk("[EDDIE DEBUG %s] store_count: 0x%x\n", __func__, store_count);

#ifdef USE_RW32_PKTS
    // Step 1: Calculate the queue location
    uint32_t queue_id = ioread32(drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + ID_OFFSET);
    if(queue_id != 0xacdc) {
      dev_warn(vck5000_chardev, "%s Invalid queue ID of 0x%x", __func__, queue_id);
		  return count;
    }

    // Step 2: Read the write pointer
    uint32_t wr_idx = ioread32(drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + WR_PTR_OFFSET);

    // Step 3: Read the queue size
    uint32_t queue_size = ioread32(drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + SIZE_OFFSET);

    // Step 4: Calculate packet_id
    uint32_t packet_id = wr_idx % queue_size;
    printk("[EDDIE DEBUG %s] We are writing to a packet ID of 0x%x\n", __func__, packet_id);

    // Step 5.1: Write the new write index to the queue
    iowrite32(wr_idx + 1, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + WR_PTR_OFFSET);

    // Step 6.1: Intiialize_packet - Write HSA_PACKET_TYPE_INVALID (1) to header
    iowrite32(0x00000001, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_HEADER_TYPE_OFFSET);
    // Step 6.2: Write arguments
    iowrite32((uint32_t)(offset & 0xFFFFFFFF), drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_ARG_ADDR_OFFSET); // Writing the addr low bits
    iowrite32((uint32_t)(offset >> 32), drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_ARG_ADDR_OFFSET + 0x4);  // Writing the addr high bits
    iowrite32(value, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_ARG_ADDR_OFFSET + 0x8);                     // Writing the value
    iowrite32(0x00000001, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_ARG_ADDR_OFFSET + 0xC);                // Writing the is_write
    // Step 6.3: Write pkt->type and header
    iowrite32(0x00500004, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_HEADER_TYPE_OFFSET);                   // Writing dispatch header type and rw32 dude
    // Step 6.4: Create the completion signal
    iowrite32(0x00000001, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_COMPL_OFFSET);                         // Writing a 1 to the signal which will mark the completion

    // Step 7: Ring the doorbell
    iowrite32(wr_idx, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + DOORBELL_OFFSET);
    iowrite32(0, drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + DOORBELL_OFFSET + 0x4);

    // Step 8: Poll on the read pointer to make sure that we did the dude - Make sure that there is a timeout and if there is return 0
    uint64_t timeout_val = 0;
    uint32_t read_completion = ioread32(drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_COMPL_OFFSET);
    while(read_completion != 0x00000000) {
      if(timeout_val >= QUEUE_TIMEOUT_VAL) {
        dev_warn(vck5000_chardev, "%s Timed out sending packet to the admin queue", __func__);
		    return count;
      }
      read_completion = ioread32(drv_priv->bram_bar + HERD_CONTROLLER_BRAM_SIZE + QUEUE_OFFSET + PKT_SIZE * packet_id + PKT_COMPL_OFFSET);
      timeout_val++;
    }
#else
		iowrite32(value, drv_priv->aie_bar + offset);
#endif

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

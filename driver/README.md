# VCK5000 AIR Driver

This is a Linux kernel driver to manage the VCK5000 PCIe card. It exposes a
character device /dev/amdair for interfacing with user space applications.
Applications can use the mmap() system call to access the DRAM and BRAM
memory regions on the card. This is a very raw and low-level interface so it
requires most of the logic concerning how to program the card to be in user
space libraries. Part of the low-level interface is the AIE memory range, to
address the AIE engines directly. This use is deprecated but still widely used
by our test code. This memory range is disabled by default but can be enabled
through a module parameter (see the section on Loading for more details).

It is designed to use the same ioctl interface (function numbers and
parameters) as the KFD driver, used by ROCm primarily for GPUs. Many of these
functions are not yet implemented as this is an early stage of development.

## Compiling

The source code is kept out-of-tree at this time. That means it is not tied
to any Linux upstream project and may not work with every version of the
kernel. It has been designed and tested on the 5.11 kernel but will likely
work without modifications on any 5.x kernel (and possibly more).
There is a Makefile alongside the source code. It is designed to compile the
driver for the kernel currently running in the host system. On success, it
will produce a loadable kernel module called amdair.ko in the source directory.

To compile the driver, you will need the standard build tools for the Linux
kernel. You can install the 'build-essentials' package on your distro to get
these tools if you don't have them already installed.

## Loading and Unloading

The driver will not load automatically since it is build out-of-tree and the
module subsystem is not aware of its existence. To load it, use the 'insmod'
tool with the path to the module. Of course you need the appropriate privilege
level to load a kernel module ('sudo' will do):

```
insmod amdair.ko
```

If you want to pass a module parameter (e.g. to enable the AIE memory region),
you can do it like this:

```
insmod amdair.ko enable_aie=1
```

To unload the module, use the 'rmmod' tool with the module name:
```
rmmod amdair
```

Loading the kernel module will automatically create the device file
/dev/amdair. By default, this device will be restricted to access by root. To
give more general access, use a udev rule. An example rule is found in
99-amdair.rules in this directory. The example rule is set up to give read
and write access to all users. This can be modified to fit your needs by
changing the group or owner and mode appropriately. To install the rule,
copy it to a directory that is searched by udev. On Ubuntu, the common location
is /etc/udev/rules.d/.

## Debugging

When things go wrong in an application, people love to blame the driver. To
see what the driver has done, you can review the kernel message log with the
'dmesg' tool. The driver will not tell you everything that it is doing but
might shed some light on what is happening in the system:
```
dmesg | grep amdair
```

## Known Issues & Limitations

The driver will only bind to the first card it sees. If there are more cards,
they cannot be used by this version of the driver.

Only one queue can be used from the card. If you try to allocate more queues
it will not work. This is also a limitation of the card's firmware.

Most KFD ioctl functions are not yet implemented. The ones that are relevant
to this card will be implemented eventually.


-----

<p align="center">Copyright&copy; 2019-2023 Advanced Micro Devices, Inc.</p>

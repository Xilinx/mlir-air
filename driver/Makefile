# https://www.kernel.org/doc/html/latest/kbuild/modules.html

obj-m := amdair.o
amdair-y := main.o chardev.o

ifeq ($(KERNELRELEASE),)
KDIR ?= /usr/src/linux-headers-`uname -r`

default:
	make -C $(KDIR) M=$$PWD
endif # KERNELRELEASE

clean:
	rm -f *.o *.ko *.a *.order *.mod.c *.symvers *.mod

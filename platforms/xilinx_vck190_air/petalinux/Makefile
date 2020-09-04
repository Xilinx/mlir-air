CPU_ARCH=a72
OUTPUT=../platform_repo/tmp/sw_components/
SYSROOT=../platform_repo/sysroot
#all: refresh_hw xrt zocl kernel_config rootfs_config linux sw_comp bootimage

all: refresh_hw xrt zocl kernel_config rootfs_config linux bootimage

export COMMON_RFS_KRNL_SYSROOT = TRUE

refresh_hw:
	petalinux-config --get-hw-description=${XSA_DIR} --silentconfig

sw_config: xrt zocl kernel_config rootfs_config

xrt:
	petalinux-config -c xrt --silentconfig

zocl:
	petalinux-config -c zocl --silentconfig

kernel_config:
	petalinux-config -c kernel --silentconfig

rootfs_config:
	petalinux-config -c rootfs --silentconfig

linux:
	petalinux-build

ifeq ($(COMMON_RFS_KRNL_SYSROOT), TRUE)
sw_comp:
	mkdir -p $(OUTPUT)
	cp -rf $(XSA_DIR)/src $(OUTPUT)
	mkdir -p ${OUTPUT}/src/boot
	cp -f images/linux/boot.scr ${OUTPUT}/src/${CPU_ARCH}/xrt/image/boot.scr
	#cp -f images/linux/plm.elf ${OUTPUT}/src/boot/plm.elf
	cp -f images/linux/psmfw.elf ${OUTPUT}/src/boot/psmfw.elf
	cp -f images/linux/system.dtb ${OUTPUT}/src/boot/system.dtb
	cp -f images/linux/bl31.elf ${OUTPUT}/src/boot/bl31.elf
	cp -f images/linux/u-boot.elf ${OUTPUT}/src/boot/u-boot.elf
	#cp -f images/linux/pmc_cdo.bin ${OUTPUT}/src/boot/pmc_cdo.bin
	#cp -f temp/plm.elf ${OUTPUT}/src/boot/plm.elf

else
sw_comp:
	mkdir -p $(OUTPUT)
	cp -rf $(XSA_DIR)/src $(OUTPUT)
	mkdir -p ${OUTPUT}/src/boot
	mkdir -p ${OUTPUT}/src/${CPU_ARCH}/xrt/filesystem
	cp -f images/linux/Image ${OUTPUT}/src/${CPU_ARCH}/xrt/image/Image
	cp -f images/linux/boot.scr ${OUTPUT}/src/${CPU_ARCH}/xrt/image/boot.scr
	cp -f images/linux/rootfs.tar.gz ${OUTPUT}/src/${CPU_ARCH}/xrt/filesystem/rootfs.tar.gz
	cp -f images/linux/rootfs.ext4 ${OUTPUT}/src/${CPU_ARCH}/xrt/filesystem/rootfs.ext4
	cp -f images/linux/rootfs.cpio.gz.u-boot ${OUTPUT}/src/${CPU_ARCH}/xrt/filesystem/rootfs.cpio.gz.u-boot


	#cp -f images/linux/plm.elf ${OUTPUT}/src/boot/plm.elf
	cp -f images/linux/psmfw.elf ${OUTPUT}/src/boot/psmfw.elf
	cp -f images/linux/system.dtb ${OUTPUT}/src/boot/system.dtb
	cp -f images/linux/bl31.elf ${OUTPUT}/src/boot/bl31.elf
	cp -f images/linux/u-boot.elf ${OUTPUT}/src/boot/u-boot.elf
	#cp -f images/linux/pmc_cdo.bin ${OUTPUT}/src/boot/pmc_cdo.bin
	#cp -f temp/plm.elf ${OUTPUT}/src/boot/plm.elf
endif

bootimage:
	@echo "BOOT image for base platforms"
	petalinux-package --boot --plm --psmfw --uboot --dtb

sysroot:
	mkdir -p ${SYSROOT}
	petalinux-build --sdk
	petalinux-package --sysroot -d ${SYSROOT}

clean:
	petalinux-build -x cleanall
	$(RM) -r images/linux/*

ultraclean: clean
	${RM} -r build ip_cache components


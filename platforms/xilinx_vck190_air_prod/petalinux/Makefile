SYSROOT=./sysroot

all: refresh_hw xrt zocl kernel_config rootfs_config linux bootimage

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

bootimage:
	@echo "BOOT image for base platforms"
	petalinux-package --boot --plm --psmfw --uboot --dtb

sysroot:
	mkdir -p ${SYSROOT}
	petalinux-build --sdk
	petalinux-package --sysroot -d ${SYSROOT}

bsp:
	petalinux-package --bsp -o xilinx_vck190_air.bsp -p .

clean:
	$(RM) -r images/linux/*
	petalinux-build -x cleanall || true
	${RM} -r build ip_cache components


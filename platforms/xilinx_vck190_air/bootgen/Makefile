
all: BOOT.BIN

BOOT.BIN: bl31.elf  bootgen.bif  plm.elf  psmfw.elf  system.dtb  u-boot.elf  xilinx_vck190_air.pdi
	bootgen -arch versal -image bootgen.bif -w -o BOOT.BIN

clean:
	rm -f BOOT.BIN

all: BOOT.BIN

BOOT.BIN: bl31.elf bootgen.bif plm.elf psmfw.elf system.dtb u-boot.elf xilinx_vck190_air.pdi
	bootgen -arch versal -image bootgen.bif -w -o BOOT.BIN

%.elf: ../petalinux/images/linux/%.elf
	cp $< $@

%.dtb: ../petalinux/images/linux/%.dtb
	cp $< $@

xilinx_vck190_air.pdi: ../aie_platform/_x/link/vivado/vpl/prj/prj.runs/impl_1/project_1_wrapper.rcdo pdi/boot_image_mb.bif pdi/microblaze_reset.cdo
	bootgen -arch versal -image pdi/boot_image_mb.bif -w -o xilinx_vck190_air.pdi

pdi/microblaze_reset.cdo: pdi/microblaze_reset.txt
	cdoutil -output-binary-be -output-file pdi/microblaze_reset.cdo pdi/microblaze_reset.txt

clean:
	rm -f BOOT.BIN xilinx_vck190_air.pdi pdi/microblaze_reset.cdo bl31.elf plm.elf psmfw.elf system.dtb u-boot.elf

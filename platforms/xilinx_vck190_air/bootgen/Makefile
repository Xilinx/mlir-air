
all: BOOT.BIN

BOOT.BIN: bl31.elf bootgen.bif plm.elf psmfw.elf system.dtb u-boot.elf xilinx_vck190_air.pdi
	bootgen -arch versal -image bootgen.bif -w -o BOOT.BIN

%.elf: ../petalinux/images/linux/%.elf
	cp $< $@

%.dtb: ../petalinux/images/linux/%.dtb
	cp $< $@

xilinx_vck190_air.pdi: ../vivado/myproj/project_1.runs/impl_1/project_1_wrapper.rcdo pdi/project_1_wrapper.pdi.bif pdi/microblaze_reset.cdo
	bootgen -arch versal -image pdi/project_1_wrapper.pdi.bif -w -o xilinx_vck190_air.pdi

pdi/microblaze_reset.cdo: pdi/microblaze_reset.txt
	cdoutil -output-binary-be -output-file pdi/microblaze_reset.cdo pdi/microblaze_reset.txt

clean:
	rm -f BOOT.BIN xilinx_vck190_air.pdi pdi/microblaze_reset.cdo bl31.elf plm.elf psmfw.elf system.dtb u-boot.elf

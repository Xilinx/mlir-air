VIVADO=$(XILINX_VIVADO)/bin/vivado

all: hw

hw:
	$(VIVADO) -mode batch -notrace -source xilinx_vck5000_air_xsa.tcl

clean:
	$(RM) -rf vivado.* vitis* xrc.* myproj *.backup.* NOC_Power.xpe .Xil *.str Work

ultraclean: clean
	$(RM) -rf $(PLATFORM).xsa $(PLATFORM)_presynth.pdi

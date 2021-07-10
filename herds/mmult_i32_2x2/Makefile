
include ../config.make

all: elfs aie_ctrl.so herd.exe

%.o: %.cpp aie.inc
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: $(ARM_OBJ_DIR)/%.o test_library.o
	$(CC) $(LDFLAGS) -o $@ $^

mmult.mlir: mmult.py
	python3 $< > $@ 

aie.inc: aie.herd_0.mlir
	$(AIE_OPT) -lower-affine --convert-scf-to-std \
				--aie-create-flows --aie-find-flows \
				--aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

mmult.air.mlir: mmult.mlir
	aten-opt  -o $@ $< \
			-aten-to-air \
			-air-to-linalg \
			-air-linalg-codegen='herd-size=2,2' \
			-affine-to-air \
			-convert-linalg-to-affine-loops

aie_ctrl.mlir: mmult.air.mlir
	$(ATEN_OPT) -air-to-aie="air-to-aie-emit-while-loop=true air-to-aie-row-offset=2 air-to-aie-col-offset=7 air-to-aie-output-prefix=/dev/null" \
				--convert-vector-to-llvm \
				-air-to-std \
				-airrt-to-llvm \
				-aten-lowering \
				-o $@ $<

aie_ctrl.llvm.mlir: aie_ctrl.mlir
	$(ATEN_OPT) -air-return-elimination \
		--lower-affine \
		--convert-scf-to-std \
		--convert-std-to-llvm='emit-c-wrappers'  \
		-o $@ $<

aie.herd_0.mlir: mmult.air.mlir
	aten-opt $< -convert-linalg-to-loops \
				-air-to-aie="air-to-aie-emit-while-loop=true air-to-aie-row-offset=2 air-to-aie-col-offset=7 air-to-aie-output-prefix=./" -o /dev/null

peano.mlir: aie.herd_0.mlir
	aten-opt -lower-affine -o $@ $<

elfs: peano.mlir
	aiecc.py --no-xbridge $<

aie_ctrl.o: aie_ctrl.llvm.mlir
	mlir-translate --mlir-to-llvmir $< | llc --march=aarch64 --filetype=obj -o $@

aie_ctrl.so: aie_ctrl.o
	$(CC) $(CFLAGS) -shared $< -o $@

clean::
	rm -rf acdc_project mmult.*mlir aie*.mlir peano.mlir *.elf aie_ctrl.* *.exe *.inc *.o

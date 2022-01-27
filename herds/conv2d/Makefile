
include ../config.make

#all: conv2d_air.a test_cpu.exe
all: test_cpu.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: $(ARM_OBJ_DIR)/%.o
	$(CC) $^ \
		$(LDFLAGS) \
		-rdynamic \
		-lxaiengine \
		-lmetal \
		-lopen_amp \
		-ldl \
		-o $@

conv2d_aten.mlir: conv2d.py
	python3 $< > $@

# convert to affine loops with the ordering suitable for multiple tilings...
conv2d_affine.mlir: conv2d_aten.mlir
	$(ATEN_OPT) -aten-to-xten \
				-xten-to-affine='xten-loop-order=3,0,1,2,4' \
				-cse -cse \
				-o $@ $<

# first tiling to model L3->L2 tiling associated with each 'logical column' of overlay/herd
conv2d_tiled_0.mlir: conv2d_affine.mlir
	$(AIR_OPT) -affine-loop-opt='affine-opt-tile-sizes=104' \
				-affine-loop-normalize -affine-simplify-structures \
				-cse \
				-o $@ $<

# second tiling to model L1 sub-volume partitioning (minus the weights/OFM tiling in 'logical rows' 
# of the overlay)
conv2d_tiled_1.mlir: conv2d_tiled_0.mlir
	$(AIR_OPT) -affine-loop-normalize \
				-affine-loop-opt='affine-opt-tile-sizes=1,2,16,26,1,1,13' \
				-affine-loop-normalize -affine-simplify-structures \
				-cse \
				-o $@ $<

# third tiling to model weights/OFM partitioning in logical rows 
conv2d_tiled_2.mlir: conv2d_tiled_1.mlir
	$(AIR_OPT) -affine-loop-normalize \
				-affine-loop-opt='affine-opt-tile-sizes=1,8,1,1,1,1,1,1,1,1,1' \
				-affine-loop-normalize -affine-simplify-structures \
				-cse \
				-o $@ $<

# permute loops so that 'processing iterations' mapped to MLLib kernels are the inner-most in the loop nest
# TODO - it seems that loop permutation does NOT work for more that the first 8 loops???? This should be 
# sufficient for conv2d, but the more general cases should be handled too.
conv2d_perm_0.mlir: conv2d_tiled_2.mlir
	$(AIR_OPT) -air-regularize-loop \
				-air-loop-permutation="loop-order=0,1,2,3,6,7,4,5,8,9,10,11" \
				-o  $@ $<

# TODO - add label to loop nest since permutation seems to get rid of it...
conv2d_perm_1.mlir: conv2d_perm_0.mlir
	cat conv2d_perm_0.mlir | awk 'BEGIN {CURLY = 0; AFFINE = 0} { if ($$1 == "affine.store") AFFINE = 1; if (AFFINE > 0 && $$1 == "}") CURLY += 1; if (CURLY == 12) {print $$0 " {affine_opt_label = \"\"}" ; CURLY = 0}  else print $$0}' > conv2d_perm_1.mlir


# copy buffers/memrefs at the 'right' depth to support modeling of DMA transfers
# TODO: unable to fully control placement of memrefs allocation resulting in 'code' between 
# first and second affine loop; this code makes these two loops appear not perfectly nested 
# casuing a failure (crash now) when attempting to assign to a herd. Workaround is to manually
# move the alloc/DMA code inside second loop (completely valid)
conv2d_tiled_3.mlir: conv2d_perm_1.mlir
	$(AIR_OPT) -affine-loop-opt='affine-opt-copy-depths=1 affine-opt-copy-fast-space=2' \
		        -o $@ $<

# TODO: workaround to manually move memref allocation/deallocation for L2 tiles inside second loop
conv2d_tiled_4.mlir: conv2d_tiled_3.mlir
	sed '/affine.for %arg4 = 0 to 4 {/d' conv2d_tiled_3.mlir > t.mlir; \
    sed 's/affine.for %arg3 = 0 to 4 {/affine.for %arg3 = 0 to 4 {\n      affine.for %arg4 = 0 to 4 {/' t.mlir > t2.mlir; \
	sed '/memref.dealloc %7 : memref<1xi32>/d' t2.mlir | sed '/memref.dealloc %6 : memref<1x48x106x416xf32, 2>/d' > t3.mlir; \
	sed 's/memref.dealloc %10 : memref<1x16x104x416xf32, 2>/memref.dealloc %10 : memref<1x16x104x416xf32, 2>\n      memref.dealloc %7 : memref<1xi32>\n      memref.dealloc %6 : memref<1x48x106x416xf32, 2>/' t3.mlir > conv2d_tiled_4.mlir
		
# assign herd/overlay i.e. convert 2 outer loops into affine parallel structure as precursor to lowering to AIE dialect
conv2d_air_0.mlir: conv2d_tiled_4.mlir
	$(AIR_OPT) -air-herd-assign=herd-assign-depth=0 \
		        -affine-to-air \
				-cse \
				-o $@ $<

conv2d_air.a: conv2d_air_0.mlir
	aircc.py -o $@ -row-offset=2 -col-offset=7 -v $<

# lowering to AIE dialect i.e. conversion of affine parallel structure into 'aiert' structures (module/herd) with corresponding 'metadata' as attributes that describe shim allocations
conv2d_aie_0.mlir: conv2d_air_0.mlir
	$(AIR_OPT) --pass-pipeline="air-to-aie{air-to-aie-emit-while-loop=true air-to-aie-row-offset=2 air-to-aie-col-offset=7 air-to-aie-output-prefix=/dev/null}"\
		-o $@ $<

conv2d_cpu.std.mlir: conv2d_air_0.mlir
	$(AIR_OPT) \
		-air-to-std='lower-to-cpu=true' -airrt-to-llvm -cse $< | \
	$(ATEN_OPT) \
		-lower-affine -cse \
		-tensor-constant-bufferize -func-bufferize -finalizing-bufferize \
		-convert-scf-to-std \
		-o $@


conv2d_cpu.llvm.mlir: conv2d_cpu.std.mlir
	$(ATEN_OPT) \
		-convert-std-to-llvm \
		-lower-to-libaten \
		-o $@ $<

conv2d_cpu.o: conv2d_cpu.llvm.mlir
	mlir-translate --mlir-to-llvmir $< | llc -relocation-model=pic --filetype=obj -o $@

test_cpu.o: test_cpu.cpp
	clang++ -I$(ACDC_AIR)/runtime_lib/airhost/include -c -o $@ $<

test_cpu.exe: test_cpu.o conv2d_cpu.o
	clang -o $@ $^ -lstdc++

clean:
	rm -rf aie_inc.cpp *.o *.exe *.elf ./$(AIE_SRC_DIR)
	rm -rf aie_ctrl.so aie_ctrl.o
	rm -rf *.mlir air_project

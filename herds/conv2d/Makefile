# General Makefile for tests

AIE_SRC_DIR = acdc_project
ARM_SRC_DIR = .
ARM_OBJ_DIR = .

AIE_OPT = aie-opt
AIE_XLATE = aie-translate
AIR_OPT = air-opt

AIECC = aiecc.py
SYSROOT = /group/xrlabs/platforms/pynq-vck190-sysroot/
CC = clang --target=aarch64-linux-gnu -std=c++11 --sysroot=$(SYSROOT) \
   -I$(SYSROOT)/opt/xaiengine/include -L$(SYSROOT)/opt/xaiengine/lib -fuse-ld=lld -rdynamic -lxaiengine

uname_p := $(shell uname -p)
ifeq ($(uname_p),aarch64)
	CFLAGS += -I/opt/xaiengine/include
	LDFLAGS += -L/opt/xaiengine/lib
endif

CFLAGS += -I../../lib/include

all: aie_ctrl.so elfs

%.o: %.cpp aie_inc.cpp
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

conv2d.mlir: conv2d.py
	python3 $< > $@

aie_inc.cpp: aie.mlir
	$(AIE_OPT) -lower-affine --convert-scf-to-std \
			   	--aie-create-flows --aie-find-flows \
				--aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

# convert to affine loops with the ordering suitable for multiple tilings...
# TODO: loop ordering is hardcoded in aten-to-air lowering for now
# first tiling to model L3->L2 tiling associated with each 'logical column' of overlay/herd
conv2d_tiled_0.mlir: conv2d.mlir
	$(AIR_OPT) -aten-to-air \
				-air-to-affine='air-loop-order=3,0,1,2,4' \
				-affine-loop-opt='affine-opt-tile-sizes=104' \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

# second tiling to model L1 sub-volume partitioning (minus the weights/OFM tiling in 'logical rows' 
# of the overlay)
conv2d_tiled_1.mlir: conv2d_tiled_0.mlir
	$(AIR_OPT) -affine-loop-normalize \
				-affine-loop-opt='affine-opt-tile-sizes=1,2,16,26,1,1,13' \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

# third tiling to model weights/OFM partitioning in logical rows 
conv2d_tiled_2.mlir: conv2d_tiled_1.mlir
	$(AIR_OPT) -affine-loop-normalize \
				-affine-loop-opt='affine-opt-tile-sizes=1,8,1,1,1,1,1,1,1,1,1' \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

# copy buffers/memrefs at the 'right' depth to support modeling of DMA transfers
# TODO: unable to fully control placement of memrefs allocation resulting in 'code' between 
# first and second affine loop; this code makes these two loops appear not perfectly nested 
# casuing a failure (crash now) when attempting to assign to a herd. Workaround is to manually
# move the alloc/DMA code inside second loop (completely valid)
conv2d_tiled_3.mlir: conv2d_tiled_2.mlir
	$(AIR_OPT) -affine-loop-opt='affine-opt-copy-depths=1 affine-opt-copy-fast-space=2' \
		        -o $@ $<

# TODO: workaround to manually move memref allocation/deallocation for L2 tiles inside second loop
conv2d_tiled_4.mlir: conv2d_tiled_3.mlir
	sed '/affine.for %arg2 = 0 to 4 {/d' conv2d_tiled_3.mlir > t.mlir; \
    sed 's/affine.for %arg1 = 0 to 4 {/affine.for %arg1 = 0 to 4 {\n      affine.for %arg2 = 0 to 4 {/' t.mlir > t2.mlir; \
	sed '/memref.dealloc %7 : memref<1xi32>/d' t2.mlir | sed '/memref.dealloc %6 : memref<1x48x106x416xf32, 2>/d' > t3.mlir; \
	sed 's/memref.dealloc %10 : memref<1x16x104x416xf32, 2>/memref.dealloc %10 : memref<1x16x104x416xf32, 2>\n      memref.dealloc %7 : memref<1xi32>\n      memref.dealloc %6 : memref<1x48x106x416xf32, 2>/' t3.mlir > conv2d_tiled_4.mlir
		
# assign herd/overlay only.
# TODO: lowering affine-to-air crashes currently; separating that step
conv2d_air.mlir: conv2d_tiled_4.mlir
	$(AIR_OPT) -acap-herd-assign=herd-assign-depth=0 \
				-cse \
				-o $@ $<

aie.mlir: conv2d_air.mlir
	$(AIR_OPT) -air-to-aie \
				-air-to-aie-emit-while-loop=true \
				-air-to-aie-row-offset 2 \
				-air-to-aie-col-offset 7 \
				-air-to-aie-output-file $@ $< \
				-o aie_t.mlir
peano.mlir: aie.mlir
	$(AIR_OPT) -lower-affine -o $@ $< 

elfs: peano.mlir
	$(AIECC) --no-xbridge peano.mlir


aie_ctrl.mlir: conv2d_air.mlir
	$(AIR_OPT) -air-to-aie $< -air-to-aie-output-file /dev/null \
				-air-to-aie-row-offset 2 \
				-air-to-aie-col-offset 7 \
				-air-to-std \
				-aten-lowering \
				-o $@

aie_ctrl.llvm.mlir: aie_ctrl.mlir
	$(AIR_OPT) -air-return-elimination \
		--lower-affine \
		--convert-scf-to-std \
		--convert-std-to-llvm='emit-c-wrappers'  \
		-o $@ $<

aie_ctrl.o: aie_ctrl.llvm.mlir
	mlir-translate --mlir-to-llvmir $< | llc --march=aarch64 --filetype=obj -o $@

aie_ctrl.so: aie_ctrl.o
	$(CC) -shared $< -o $@

%.elf: $(AIE_SRC_DIR)/main.cc
	cd ./$(AIE_SRC_DIR) && \
	xchessmk $*.prx && \
	cp work/Release_LLVM/$*.prx/$* ../$*.elf

clean:
	rm -rf aie_inc.cpp *.o *.exe *.elf ./$(AIE_SRC_DIR)
	rm -rf aie_ctrl.so aie_ctrl.o
	rm -rf *.mlir

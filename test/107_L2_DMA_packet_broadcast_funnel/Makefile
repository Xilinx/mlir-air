
SOURCE_FILES = test.cpp

include ../config.make
all: aie_inc.cpp test.exe core_7_4.elf
#all: test.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: %.o
	$(CC) $< \
		$(LDFLAGS) \
		-rdynamic \
		-lxaiengine \
		-lmetal \
		-lopen_amp \
		-ldl \
		-o $@

aie_inc.cpp: aie.mlir
	aie-opt --aie-create-packet-flows --aie-assign-buffer-addresses $^ | aie-translate --aie-generate-xaie -o $@

core_7_4.elf: aie.mlir
	aiecc.py --pathfinder aie.mlir

clean::
	rm -rf *.o *.exe aie_inc.cpp

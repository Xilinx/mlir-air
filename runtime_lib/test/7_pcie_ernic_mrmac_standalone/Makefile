CC=g++
INCLUDES=-I../../airhost/include
CFLAGS=${INCLUDES}
OBJFILES=pcie-ernic.o pcie-ernic-dev-mem-allocator.o

default: pci driver1 driver2

.PHONY: clean pci

driver1: ${OBJFILES} driver1.o
	$(CC) ${CFLAGS} -o driver1 driver1.o ${OBJFILES}

driver2: ${OBJFILES} driver2.o
	$(CC) ${CFLAGS} -o driver2 driver2.o ${OBJFILES}

driver2.o:
	$(CC) ${CFLAGS} -c driver2.c

driver1.o:
	$(CC) ${CFLAGS} -c driver1.c

pcie-ernic.o:
	$(CC) ${CFLAGS} -c ../../airhost/pcie-ernic.cpp

pcie-ernic-dev-mem-allocator.o:
	$(CC) ${CFLAGS} -c ../../airhost/pcie-ernic-dev-mem-allocator.cpp

# This is to automate getting the BDF of the
# device and putting it in a header file. This
# has the additional benefit of making the compilation
# fail if you are trying to access a BAR that isn't 
# on the device.
pci:
	../scripts/dump_pci_bdf.sh ../../airhost/include/

# This is going to run a script which sets the
# PCI command register as well as adds huge pages
# to the system. Only need to run once when the 
# machine is rebooted
init_pci:
	sudo ../scripts/init_pci.sh

clean:
	rm -f driver1
	rm -f driver1.o
	rm -f driver2
	rm -f driver2.o
	rm -f pcie-bdf.h
	rm -f ${OBJFILES}

CC=g++
INCLUDES=-I../../../install-pcie/runtime_lib/airhost/include/ 
CFLAGS=${INCLUDES}
OBJFILES=pcie-ernic.o pcie-ernic-dev-mem-allocator.o

default: driver1 driver2

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
	$(CC) ${CFLAGS} -c ../../../install-pcie/runtime_lib/airhost/pcie-ernic.cpp

pcie-ernic-dev-mem-allocator.o:
	$(CC) ${CFLAGS} -c ../../../install-pcie/runtime_lib/airhost/pcie-ernic-dev-mem-allocator.cpp

clean:
	rm -f driver1
	rm -f driver1.o
	rm -f driver2
	rm -f driver2.o
	rm -f ${OBJFILES}

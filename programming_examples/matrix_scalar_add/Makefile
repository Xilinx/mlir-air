srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

single-core-dma:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/run.py single-core-dma

clean:
	rm -rf build __pycache__

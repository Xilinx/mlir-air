srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

run:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/run.py

clean:
	rm -rf build __pycache__

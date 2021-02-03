include ../config.make

export BUILD_DIR := $(shell readlink -f ../build$(BUILD)/air)

${BUILD_DIR}/.dir:
	mkdir -p $(dir $@)
	touch $@

build: ${BUILD_DIR}/.dir
	echo `pwd`; ./config.sh ${BUILD_DIR} . ${INSTALL_DIR}; cd ${BUILD_DIR}; ninja

test:
	cd ${BUILD_DIR}; ninja check-all

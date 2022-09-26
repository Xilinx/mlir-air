FILESEXTRAPATHS_prepend := "${THISDIR}/${PN}:"

SRC_URI_append = " file://bsp.cfg"
KERNEL_FEATURES_append = " bsp.cfg"

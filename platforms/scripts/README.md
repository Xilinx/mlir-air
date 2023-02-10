# Platform scripts

The 'scripts' directory contains files for platform management and development.
There are 2 general uses: driver development and firmware development.

## Driver development
We use QEMU as a host platform for driver development because it avoids the
need to reboot the physical host when something goes wrong. That has an impact
on other users and takes more time. The 'start-qemu.sh' script is used to run
QEMU in its current configuration. It currently supports a buildroot target and
an Ubuntu target. The script assumes that the dependencies already exist before
trying to run QEMU.

The VCK5000 hardware is directly assigned using VFIO to QEMU. VFIO comes with
its own requirements, such as enabling the IOMMU on the host system and
assigning all PCIe devices that share a single IOMMU as a group.

### Buildroot
Buildroot [ https://buildroot.org/ ] is an easy way to generate a custom Linux
operating system with minimal system size and memory footprint. We use it
because it boots quickly and is convenient for development cycles, with the
source code being kept on the host.

To set up the environment, use the 'setup-qemu.sh' script. This will create a
config file for you if you don't already have one, or update an existing one if
it already exists. The config file is called 'env_config' and lives in the
scripts directory, but should not be added to source control because it is
different for every user. It contains several parameters as environment
variables that are used by the 'start-qemu.sh' script. It will download
buildroot and linux kernel projects (if needed) and check out the correct
branches. You can use existing repos if you already have these projects and
want to save the disk space. The script will create the 'air' branch so you
can switch between branches without losing your place. It will then create
a config file for buildroot called 'br_config' (also in the scripts directory).
The buildroot project will then be ready for building, using 'make' as usual.
The output will be in OUTPUT_DIR (the 'buildroot' directory by default).
The driver is built as an out-of-tree Linux kernel module, but included in the
buildroot build with a custom driver package. This is sufficient for testing
the driver. For further customization and instructions, please refer to the
buildroot manual under 'out-of-tree builds'.

### Ubuntu
Ubuntu has some advantages over 'buildroot' when a more fully featured user
space environment is required. This is the case when trying to run pre-built
executables that require certain .so libraries, or installations of large
packages such as CUDA. Our setup for Ubuntu uses a qcow2 virtual hard drive,
called 'ubuntu-20.04.qcow2'. This image is not provided, you must set this up
yourself. Setting up Ubuntu as a virtual machine is well-documented elsewhere,
so I will not go into all of the details here. The basic idea is to create an
empty qcow2 image, download an Ubuntu ISO image, use the '-cdrom' (or
equivalent) QEMU option to load it and install Ubuntu into the qcow2 image.

## Firmware and FPGA loading
FPGA images for the VCK5000 are generally delivered as a PDI file. This is
a standard development flow using Vitis and Vivado tools. The PDI is a wrapper
format, and contains several other files within. It generally contains a
bitstream targeting the FPGA and an ELF file targeting the embedded ARM
processors. Programming the PDI can potentially impact the PCIe bus, since
the VCK5000 endpoint becomes unresponsive to the host's root complex during
the programming operation. In some cases, it can require a reboot of the host.
To reduce the time of development cycles (and general system impact on other
users) it is advantageous to program the ELF file separately, which does not
have any impact on the PCIe bus.

The chosen method for programming is using the 'xsct' tool with JTAG-over-USB.
Xsct reads a TCL script (load.tcl) to program the device. The filenames for
the ELF and PDI files are passed as parameters using environment variables.
These environment variables can be set in a variety of ways. Regardless of how
they are set, the TCL script will determine which files are to be programmed.
They are both optional so a single PDI file that contains an embedded ELF
can program the FPGA and ARM, or alternatively a standalone ELF can be used
without disturbing the FPGA.

### Multiple cards
In systems that contain multiple VCK5000 cards, it is necessary to target a
specific card for programming. The loading script has another parameter called
CARD_IDX that indicates which card is to be programmed. It is an index of
installed cards based on the JTAG scan-chain ordering, as seen by 'xsct'.
At this time, I cannot say for certain if there is any deterministic correlation
between the ordering of PCIe vs USB vs JTAG devices. The best advise is to
use the 'xsct' or 'xsdb' tool in interactive mode to first determine the
JTAG ID of the target card to be sure you are programming the correct one.

-----

<p align="center">Copyright&copy; 2019-2023 Advanced Micro Devices, Inc.</p>

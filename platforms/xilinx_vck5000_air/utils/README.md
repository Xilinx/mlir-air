# Utils

## pci_remove.sh

In some cases, updating the FPGA image or firmware of a live PCIe device can
cause the host to freeze (become unresponsive). To avoid this issue, you can
use the pci_remove.sh script. It disconnects the device from its driver,
preventing any further access by the host until the device is re-enabled.
To run the script, you must pass the device ID and vendor ID of the card that
you wish to disable. After the script runs, you can safely update the FPGA or
firmware. You must have root (i.e. sudo) access to run this script, as it
relies on a privileged kernel interface.

To return the card to a working state and re-bind it with its driver, use:

`sudo sh -c "echo '1' > /sys/bus/pci/rescan"`

This will find all unbound PCIe (and PCI) devices, not just the card you are
working with. All PCIe devices that are already known to Linux will continue
to work as usual, uninterrupted.

## init_pci.sh

There are two things that need to be done to initialize the PCIe EP when using
the ERNIC IPs. First, if host memory is being intiailzed using the ERNIC
memory allocator, then huge pages must be allocated. This script checks to 
see if any huge pages are allocated, and allocates 50 additional huge pages. 
Also, in some systems, the OS does not initialize the BARs automatically. 
This script checks if the BARs are disabled on the endpoint, and if so
automatically sets the PCIe registers to enable the BARs.

## dump_pci_bdf.sh

In runtime_lib/test/7_pcie_ernic_mrmac_standalone/ the test bypasses the AIR
infrastructure to directly communicate with the ERNICs. Because of this, it
loses the AIR functionality to discover PCIe devices. Therefore, this script
will find all VCK5000 devices and output the BDFs into a header file
which can be read by the test.


-----

<p align="center">Copyright&copy; 2019-2022 Advanced Micro Devices, Inc.</p>

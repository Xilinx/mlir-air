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

-----

<p align="center">Copyright&copy; 2019-2022 Advanced Micro Devices, Inc.</p>

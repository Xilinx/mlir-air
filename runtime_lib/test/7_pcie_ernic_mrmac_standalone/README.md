## 7_pcie_ernic_mrmac_standalone

This test utilizes the PCIe ERNIC functionality in the AIR runtime to directly communicate with the 
ERNICs in the VCK5000 AIR Scale Out platform. This test contains two processes, driver1 and driver2,
which are meant to run across two machines, each with a VCK5000 with a 100G Ethernet cable connecting 
two of the QSFP cages. Specifically, QSFP cage 1 on one machine should be connected to QSFP 
cage 2 on another machine. driver1 should run on the machine with QSFP cage 1 connected, and 
driver2 should run on the machine with QSFP cage 2 connected.

What the software does:
	1. Driver 1 and two both create their queues and register memory with their respective ERNICs
	   During this driver2 also configures both CMACs.
	2. Driver 1 performs two SENDs to send (1) the rkey (2) virtual address of its registered memory
	3. Driver 2 polls on it's RQ, receives the rkey and virtual address
	4. Driver 2 sends a READ WQE to the first 256B of the registered memory, should read 0xedededed
	5. Driver 2 sends a WRITE WQE to a later part of the registered memory, writing what was read in the previous step
	6. Both drivers print the parts of the memory region that were read/written to. Should both be 0xedededed
	7. Both drivers print some statistics gathered from their ERNIC. Should look as follows:
               a. Driver 1:
                        Number of SEND + WRITE packets received: 1
                        Number of READ RESPONSE packets received: 0
                        Number of READ + WRITE WQEs processed: 0
                        Number of ACKs received: 2
                        Number of ACKs sent: 1
                        Numer of NACKs sent: 0
                        Number of READ RESPONSE sent: 1
                b. Driver 2:
                        Number of SEND + WRITE packets received: 2
                        Number of READ RESPONSE packets received: 1
                        Number of READ + WRITE WQEs processed: 2
                        Number of ACKs received: 1
                        Number of ACKs sent: 2
                        Numer of NACKs sent: 0
                        Number of READ RESPONSE sent: 0

Steps when running the software:
    1. Run `lspci -vvv -s <BUS>.<DOMAIN>.*` to ensure that you can see the platform.
    2. If the BARs of the device are disabled, go to `/mlir-air/platforms/xilinx_vck5000_air_scale_out/utils/` and run `sudo ./init_pci.sh`.
    3. Generate the `pcie-bdf.h` to point to the proper BAR addresses. Can be done by going to `/mlir-air/platforms/xilinx_vck5000_air_scale_out/utils/` and running `./dump_pci_bdf.sh ../../../runtime_lib/test/7_pcie_ernic_mrmac_standalone/`
    4. Run `make`.
    5. Run "sudo ./driver2" on one machine, this will poll on the ERNIC's RQ waiting to receive two RDMA SENDS.
    6. Run "sudo ./driver1" on the other machine. Press [Enter] when prompted to generate two RDMA SENDS. You should then see driver 2 complete and print statistics from the ERNIC.
    7. Press [Enter] on driver1 to complete and print statistics from the ERNIC
    8. Verify that the statistics printed for each ERNIC is correct

Successful driver1 output (Note this is after running 7 times so ERNIC registers may be different):
```
[INFO] AXIL memory mapped into userspace
        VA: 0x7f85f6b58000
        Size: 2097152
        Offset: 786432
Opening /sys/bus/pci/devices/0000:21:00.0/resource0 with size 67108864
[INFO] Device memory mapped into userspace
        VA: 0x7f85f2b58000
        Segment Offset: 0x40000
        Global Offset: 0x800000000
        Size: 67108864
[INFO] Configuring MRMAC
        MRMAC 0 Offset: 0x110000
        MRMAC 1 Offset: 0x120000
        MRMAC Reset Offset: 0x100000
[INFO] MRMAC Bring up process:
        *** Write M_AXI_LITE GPIO to assert gt_reset_all_in  ***
        dual_reset is set to: 1
        *** check for stat_mst_reset_done = 1 from test bench reset control*** 
        *** First GT reset done***
        *** check for stat_mst_reset_done = 1 from test bench reset control*** 
[INFO] Configuring MR-MAC
        Configuring MR-MAC at offset 110000
[INFO] MR-MAC reset control GPIO @ 
INFO : READ MRMAC CORE VERSION ..........
 MR-MAC Core_Version  =  1  
INFO : START MRMAC CONFIGURATION ..........
 *** check for stat_mst_reset_done = 1 *** 
INFO :Polling on RX ALIGNED ..........
[WARNING] RX synced error of MRMAC at offset 0x110000. Assuming not connected and reutrning
[INFO] Configuring MR-MAC
        Configuring MR-MAC at offset 120000
[INFO] MR-MAC reset control GPIO @ 
INFO : READ MRMAC CORE VERSION ..........
 MR-MAC Core_Version  =  1  
INFO : START MRMAC CONFIGURATION ..........
 *** check for stat_mst_reset_done = 1 *** 
INFO :Polling on RX ALIGNED ..........
[INFO] Polling on MRMAC RX align. RX align status reg  =  182 
[INFO] Polling on MRMAC RX align. RX align status reg  =  182 
[INFO] RX align status reg  =  10007 
[INFO] MRMAC at offset 120000 is aligned!
Data buffer information:
        VA: 0x7f85f2bae000
        PA: 0x800056000
        Size: 512
        On device: 1

Registering memory at VA 0x7f85f2bae000
Memory before remote write : 0xfeeeeeed
[INFO] QP2 State:
QP 2 information:
        Network information:
                Destination MAC MSB: 0x16c4
                Destination MAC LSB: 0x50560f2e
                Destination IP: 0x610c6007
                Destination QP ID: 2
        Memory addresses:
                Queue Depth: 16777472
                RQ:
                        VA: 0x7f85f2bac000
                        PA: 0x800054000
                        Allocated Memory: 4096
                        Queue Depth: 256
                        On device: 1
                SQ:
                        VA: 0x7f85f2baa000
                        PA: 0x800052000
                        Allocated Memory: 4096
                        Queue Depth: 256
                        On device: 1
                CQ:
                        Enabled: 0
                        VA: 0x7f85f2bab000
                        PA: 0x800053000
                        Allocated Memory: 4096
                        Queue Depth: 256
                        On device: 1

rkey_buff:
Data buffer information:
        VA: 0x7f85f2bae200
        PA: 0x800056200
        Size: 4096
        On device: 1
vaddr_buff:
Data buffer information:
        VA: 0x7f85f2baf200
        PA: 0x800057200
        Size: 4096
        On device: 1
Queue created and memory registered
Press [Enter] to continue.....

Sending SEND Packet to ERNIC 0
Sending SEND Packet to ERNIC 0
Two SENDS complete
Press [Enter] to continue.....

Memory written to by other ERNIC : 0xedededed
MRMAC Statistics:

MRMAC at Offset 0x110000 Statistics           

  STAT_TX_TOTAL_PACKETS           = 0,           STAT_RX_TOTAL_PACKETS           = 0

  STAT_TX_TOTAL_GOOD_PACKETS      = 0,           STAT_RX_TOTAL_GOOD_PACKETS      = 0

  STAT_TX_TOTAL_BYTES             = 0,           STAT_RX_BYTES                   = 0

  STAT_TX_TOTAL_GOOD_BYTES        = 0,           STAT_RX_TOTAL_GOOD_BYTES        = 0


MRMAC at Offset 0x120000 Statistics           

  STAT_TX_TOTAL_PACKETS           = 4,           STAT_RX_TOTAL_PACKETS           = 4

  STAT_TX_TOTAL_GOOD_PACKETS      = 4,           STAT_RX_TOTAL_GOOD_PACKETS      = 4

  STAT_TX_TOTAL_BYTES             = 1024,        STAT_RX_BYTES                   = 544

  STAT_TX_TOTAL_GOOD_BYTES        = 1024,        STAT_RX_TOTAL_GOOD_BYTES        = 544

ERNIC 0 State:
Checking ERNIC 0 Status Registers...
        Number of SEND + WRITE packets received: 7
        Number of READ RESPONSE packets received: 0
        Number of READ + WRITE WQEs processed: 0
        Number of ACKs received: 14
        Number of ACKs sent: 7
        Number of NACKs sent: 0
        Number of READ RESPONSE sent: 7
        Error buffer PA: 0x800040000
        Inc Packet Error Buffer PA: 0x800050000
        Response Error BUffer PA: 0x800051000

PASSED
Freeing QP 2
```


Successful driver2 output (Note this is after running 7 times so ERNIC registers may be different):
```
[INFO] AXIL memory mapped into userspace
        VA: 0x7f419f5f6000
        Size: 2097152
        Offset: 524288
Opening /sys/bus/pci/devices/0000:21:00.0/resource0 with size 67108864
[INFO] Device memory mapped into userspace
        VA: 0x7f419b5f6000
        Segment Offset: 0x20000
        Global Offset: 0x800000000
        Size: 67108864
[INFO] Configuring MRMAC
        MRMAC 0 Offset: 0x110000
        MRMAC 1 Offset: 0x120000
        MRMAC Reset Offset: 0x100000
[INFO] MRMAC Bring up process:
        *** Write M_AXI_LITE GPIO to assert gt_reset_all_in  ***
        dual_reset is set to: 1
        *** check for stat_mst_reset_done = 1 from test bench reset control*** 
        *** First GT reset done***
        *** check for stat_mst_reset_done = 1 from test bench reset control*** 
[INFO] Configuring MR-MAC
        Configuring MR-MAC at offset 110000
[INFO] MR-MAC reset control GPIO @ 
INFO : READ MRMAC CORE VERSION ..........
 MR-MAC Core_Version  =  1  
INFO : START MRMAC CONFIGURATION ..........
 *** check for stat_mst_reset_done = 1 *** 
INFO :Polling on RX ALIGNED ..........
[INFO] Polling on MRMAC RX align. RX align status reg  =  182 
[INFO] Polling on MRMAC RX align. RX align status reg  =  182 
[INFO] RX align status reg  =  10007 
[INFO] MRMAC at offset 110000 is aligned!
[INFO] Configuring MR-MAC
        Configuring MR-MAC at offset 120000
[INFO] MR-MAC reset control GPIO @ 
INFO : READ MRMAC CORE VERSION ..........
 MR-MAC Core_Version  =  1  
INFO : START MRMAC CONFIGURATION ..........
 *** check for stat_mst_reset_done = 1 *** 
INFO :Polling on RX ALIGNED ..........
[WARNING] RX synced error of MRMAC at offset 0x120000. Assuming not connected and reutrning
Data buffer information:
        VA: 0x7f419b62c000
        PA: 0x800036000
        Size: 4096
        On device: 1

Memory before remote read: 0x0
Registering memory at VA 0x7f419b62c000
[INFO] QP2 State:
QP 2 information:
        Network information:
                Destination MAC MSB: 0x2f76
                Destination MAC LSB: 0x17dc5e9a
                Destination IP: 0xf38590ba
                Destination QP ID: 2
        Memory addresses:
                Queue Depth: 16777472
                RQ:
                        VA: 0x7f419b62a000
                        PA: 0x800034000
                        Allocated Memory: 4096
                        Queue Depth: 256
                        On device: 1
                SQ:
                        VA: 0x7f419b628000
                        PA: 0x800032000
                        Allocated Memory: 4096
                        Queue Depth: 256
                        On device: 1
                CQ:
                        Enabled: 0
                        VA: 0x7f419b629000
                        PA: 0x800033000
                        Allocated Memory: 4096
                        Queue Depth: 256
                        On device: 1

Polling on first SEND...
rkey read from ERNIC 0: 0x10
Polling on second SEND...
vaddr from ERNIC 0: 0x7f85f2bae000
Memory read from remote: 0xedededed
        next word: 0xedededed
MRMAC States:

MRMAC at Offset 0x110000 Statistics           

  STAT_TX_TOTAL_PACKETS           = 4,           STAT_RX_TOTAL_PACKETS           = 4

  STAT_TX_TOTAL_GOOD_PACKETS      = 4,           STAT_RX_TOTAL_GOOD_PACKETS      = 4

  STAT_TX_TOTAL_BYTES             = 544,         STAT_RX_BYTES                   = 1024

  STAT_TX_TOTAL_GOOD_BYTES        = 544,         STAT_RX_TOTAL_GOOD_BYTES        = 1024


MRMAC at Offset 0x120000 Statistics           

  STAT_TX_TOTAL_PACKETS           = 0,           STAT_RX_TOTAL_PACKETS           = 0

  STAT_TX_TOTAL_GOOD_PACKETS      = 0,           STAT_RX_TOTAL_GOOD_PACKETS      = 0

  STAT_TX_TOTAL_BYTES             = 0,           STAT_RX_BYTES                   = 0

  STAT_TX_TOTAL_GOOD_BYTES        = 0,           STAT_RX_TOTAL_GOOD_BYTES        = 0

ERNIC 1 State:
Checking ERNIC 1 Status Registers...
        Number of SEND + WRITE packets received: 14
        Number of READ RESPONSE packets received: 7
        Number of READ + WRITE WQEs processed: 14
        Number of ACKs received: 7
        Number of ACKs sent: 14
        Number of NACKs sent: 0
        Number of READ RESPONSE sent: 0
        Error buffer PA: 0x800020000
        Inc Packet Error Buffer PA: 0x800030000
        Response Error BUffer PA: 0x800031000

PASSED
Freeing QP 2
```

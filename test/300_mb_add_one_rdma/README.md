# Test 300_mb_add_one_rdma
## How to run the test

This test is utilized to excercise the one-sided RDMA functionality of the AIR Scale Out platform. It performs the same computation as shown in 13_mb_add_one, but it reads the source vector from a remote node utilizing an RDMA READ request. In this test, the process defined in test.cpp is running the AIR kernel, and it communicates over the network to the process defined in driver1.c, which bypasses AIR and directly communicates with one of the ERNICs in the AIR Scale Out platform.

This test can be run in two configurations:

1. Standard mode: in which a QSFP cable is connecting the two cards. As the platform currently does not support dynamic detection of QSFP cages, in this test QSFP cage 1 on one machine should be connected to QSFP cage 2 on the other machine. Then, driver1 should be run on the machine with QSFP cage 1 connected, and test.elf should run on the machine with QSFP cage 2 connected.
2. Loopback mode, in which a QSFP cable is loopbacked between the two cage.

When running in standard mode, driver1.c should have its configure_cmac bit set to `true` as it must configure the MAC in the platform. When the test is running in loopback mode, the configure_cmac bit can be set to `false` as the AIR process will initialize both MACs.

Once the system configuration is complete, start the test by running `sudo ./test.elf` on the proper machine. You will see this process perform AIR initialization, as well as configure the ERNIC and MAC. It will then report `[INFO] Polling on SEND 1 when receiving buffer` which means the source buffer is remote and it is polling on the metadata to be able to access that memory remotely. At this point, run `sudo ./driver1` this will initialize the other ERNIC (and MAC depending on the configuration) which will result in the computation completing in the `test.elf` process. You can then press `[Enter]` in the `driver1` process to complete the process and print statistics from the ERNIC.

## Coverage

| Coverage | How |
| -------- | --- |

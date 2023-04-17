# Test 301_mb_add_one_rdma_two_sided
## How to run the test

This test is utilized to excercise the two-sided RDMA functionality of the AIR Scale Out platform. It performs the same computation as shown in 13_mb_add_one, but it reads the source vector and writes the destination vector to a remote node using two sided send and receive operations. In this test, the process defined in test.cpp is running the AIR kernel, and it communicates over the network to the process defined in driver1.c, which bypasses AIR and directly communicates with one of the ERNICs in the AIR Scale Out platform.

This test can be run in two configurations:

1. Standard mode: in which a QSFP cable is connecting the two cards. As the platform currently does not support dynamic detection of QSFP cages, in this test QSFP cage 1 on one machine should be connected to QSFP cage 2 on the other machine. Then, driver1 should be run on the machine with QSFP cage 1 connected, and test.elf should run on the machine with QSFP cage 2 connected.
2. Loopback mode, in which a QSFP cable is loopbacked between the two cage.

When running in standard mode, driver1.c should have its configure_cmac bit set to `true` as it must configure the MAC in the platform. When the test is running in loopback mode, the configure_cmac bit can be set to `false` as the AIR process will initialize both MACs.

Once the system configuration is complete, start the test by running `sudo ./test.elf` on the proper machine. You will see this process perform AIR initialization, as well as configure the ERNIC and MAC. It will then report `[INFO] Polling on RECV for src_tensor` which means it is polling on a blocking receive. At this point, run `sudo ./driver1` this will initialize the other ERNIC (and MAC depending on the configuration) which will result in the computation completing in the `test.elf` process. Both processes should print out the source and destination vector and whether the test passed or failed.

## Coverage

| Coverage | How |
| -------- | --- |

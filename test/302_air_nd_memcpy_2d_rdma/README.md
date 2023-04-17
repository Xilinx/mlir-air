# Test 302_air_nd_memcpy_2d_rdma
## How to run the test

This test is utilized to excercise the two-sided RDMA functionality of the AIR Scale Out platform in an AIR level test. It performs the same datamovement as shown in 21_air_nd_memcpy_2d, but the source and destination vectors can be stored on a remote node and accessed through RDMA READS and WRITES. In this test, the process defined in test.cpp is running the AIR kernel, and it communicates over the network to the process defined in driver1.c, which bypasses AIR and directly communicates with one of the ERNICs in the AIR Scale Out platform.

This test can be run in two configurations:

1. Standard mode: in which a QSFP cable is connecting the two cards. As the platform currently does not support dynamic detection of QSFP cages, in this test QSFP cage 1 on one machine should be connected to QSFP cage 2 on the other machine. Then, driver1 should be run on the machine with QSFP cage 1 connected, and test.elf should run on the machine with QSFP cage 2 connected.
2. Loopback mode, in which a QSFP cable is loopbacked between the two cage.

When running in standard mode, driver1.c should have its configure_cmac bit set to `true` as it must configure the MAC in the platform. When the test is running in loopback mode, the configure_cmac bit can be set to `false` as the AIR process will initialize both MACs.

When running this test, a `driver_mode` is passed in as a command line argument to both `test.elf` and `driver1` to define where data is stored in the system. The usage statement is as follows:
```
USAGE: ./herd.exe {mode}
        0: Remote src buffer
        1: Remote dst buffer
        2: Remote src and dst buffer
        3: Local src and dst buffer
```

Once the system configuration is complete, start the test by running `sudo ./test.elf <driver_mode>` on the proper machine. You will see this process perform AIR initialization, as well as configure the ERNIC and MAC. Then, if `driver1` is used (not the case if `driver_mode` is 3), when run `sudo ./driver1 <driver_mode>`. This will initialize the other ERNIC (and MAC depending on the configuration) which will result in the computation completing in the `test.elf` process. Both processes should print out the source and destination vector and whether the test passed or failed.

## Coverage

| Coverage | How |
| -------- | --- |

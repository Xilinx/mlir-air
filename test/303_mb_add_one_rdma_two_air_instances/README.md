# Test 303_mb_add_one_rdma_two_air_instances
## How to run the test

This test excercises the one-sided RDMA functionality of the AIR Scale out platform with two AIR instances. This test performs the same computation as shown in 13_mb_add_one, but the source vector is located on host1 (host1_test.cpp) and the destination vector is located on host0 (host0_test.cpp). In this test, host0 performs a remote RDMA read to read the source vector, performs the add one computation on the first half of the data, and writes the first half of the data back to the destination buffer, which is stored locally. Meanwhile, host1 performs the add one comuptation on the second half of the data, and performs an RDMA write . Both processes use air_barrier() to synchronize when the source data is ready to be ingested and when the computations are complete.

As this test uses two AIR instances, it must be run in Standard mode which we have been defining as follows:

1. Standard mode: in which a QSFP cable is connecting the two cards. As the platform currently does not support dynamic detection of QSFP cages, in this test QSFP cage 1 on one machine should be connected to QSFP cage 2 on the other machine. Then, driver1 should be run on the machine with QSFP cage 1 connected, and test.elf should run on the machine with QSFP cage 2 connected.

When the system configuration is complete, run `sudo ./host0_test.elf` to run host0. This will poll on the reception of the metadata of the source vector. Then, on the other machine, run `sudo ./host1_test.elf`. Both tests should then proceed, synchonizing through barriers. host0__test.elf will check the entire destination buffer, as the buffer is located on host0. host1_test.elf will check its temporary destination buffer, which just contains the second half of the buffer. Both tests will print PASS or FAIL depending on the result.

## Coverage

| Coverage | How |
| -------- | --- |

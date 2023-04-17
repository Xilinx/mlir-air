# Test 304_mb_add_one_rdma_two_sided_two_air_instances
## How to run the test

This test excercises the two-sided RDMA functionality of the AIR Scale out platform with two AIR instances. This test performs the same computation as shown in 13_mb_add_one, where the source and destination vector are both located on a single device (host0), but the computation is distributed through RDMA two-sided communication across two AIR instances. In this test, host0 initializes the source and destination buffers, and performs an RDMA send to send the second half of data to host1. Both devices perform their add one kernels in parallel, then host1 performs an RDMA send to send the second half of the destination buffer to host0. 

As this test uses two AIR instances, it must be run in Standard mode which we have been defining as follows:

1. Standard mode: in which a QSFP cable is connecting the two cards. As the platform currently does not support dynamic detection of QSFP cages, in this test QSFP cage 1 on one machine should be connected to QSFP cage 2 on the other machine. Then, driver1 should be run on the machine with QSFP cage 1 connected, and test.elf should run on the machine with QSFP cage 2 connected.

When the system configuration is complete, run `sudo ./host1_test.elf` to run host1. This will poll on the reception of the second half of the source vector. Then, on the other machine, run `sudo ./host0_test.elf`. Both tests should then proceed, synchonizing through barriers. host0__test.elf will check the entire destination buffer, as the buffer is located on host0. host1_test.elf will check its temporary destination buffer, which just contains the second half of the buffer. Both tests will print PASS or FAIL depending on the result.

#

## Coverage

| Coverage | How |
| -------- | --- |

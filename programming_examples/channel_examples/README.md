# Channel Examples

This collection of examples focuses on one of the key abstractions of air: *channels*. The patterns shown here may be used to create more complex examples.

## Running and Testing

#### ```herd-to-herd```: Using a channel to pass data between herd.

There are two part of this example: two herds within one segment (single segment), and one herd per segment for two segments (multi-segment).

The single segment example example ([herd_to_herd/single_segment/herd_to_herd.py](herd_to_herd/single_segment/herd_to_herd.py)) defines two *herds* within the same *launch* and *segment*. There is a *producer herd*, which writes data to a `Herd2Herd` channel, and a *consumer herd*, which reads data form the `Herd2Herd` channel.

The multi-segment example ([herd_to_herd/multi_segment/herd_to_herd.py](herd_to_herd/multi_segment/herd_to_herd.py)) defines two `segment`s, each with one `herd`, within the same `launch`. There is a *producer_segment* with a *producer herd*, which writes data to a `Herd2Herd` channel, and a *consumer_segment* with a *consumer herd*, which reads data form the `Herd2Herd` channel.

Warning: The multi-segment example is a work in progress!

#### ```channel-size```: Use the channel size argument

This example ([channel_size/channel_size.py](channel_size/channel_size.py)) is a data passthrough example using the same tiling structure as the [matrix_scalar_add/multi_core_channel](../matrix_scalar_add/multi_core_channel.py) examples, only instead of using a separately defined channel for each tile/core, a bundle of channels is created (using the `ChannelOp` `size` parameter) and indexed into (the `ChannelGet` and `ChannelPut` `indices` parameter).

#### ```hierarchical```: Use channels for sending data from Launch to Segment to Herd and back again

This example ([hierarchical/hierarchical.py](hierarchical/hierarchical.py)) is a data passthrough example that uses a channel to send data from Launch to Segment (L3->L2 memory) and then from Segment to Herd (L2->L1 memory). The data is then sent back on an analogous path.

#### WIP: ```worker-to-self```:

This example ([worker_to_self/worker_to_self.py](worker_to_self/worker_to_self.py)) is a work-in-progress data passthrough example using the same tiling structure as the [matrix_scalar_add/multi_core_channel](../matrix_scalar_add/multi_core_channel.py) examples, only the sole worker in the herd does some extra shuffling between input and output by putting the current data tile into a channel and then getting it from the same channel.

WARNING: This example currently fails for unknown reasons.

#### WIP: ```worker-to-worker```:

This example ([worker_to_worker/worker_to_worker.py](worker_to_worker/worker_to_worker.py)) is a work-in-progress data passthrough example using the same tiling structure as the [matrix_scalar_add/multi_core_channel](../matrix_scalar_add/multi_core_channel.py) examples, only the each worker trades a tile of input data to another worker in the herd by sending it via channel.

WARNING: This example currently fails for unknown reasons.

#### Usage (For All Examples)

To generate AIR MLIR from Python:
```bash
cd <example_dir>
make clean && make print
```

To run:
```bash
cd <example_dir>
make clean && make
```

To run with verbose output:
```bash
cd <example_dir>
python <example_file>.py -v
```

You may be able to configure examples (data types, sizes); to get additional usage information, run:
```bash
cd <example_dir>
python <example_file>.py -h
```

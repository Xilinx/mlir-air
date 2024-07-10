# Channel Examples

This example focuses on one of the key abstractions of air: *channels*. This is a collection of examples that use channels in various ways. The patterns shown here may be used to create more complex examples.

## Running and Testing

#### ```herd-to-herd```: Using a channel to pass data between herds

This example ([herd_to_herd/herd_to_herd.py](herd_to_herd/herd_to_herd.py)) defines two `herd`s within the same `launch` + `segment`. There is a *producer herd*, which writes data to a `Herd2Herd` channel, and a *consumer herd*, which reads data form the `Herd2Herd` channel.

```bash
cd herd_to_herd
make clean && make
```

#### ```channel-size```: Use the channel size argument

This example ([channel_size/channel_size.py](channel_size/channel_size.py)) is a data passthrough example using the same tiling structure as the [matrix_scalar_add/multi_core_channel](../matrix_scalar_add/multi_core_channel.py) examples, only instead of using a separately defined channel for each tile/core, a bundle of channels is created (using the `ChannelOp` `size` parameter) and indexed into (the `ChannelGet` and `ChannelPut` `indices` parameter).

```bash
cd channel_size
make clean && make
```

#### WIP: ```worker-to-self```:

This example ([worker_to_self/worker_to_self.py](worker_to_self/worker_to_self.py)) is a work-in-progress data passthrough example using the same tiling structure as the [matrix_scalar_add/multi_core_channel](../matrix_scalar_add/multi_core_channel.py) examples, only the sole worker in the herd does some extra shuffling between input and output by putting the current data tile into a channel and then getting it from the same channel.

WARNING: This example currently fails because it is assumed channel gets/parts are not from the same memory region, and this example breaks this assumption.

```bash
cd worker_to_self
make clean && make
```

#### WIP: ```worker-to-worker```:

This example ([worker_to_worker/worker_to_worker.py](worker_to_worker/worker_to_worker.py)) is a work-in-progress data passthrough example using the same tiling structure as the [matrix_scalar_add/multi_core_channel](../matrix_scalar_add/multi_core_channel.py) examples, only the each worker trades a tile of input data to another worker in the herd by sending it via channel.

WARNING: This example currently fails for unknown reasons.

```bash
cd worker_to_worker
make clean && make
``

#### WIP: more examples!
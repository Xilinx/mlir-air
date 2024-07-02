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

This example ([channel_size/herd_to_herd.py](channel_size/herd_to_herd.py)) keeps the same structure as the `herd-to-herd` example but uses `CHANNEL_SIZE` channels to communicate partial blocks of the input image between the producer and the consumer herds. The `ChannelOp` `size` argument is a convenience method for creating multiple channels; the `ChannelPut`/`ChannelGet` `indices` argument allows you to access a specific channel in that grouping.

```bash
cd channel_size
make clean && make
```

#### WIP: more examples!
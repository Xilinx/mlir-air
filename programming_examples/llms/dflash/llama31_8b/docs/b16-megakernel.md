# Llama-3.1-8B B16 decode megakernel

`BLOCK_SIZE=16` uses an anchor and fifteen proposals. The official UltraChat
drafter was trained at block size 10; this is an explicit execution extension.
B8 remains the default. Both configurations share checkpoints, packed weights,
NPU prefill, the BF16/BFP context-preparation kernel and the host loop. A
sixteen-row context update uses two eight-row prepass dispatches.

The 32-layer target and five-layer drafter reuse the
[4096-wide B16 mapping](../../common/docs/b16-ddr-replay.md): four residual
owners, DMA capture/replay of normalized rows through DDR, a 2×2 projection
schedule and two attention waves. Llama retains its target/drafter FFN widths,
role-specific RMS epsilon, Llama3 RoPE, feature taps, untied vocabulary head
and EOS IDs. No additional model-specific AIR graph is introduced.

```sh
make compile BLOCK_SIZE=16
make run BLOCK_SIZE=16
make chat BLOCK_SIZE=16
make verify BLOCK_SIZE=16
make profile BLOCK_SIZE=16
make compile-baseline BLOCK_SIZE=16
make full20 BLOCK_SIZE=16
make full80 BLOCK_SIZE=16  # optional
```

B16 artifacts live in `build_b16/`, and both target and drafter must select the
same block size. C4096's last full-block prefix is 4080. Compile, verify,
profile and Full20 LIT entries follow the existing nightly filters, with B16
artifact, profile-log and verification identities. Full80 remains opt-in.
The default native B1 comparator retains its own arithmetic and epsilon;
exact loop checks use the sequential oracle with matching B16 arithmetic.
See [validation](validation.md) and [compatibility](compatibility.md).

The shared design document explains why the source mapping requires compiler
fixes for mixed DMA feeds and mixed-memory packet producers, and how it
replaces the historical generated-MLIR-AIE repairs. Build/install AIR with
those fixes, using the repository's normal dependency versions.

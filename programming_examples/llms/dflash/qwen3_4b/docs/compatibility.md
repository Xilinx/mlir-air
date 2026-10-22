# Upstream compatibility

The B16 addition is based on upstream MLIR-AIR commit
`3a6e98644aa9f2e5c8f7403b530d3aabadbe3818` (October 22, 2026), which already
includes all three B8 model examples and their compiler prerequisites. B16
uses the normal AIR environment and adds no compiler changes. Both block
sizes use the same pinned checkpoints, packed caches and Python dependencies.

## Dependency versions for this integration

| Dependency | Version |
|---|---|
| MLIR-AIE | `1.4.5.dev51+ge040370` |
| Peano / llvm-aie | `22.0.0.2026101201+83ba9ea9` |
| Host LLVM/MLIR | `24.0.0.2026100106+bdf04ebb` |
| `llvm-link` for inline attention | LLVM 21; upstream fused_decode requires LLVM <23 |
| Device | NPU2, eight-column partition, Linux XRT 2.21.0 |
| Host | Python 3.12; a C++17 `g++` for upstream `TxnBuilder` |

The extra LLVM linker follows the upstream `fused_decode` preflight and CI
setup. Peano still compiles the device kernels. It is not a replacement compiler
for this example. Check `make -C programming_examples/fused_decode preflight-llvm-link`
when setting up the environment.

`make` exports the existing `AIR_TARGET_DEVICE=npu2` option so shared prefill
builders also work when the compilation host cannot query the device. The decoder
builder explicitly selects `npu2`. Decoder builds use XRT's xclbin and generated
transaction-header flow; the standalone prepass uses the multi-launch ELF flow.
Both are supported upstream APIs.

## Compiler prerequisites already merged upstream

The original B8 integration at `4bf1d78f` used MLIR-AIE dev46 and introduced
the following fixes. They are part of the current upstream compiler; they are
not additional changes in the B16 PR.

| Change | Reason | Regression coverage |
|---|---|---|
| Acquire a shared L1 send-buffer lock before its producer accesses | Acquiring immediately before `channel.put` can let an opaque producer overwrite data still being read by the previous DMA. This caused nonrepeatable target outputs. | `air_channel_to_locks_opaque_producers.mlir`, alongside the original shared-buffer test |
| Honor `air.disable_alloc_sink` on allocations / enclosing `air.execute` | Keep designated scratch storage outside the reduction loop so scratch does not prevent projection-weight ping-pong transformation. | `disable_alloc_sink.mlir` |
| Preserve MemTile ownership during `index_switch` arm-buffer rewriting | Core relay copies and compute-tile channel pins do not apply to L2 forwarding; buffer unification must retain compatible allocation attributes, including MemTile placement. The clean build exposed invalid channel IDs and mixed packet/circuit routes. | `air_channel_index_switch_memtile.mlir`, plus the original core relay/dedication tests |

That integration passed 647 MLIR tests, with 7 expected failures and 7
unsupported tests, and its DFlash compile LIT. Those historical receipts and
performance results remain in [validation.md](validation.md). B16's current
validation is recorded in [b16-megakernel.md](b16-megakernel.md).

The B16 base also includes the NPU-only `aircc` cleanup (#2156). The compiler
is rebuilt from that base before producing B16 artifacts. The shared fused
prefill change in #2158 does not affect this example: its adapter calls
`qwen3_4b_q4nx_prefill.Qwen3Q4nxPrefill`, which uses the native Qwen3 prefill
builders rather than `llms/shared/fused_prefill`.

Other historical DFlash-branch compiler patches are not prerequisites here:
upstream already preserves the constant bias in dynamic DMA offsets. B8 keeps
its existing lock selection; B16 uses upstream automatic lock selection,
without the historical B16 chain-lock annotation. Current builds need no
refeed routing patch or mixed-result branch-tracing patch. No generated AIE IR,
BD assignments or routes are edited after compilation.

## Shared kernel changes

Two changes live under `programming_examples/fused_decode/kernels/`:

- The exponential LUT explicitly masks fractional index bits before its integer
  lookup. This preserves the interval under nearest-even rounding, which B8 uses.
- `ATTN_NORM_VECTOR` enables an eight-lane attention output normalization
  path. Existing callers retain their default path unless they set this macro.

These are device-kernel changes, separate from the AIR compiler fixes above.

## Shared weight-loader change

The existing Qwen3-4B Q4NX prefill loader now converts its tied embedding/LM-head
tensor to BF16 once, before loading decoder layers. Previously it retained an
FP32 embedding throughout layer loading and returned two separate BF16 copies.
The change preserves every weight value and makes the returned embedding and
LM head share storage, as required by this model's tied-weight configuration.
It reduces initialization memory without changing the compiler or device code.

The DFlash prefill adapter additionally shares the already-uploaded MLP weight
BOs before the upstream loader allocates its LM-head partitions. It checks exact
weight equality before replacing host arrays with read-only BO views. Moving
this existing sharing step earlier releases about 5 GiB of duplicate host
storage before the next allocation phase; model arithmetic is unchanged.
Decoder initialization reads the bundle's BF16 embedding directly and clears
its mapped input/output/KV BOs in place. This avoids an intermediate FP32
embedding and a second capacity-sized zero array while prefill is resident.

## Weights and arithmetic

Target and draft decoder projections use group-32 affine Q4 packing and BF16
activations with BFP16 matrix products and FP32 accumulators. Context preparation
uses the official drafter's BF16 weights with the same BFP matrix primitive.

The target packer intentionally retains the selected study's sequence:
Q4NX bundle → BF16 dequantization → Q4 repacking. Current upstream B1 can pack
raw Q4 bundle values directly. Keep those caches separate; identical model names
do not imply identical packed arithmetic. This example uses its own cache
namespace and records checkpoint revisions and cache hashes.

## Rebuilds and evidence

Each decoder/prepass manifest records its configuration, compiler binary,
package versions, source hashes and artifact hashes. `make compile` verifies
existing artifacts and rejects stale builds. Use a fresh `BUILD_DIR` after
changing the compiler, sources or dependency versions. For prefill, also use a
fresh directory: its shared cache is not keyed by all toolchain inputs.

Historical performance is labeled separately from this port's new-toolchain
validation. See [validation.md](validation.md) for the executed coverage.

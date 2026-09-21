#!/usr/bin/env python3
"""Emit a megakernel decode chain as AIR MLIR.

Qwen3-0.6B has 28 layers of nine stages each. Nobody writes that by hand, and
the hand-written tests next to this one stop being a plan the moment the layer
count stops being 2. Fleet has the same problem and solves it the same way: its
task graph is described in Python and `src/kernel/runtime.cc` prints the C++.
This prints MLIR instead.

The layer is shaped like the one Fleet builds for Qwen3
(python/mirage/mpk/models/qwen3/builder.py):

    r    = rmsnorm(x) * n1                 0  one task, a reduction
    qkv  = r @ Wqkv                        1  q, k and v in one matmul
    q,k  = rope(rmsnorm_head(q,k)), and    2  one task per (token, head)
           k,v appended to the KV cache
    a    = softmax(q.K/sqrt(hd)) @ V       3  one task per (token, head)
    ao   = a @ Wo                          4
    xa   = rmsnorm(x + ao) * n2            5  one task, a reduction
    gu   = xa @ Wgu                        6  gate and up fused, as Fleet does
    act  = silu(gu_g) * gu_u               7  SwiGLU
    x    = xa + act @ Wd                   8  residual folded in, as Fleet does

Attention is grouped-query: `heads` query heads share `kv-heads` key/value
heads, and the per-head norms on q and k are the ones Qwen3 has and Qwen2 does
not (kv_cache_update_mi300.cuh:130). RoPE is Fleet's rotate-half
(tasks/ampere/norm.cuh:101-118).

Every stage boundary is an event, and the whole chain is one launch. The stage
structure is what scales: adding a layer adds queue slots and event slots, not
a kernel launch.

With --tokens M > 1 the chain runs M tokens at once, which is what a prefill or
a speculative window looks like. The M tokens append their own k and v to the
cache and attend the prefix plus whichever window entries precede them, so
stage 2 is a real cross-token dependency: token 3's scores read what token 0
wrote. Every linear stage becomes an (m, n) grid rather than a row of n, and
the traversal of that grid is Fleet's -- see strided_stage. At M = 1 the
decomposition collapses to the index arithmetic this generator used before
--tokens existed.

    ./gen.py --layers 4 --dim 128 > chain.mlir
    ./gen.py --layers 2 --dim 128 --tokens 4 > prefill.mlir

Running the real model
----------------------

Everything below was measured on one MI350X (gfx950) against a real
Qwen3-0.6B checkpoint, six decode steps, 28 layers.

    QWEN_DIR=/path/to/qwen3-0.6b TASKS=128 WORKERS=128 WAVES=8 STEPS=6 \\
        run_qwen.sh

That is the configuration that runs at **2.94 ms a token**, against Fleet's
mirage_mpk at 2.421 measured on the same node and a memory-bandwidth floor of
0.14. It prints the tokens, the host reference's tokens, and an independent
numpy Qwen3's, and says PASS only when all three agree.

Nothing here needs a flag to be fast; the defaults are the fast ones. The
flags that exist are the slow forms, kept because each rests on an assumption
worth being able to withdraw:

    --acquire-per-wave  every wave takes the post-rendezvous acquire fence,
                        rather than the one wave that waited taking it before
                        the barrier.                        1.48x slower.
    --dynamic-claim     hand pieces out from per-chiplet queues rather than
                        partitioning them by chiplet rank.  1.60x slower.
    --weights-reduction-major
                        the bf16 weights as [reduction][output], so a lane
                        walking its reduction strides by the whole width.
                        5.7% slower.
    --half-dim-tasks    o_proj and down on half the workgroups in pieces
                        twice as wide.                      6.0% slower.
    --lds-klanes        a column's lane partials meet entirely in LDS rather
                        than folding by shuffle first.      3.2% slower.
    --round-robin-claim chiplet d takes the pieces congruent to d rather than
                        a contiguous block of them.         3.2% slower.
    --split-arrival     a die's piece count and arrival count in two words
                        rather than the halves of one.      1.027x slower.
    --reduce-unroll 1   one weight load in flight per lane at a time.
                        1.61x slower at 1, 17.5% at 4, 48% at 16; 8 is the
                        default and is a real minimum, not a ceiling.
    --spin-sleep 0      poll the event word as tight as the hardware will
                        run; 16 is the default and worth 0.8%.

`--fuse-swiglu` is the one flag that is off without being slower on purpose:
it removes a stage and is 0.63% slower anyway, for a reason worth reading
before trying it again -- see the note next to `slice_i`.

Then the flags that only exist to measure -- `--timers`,
`--timers-total-only`, `--pad-stages`, `--pad-strip`. See "Measuring it".

Where the 2.94 ms goes
----------------------

A stage boundary -- claim, signal, rendezvous, acquire -- costs 5.04 us
whatever the stage computes, and a decode step has 257 of them, so 1.30 ms is
boundary. That is 44% of the launch, up from a third, because everything
around it got faster and it did not: re-priced on this build it came in
within 2% of what it was two builds and 0.5 ms/token ago, which is what a
cost independent of the body should do.

Priced by `--pad-stages` and `--pad-strip`, which add empty stages and then
remove one piece of the boundary at a time from them. The slope is linear --
one pad a layer gives 5.09 us and three give 5.04:

    the spin and its barrier      2.40 us      48%
    the atomics                   1.71 us      34%
    the acquire fence             0.79 us      16%
    the claim                     0.14 us       3%

Both of the changes that made the boundary 12.28 us into 5.15 came out of that
ladder. The acquire fence was 7.42 us of the 12.28 until it stopped being
taken once per wave -- 1.48x on the whole model -- and signalling was three
memory operations until a die's piece count and arrival count moved into the
two halves of one word, which was 2.7%.

What is left is half rendezvous, and the rendezvous is latency rather than
congestion -- backing the poll off by a factor of sixteen is worth 0.8%. So
it would be spent by having fewer stages, except that the one fusion
available does not pay. `--fuse-swiglu` deletes 28 of the 257 boundaries,
which the ladder says is 0.14 ms or 4.9%, and the whole model still comes out
1.3% slower -- so the fused gate_up body costs about 6% more than gate_up and
swiglu separately. The explanation on offer used to be cache-line coverage,
that a fused piece reads two 24-column weight strips where the split stage
read one of 48; the layout change removed coverage as a mechanism entirely
and the fusion is still slower, so that explanation is gone and there is no
replacement.

What is left of the body is the four matmuls, and what sets their speed is
neither the traffic nor the arithmetic. Achieved bandwidth across the five
matmul classes is ordered by one thing: the length of the dependent chain
that combines the partials of the lanes sharing an output column, divided by
the weights a lane loads between two of them. Three other explanations were
built and measured first and none of them ordered these five:

  * cross-XCD cache-line duplication got the order backwards. --blocked-claim
    is 3.2% and it is alignment, not duplication -- gate_up's 48-column piece
    is the one width here that does not divide a 128-byte line.
  * cache-line coverage got the order right and the size wrong by seven
    times. Flipping the weights to [output][reduction] is 5.7%.
  * bytes in flight was already satisfied. The emitted gfx950 assembly has
    206 `global_load_dwordx4` and one `global_load_ushort`; writing the
    vector load out by hand gives identical assembly and a clock 0.17%
    apart, which is noise.

And the redundant activation reads, which looked like more traffic than the
weights, are all L2 hits -- staging them in LDS is 14.7% slower. See
--stage-lhs.

Measuring it
------------

Three instruments, and only the first two are worth a number:

  * `TIMERS=1 REPEAT=20` -- the chain reads the 100 MHz clock around the whole
    worker body and prints "WHOLE LAUNCH". The counters are zeroed per launch,
    so REPEAT=20 reports a warm one. Reproducible to about 0.2%, and it is
    what every figure above was taken with.
  * `TIMERS=1` alone also prints a per-operator table. Its windows are chained
    -- each stage ends on the read the next one starts from -- so the classes
    sum to the launch (99.9%, measured) instead of to 59% of it, which is what
    two independent reads a stage gave.
  * `workspace/bench.sh` times the process, so it is the only end-to-end
    check, but the device is a small part of its wall clock. It needs
    `HI=4000` to resolve anything; at its old default of 100 it is +/- 1.7
    ms/token and has mis-ranked five changes.

Two things to know before believing a number. A timing from a run that did not
print PASS is not a measurement. And the launch clock is bimodal: about a
third of launches come in 35% slow, all of it inside `gate_up`'s body and
undiagnosed, so take the minimum of repeated runs rather than the mean.
"""

import argparse
import math as _math
import sys

# Every weight is built from `index % modulus`, and every one of them uses a
# different modulus. Sharing one makes the matmuls resonate: a weight matrix
# W[i, j] = f(i - j) is a convolution, and if the activation it multiplies has
# the same period, the products add coherently and the gain is ~dim instead of
# ~sqrt(dim). Measured, with a period-5 activation: a period-5 weight amplifies
# 2.84x what a random walk would, every other period 0.2x. The input uses an
# eleventh, coprime with all of them.
MOD_QKV, MOD_O, MOD_G, MOD_U, MOD_D, MOD_K, MOD_V = 3, 5, 7, 13, 3, 7, 5
LN10000 = 9.2103403719761836


def _scale(mod, red, gain=1.0):
    """Weight scale: unit variance, then 1/sqrt(fan-in).

    `x % mod` has standard deviation sqrt((mod^2-1)/12), and a matmul that
    reduces over `red` terms multiplies the input's scale by sqrt(red). Without
    the second factor the gate pre-activation comes out at ~4 instead of ~1,
    silu(g)*u lands around g*u, and the result grows by two orders of magnitude
    a layer. This is the same reason real initialisers divide by fan-in.
    """
    return gain / _math.sqrt(red) / _math.sqrt((mod * mod - 1) / 12.0)


def emit(
    layers: int,
    dim: int,
    tasks: int,
    workers: int,
    repeat: int = 1,
    cache: int = 32,
    tokens: int = 1,
    inter: int = 0,
    heads: int = 4,
    kv_heads: int = 2,
    steps: int = 1,
    vocab: int = 256,
    head_dim: int = 0,
    rope_theta: float = 10000.0,
    W=None,
    prompt=None,
    prompt_len: int = 0,
    wave: int = 64,
    waves: int = 1,
    nt_weights: bool = False,
    timers: bool = False,
    dies: int = 8,
    unroll: int = 8,
    stage_timers: bool = True,
    static_claim: bool = True,
    pad_stages: int = 0,
    pad_strip: int = 0,
    acquire_once: bool = True,
    spin_sleep: int = 16,
    pack_arrival: bool = True,
    fuse_swiglu: bool = False,
    stage_lhs: bool = False,
    acquire_agent: bool = False,
    blocked_claim: bool = True,
    out_major: bool = True,
    fold_klanes: bool = True,
    full_dim_tasks: bool = True,
    count_flushes: bool = False,
) -> str:
    inter = inter or 2 * dim
    assert heads % kv_heads == 0, "heads must be a multiple of kv-heads"
    # The wave is where the parallelism inside a task lives: a task body splits
    # its outermost loop across the lanes and closes any reduction with a
    # butterfly over them. Both need the same number, and it has to be the one
    # the lowering picks -- `-air-to-rocdl{wave-size=N}`, default 64
    # (GPUPasses.td:40), times the herd's x extent, which this file emits as 1.
    #
    # Disagreeing is not silent. Too small and the strided loops leave the tail
    # of every row unwritten; too large and each wave reduces only its own
    # lanes and reads a claim no one broadcast to it. Either way the numbers
    # come out wrong and the host comparison in this same program says so.
    assert wave & (wave - 1) == 0 and wave > 1, "wave size must be a power of two"
    wave_steps = wave.bit_length() - 1
    # Wavefronts per workgroup, i.e. the herd's x extent. One wave per
    # workgroup leaves the machine almost empty at decode: MI300X holds 6080
    # waves and a stage's column count caps how many a column split can use --
    # 1024 outputs is 16 waves however it is divided, which is why Fleet splits
    # K as well ("ALL 30 workers per XCD are active vs N-split which has only 8
    # tiles/XCD for O_proj", gang_ksplit_linear_mi300.cuh:8). Here the split is
    # within the workgroup: waves divide the reduction, lanes divide the
    # columns, and the wave partials meet in LDS.
    assert waves & (waves - 1) == 0 and waves >= 1, "waves must be a power of two"
    nthreads = waves * wave
    # The scratch the waves reduce a matmul column through. The fused
    # gate_up reduces a gate column and its matching up column in the same
    # pass, so it needs two partials a thread rather than one; 2 KB more
    # LDS out of 64, and nothing else in here cares how wide it is.
    redwords = nthreads
    RT = f"memref<{redwords}xf32, 3>"
    # Scratch for the reduction operand, staged once per workgroup.
    #
    # Every lane of a matmul reduces over the whole of its input, so a column
    # group of `cols` lanes loads the same activation `cols` times -- 16 or 32
    # here. Counted over a layer that is more traffic than the weights are:
    # 54.5 MB against 31.5, and half of every lane's loads. Fleet's matmuls
    # stage both operands, so reading it once per workgroup instead looked
    # like the thing AIR was missing.
    #
    # **It is 14.7% slower**, and that is the useful part. The activation is
    # 4 to 12 KB and is read by every workgroup, so it lives in L2 and every
    # one of those redundant loads was already a hit; replacing a cached load
    # with an LDS read and two barriers a piece costs more than it saves. The
    # traffic argument is dead -- what makes Fleet's staging pay is not the
    # traffic but what it lets Fleet do with the operand once it is there.
    #
    # It is kept because the SwiGLU fusion needs it: staged, the SwiGLU is
    # applied once per element on the way into LDS rather than once per
    # output column, the emitted math.exp count goes back to the split
    # build's exactly (255 -> 59), and the fusion is 3.4% faster on top of
    # staging where it was 34.7% slower without it. The fusion mechanism
    # works; it is sitting on a foundation that does not pay yet.
    lhswords = max(2 * dim, inter, heads * (head_dim or dim // heads))
    lhswords = lhswords if stage_lhs else 0
    LT = f"memref<{lhswords}xf32, 3>"
    # 16 waves -- a 1024-thread workgroup -- does not finish. A four-layer
    # config that takes half a minute at 8 was still running after four, on
    # hardware that has the registers for it (92 VGPRs, so 20 wave slots a CU
    # against the 16 a block would need). Not understood, so it is refused
    # rather than left to burn an allocation discovering it again.
    assert nthreads <= 512, (
        f"waves={waves} gives a {nthreads}-thread workgroup; anything past 512 "
        "hangs and the reason is not yet known"
    )
    lds_globals = (
        ""
        if waves == 1
        else (
            f'  memref.global "private" @air_bcast : memref<4xi32, 3>\n'
            f'  memref.global "private" @air_red : {RT}'
            + (f'\n  memref.global "private" @air_lhs : {LT}' if stage_lhs else "")
        )
    )
    lds_handles = (
        ""
        if waves == 1
        else (
            "        %bcast = memref.get_global @air_bcast : memref<4xi32, 3>\n"
            f"        %ldsr = memref.get_global @air_red : {RT}"
            + (
                f"\n        %ldsl = memref.get_global @air_lhs : {LT}"
                if stage_lhs
                else ""
            )
        )
    )

    def bcast_i32(src, dst, tag, indent):
        """Give every thread of the workgroup the value thread 0 computed.

        At one wave per workgroup the wave *is* the workgroup and
        rocdl.readfirstlane reaches all of it. Past that it does not, and the
        value has to go through LDS -- which is the real reason a wider herd
        is not free, and the reason it was left at 1x1 until now.
        """
        pad = " " * indent
        if waves == 1:
            return f"{pad}{dst} = rocdl.readfirstlane {src} : i32"
        return (
            f"{pad}gpu.barrier\n"
            f"{pad}scf.if %isLead {{\n"
            f"{pad}  memref.store {src}, %bcast[%c0_s] : memref<4xi32, 3>\n"
            f"{pad}}}\n"
            f"{pad}gpu.barrier\n"
            f"{pad}{dst} = memref.load %bcast[%c0_s] : memref<4xi32, 3>"
        )

    def bcast_i32x2(src0, src1, dst0, dst1, tag, indent):
        """bcast_i32 for a pair, through one pair of barriers rather than two.

        The claim below is two numbers -- which queue the piece came from and
        which piece -- and they have to arrive together, because a workgroup
        that agreed on one and not the other would compute a piece nobody
        claimed. @air_bcast has four slots for exactly this.
        """
        pad = " " * indent
        if waves == 1:
            return (
                f"{pad}{dst0} = rocdl.readfirstlane {src0} : i32\n"
                f"{pad}{dst1} = rocdl.readfirstlane {src1} : i32"
            )
        return (
            f"{pad}gpu.barrier\n"
            f"{pad}scf.if %isLead {{\n"
            f"{pad}  memref.store {src0}, %bcast[%c0_s] : memref<4xi32, 3>\n"
            f"{pad}  memref.store {src1}, %bcast[%c1_s] : memref<4xi32, 3>\n"
            f"{pad}}}\n"
            f"{pad}gpu.barrier\n"
            f"{pad}{dst0} = memref.load %bcast[%c0_s] : memref<4xi32, 3>\n"
            f"{pad}{dst1} = memref.load %bcast[%c1_s] : memref<4xi32, 3>"
        )

    # Qwen3 states head_dim in its config and it is not hidden/heads: 0.6B has
    # hidden 1024, 16 heads and head_dim 128, so the q projection is wider than
    # the residual stream and o_proj is the thing that narrows it again.
    hd = head_dim or (dim // heads)
    qw = heads * hd  # width of the q projection and of o_proj's input
    h2 = hd // 2  # rope pairs
    group = heads // kv_heads  # query heads per kv head
    qkvo = (heads + 2 * kv_heads) * hd
    # The cache holds the prefix plus every window the run will append. Step s
    # writes slots [cache + s*tokens, cache + (s+1)*tokens) and attends
    # everything up to its own, so the attention length is a runtime value.
    total = cache + steps * tokens
    assert hd % 2 == 0, "head dim must be even for rope"
    for n, v in (
        ("dim", dim),
        ("inter", inter),
        ("2*inter", 2 * inter),
        ("qkv out", qkvo),
    ):
        assert v % tasks == 0, f"{n} ({v}) must divide evenly into tasks"
    # How many pieces a stage splits into is a property of the stage, not of
    # the model, and the measured sweep says so in three directions at once.
    # Bodies at 32/64/128/256 pieces, 28 layers, 6 steps:
    #
    #             32       64      128      256
    #   gate_up  761036  464392  246988   297740
    #   qkv      622300  332236  172216   206064
    #   down     416168  242288  281456   322164
    #   o_proj   283160  175468  184324   257108
    #   lm_head  600372  310400  168628   118664
    #
    # o_proj and down write a dim-wide output, so at 128 pieces a piece is
    # `dim/128` = 8 columns: the lanes spread across those 8, which is 16 bytes
    # of every 64-byte line, and the other 48 are fetched and thrown away. They
    # want the slice wider even at the price of half the workgroups. The lm
    # head has a whole vocabulary of columns and wants the opposite. gate_up
    # and qkv are already wide enough to fill a line at 128 and only want
    # workgroups.
    #
    # So: half as many pieces for the dim-wide matmuls, `tasks` for everything
    # else. Clamped to at least `dies`, because a stage with fewer pieces than
    # chiplets leaves a chiplet with none and the static claim needs every
    # chiplet to have work.
    #
    # The lm head row above says 256, and acting on it made the lm head 11%
    # *worse* (171 236 ticks to 190 700). The sweep moved `tasks` and `workers`
    # together, so what that row measured was 256 workgroups, not 256 pieces --
    # and the lm head is the one stage already running at about 1.17 TB/s, so
    # what it wants is more of the machine, not a finer division of the same
    # 128 workgroups' work. Two pieces of 594 columns instead of one of 1187 is
    # the same bytes and one more loop. Left at `tasks`, with the measurement
    # written down so the row is not read that way again.
    # o_proj and down are only `dim` wide, so splitting them `tasks` ways
    # leaves each piece 8 columns at 128 tasks. Halving the split doubles the
    # piece to 16 and idles half the workgroups, and that used to be the
    # better trade because under [reduction][output] the piece width WAS the
    # cache-line coverage: 8 columns is 16 bytes of a 128-byte line, 16 is 32.
    # Under [output][reduction] a lane walks a row and the piece width no
    # longer touches coverage at all, so the trade is now only workgroups
    # against per-lane work and it is worth asking again.
    tasks_d = max(dies, tasks if full_dim_tasks else tasks // 2)
    tasks_v = tasks
    for n, v in (("dim", dim), ("inter", inter)):
        assert v % tasks_d == 0, (
            f"{n} ({v}) must divide evenly into the {tasks_d} pieces the "
            "dim-wide matmuls split into"
        )
    slice_d = dim // tasks
    slice_dw = dim // tasks_d
    slice_i = inter // tasks
    slice_2i = (2 * inter) // tasks
    # The SwiGLU fusion, twice attempted and twice refused, and what it
    # actually points at.
    #
    # A stage boundary costs 5.15 us whatever the stage computes and swiglu
    # costs 6.6 us an instance, so nearly all of swiglu is the fact of being
    # a stage. Two ways to make it not one, both built and both measured
    # against the split form beside them in the same job:
    #
    #   into gate_up, the producer   0.63% slower
    #   into down, the consumer     34.7% slower
    #
    # The producer version gives a piece two 24-column weight strips where
    # the split stage read one of 48, and 24 bf16 is 48 bytes of a 64-byte
    # line. Interleaving the gate and up halves of Wgu so the pair is
    # adjacent was supposed to fix that; it was built across all five readers
    # of the matrix, it is correct, and it changed nothing (0.73% slower
    # still). Reverted.
    #
    # The consumer version is the one Fleet does -- silu_mul_linear computes
    # silu(gate)*up @ weight^T with gate = input[:, :K], up = input[:, K:]
    # (silu_mul_linear_mi300.cuh:39-40,114-115) -- and it leaves the weight
    # reads completely alone, so the strip argument cannot apply. It is worse
    # anyway, and this time the reason is exact: `down` reduces over `inter`
    # for each of `dim` output columns, and in this generator every lane
    # loads its own activation, so the SwiGLU is recomputed once per output
    # column. 3 072 transcendentals a layer become 3 145 728. Measured,
    # down's body goes 353 504 ticks to 809 024.
    #
    # Fleet does not pay that because its matmul stages the activation
    # through LDS and applies the SwiGLU once as it writes it there
    # (silu_mul_linear_mi300.cuh:34, 190-192). **That is the difference worth
    # having, and it is not the fusion.** Every lane here reads the whole
    # reduction out of global memory on its own; staging it once per
    # workgroup would cut the activation traffic by the number of columns a
    # lane group covers and make this fusion free rather than expensive.
    slice_q = qkvo // tasks
    # The vocabulary is the one width that does not have to divide: Qwen3's is
    # 151936 = 2^7 * 1187, so requiring it to capped `tasks` at 128 -- and the
    # measured sweep says the four layer matmuls are bound by how many
    # workgroups there are, nothing else (128/64/32 tasks give 8.87/12.72/21.52
    # ms/token). So the last piece is short and the three places that walk it
    # skip the columns past the end.
    slice_v = (vocab + tasks_v - 1) // tasks_v
    invsqrthd = hd**-0.5
    lnrope = _math.log(rope_theta)
    # Activations carry a token dimension; weights do not. That asymmetry is
    # the whole reason M > 1 changes the traversal: a weight block is worth
    # reading once and using `tokens` times.
    AT = f"memref<{tokens}x{dim}xf32>"
    QWT = f"memref<{tokens}x{qw}xf32>"
    IT = f"memref<{tokens}x{inter}xf32>"
    GT = f"memref<{tokens}x{2 * inter}xf32>"
    QT = f"memref<{tokens}x{qkvo}xf32>"
    SCT = f"memref<{tokens}x{heads}x{total}xf32>"

    # The weights arrive as bf16 -- that is the only dtype in a Qwen3
    # checkpoint -- and weights.py widens them to f32, which doubles the bytes
    # for no information at all. The device reads the bf16 copies below and
    # widens each value with a single shift; f32 stays on the host, where the
    # synthetic generator and its centring pass write to it.
    # The f32 arrivals keep the layout weights.py wrote, [reduction][output].
    # The bf16 copies the device actually reads may be either way round, and
    # which one they are is the whole of --weights-out-major.
    #
    # [reduction][output] puts a piece's columns next to each other, so a
    # wave's lanes coalesce across the columns -- but a lane walking its own
    # reduction strides by the full width, one 2-byte load per line, and a
    # wave covers only `cols * 2` bytes of every 128-byte line it touches.
    # That is 16 bytes for anything dim-wide at 128 tasks.
    #
    # [output][reduction] is what Fleet stores (linear_ck_mi300.cuh:406-408,
    # strides (REDUCTION_SIZE, 1)) and what this program already gives the lm
    # head alone -- which is the one class in it that runs above a terabyte a
    # second, 3.5x `down`, on the same dot_loop and the same unroll.
    def wtype(l, red, out, ty="bf16"):
        a, b = (out, red) if (out_major and ty == "bf16") else (red, out)
        return f"memref<{l}x{a}x{b}x{ty}>"

    WQT = wtype(layers, dim, qkvo, "f32")
    WQTB = wtype(layers, dim, qkvo)
    WT = wtype(layers, qw, dim, "f32")
    WTB = wtype(layers, qw, dim)
    WGT = wtype(layers, dim, 2 * inter, "f32")
    WGTB = wtype(layers, dim, 2 * inter)
    WDT = wtype(layers, inter, dim, "f32")
    WDTB = wtype(layers, inter, dim)

    # `[%L, red, out]` or `[%L, out, red]`, for every site that reads one.
    def wix(red, out):
        return f"{out}, {red}" if out_major else f"{red}, {out}"

    KVT = f"memref<{layers}x{total}x{kv_heads}x{hd}xf32>"
    NT = f"memref<{layers}x{dim}xf32>"
    QKNT = f"memref<{layers}x{2 * hd}xf32>"
    ROT = f"memref<{total}x{2 * hd}xf32>"
    EMT = f"memref<{vocab}x{dim}xf32>"
    EMTB = f"memref<{vocab}x{dim}xbf16>"
    LMT = f"memref<{dim}x{vocab}xf32>"
    LMTB = f"memref<{vocab}x{dim}xbf16>"
    LGT = f"memref<{tokens}x{vocab}xf32>"
    PVT = f"memref<{tokens}x{tasks_v}xf32>"
    PIT = f"memref<{tokens}x{tasks_v}xi32>"
    # One flat token stream rather than a token per (step, row). A row of a
    # step is a position in the same sequence -- the attention inside a step is
    # windowed-causal, so token m of a step already reads what token m-1 of the
    # same step appended -- and a sequence has one next token, not `tokens` of
    # them. Fleet keeps the same shape: config.tokens is indexed by absolute
    # position within a request (persistent_kernel.cuh:397).
    maxseq = steps * tokens + 1
    TKT = f"memref<{maxseq}xi32>"
    PLT = "memref<1xi32>"
    NFT = f"memref<{dim}xf32>"

    # How many of the stream's tokens are the prompt. Everything past it is
    # generated. Default: the whole run is prompt, which makes every step a
    # prefill chunk of `tokens` -- the shape this generator had before decode
    # existed, so the existing tests stay an exact regression baseline.
    if prompt:
        prompt_len = prompt_len or len(prompt)
        assert prompt_len == len(
            prompt
        ), "--prompt-len must match the length of --prompt"
    prompt_len = prompt_len or steps * tokens
    # A prompt longer than the window is consumed over several steps, which is
    # Fleet's chunked prefill (MPK_MAX_TOKENS_PER_REQUEST, persistent_kernel.cuh
    # :445). It needs enough steps to get through the prompt and still decode.
    assert 0 < prompt_len <= steps * tokens, "prompt-len out of range"

    if prompt:
        prompt_init = "\n".join(
            f"    %pt{k} = arith.constant {tid} : i32\n"
            f"    %pk{k} = arith.constant {k} : index\n"
            f"    memref.store %pt{k}, %Tok[%pk{k}] : {TKT}"
            for k, tid in enumerate(prompt)
        )
    else:
        prompt_init = (
            "    scf.for %m = %c0 to %cplen step %c1 {\n"
            "      %mm = arith.index_cast %m : index to i32\n"
            "      %t0 = arith.muli %mm, %c3i : i32\n"
            "      %t1 = arith.addi %t0, %one : i32\n"
            "      %t2 = arith.remsi %t1, %cvocabi : i32\n"
            f"      memref.store %t2, %Tok[%m] : {TKT}\n"
            "    }"
        )

    # Nine stages a layer, plus however many empty ones were asked for. A pad
    # stage claims its pieces, signals, and waits, and its body computes
    # nothing -- so the model still produces the right tokens and the launch
    # gets longer by exactly what a stage boundary costs. That is the only way
    # to price the boundary with an instrument other than the one that found
    # it: the chained per-class table charges every stage the same several
    # microseconds whatever it computes, and a table cannot check itself.
    # Here the answer is a slope on the launch clock, and it is 5.52 us.
    # Eight stages a layer when swiglu rides along in gate_up, nine when it
    # is a stage of its own, plus however many empty ones were asked for.
    layer_stages = (8 if fuse_swiglu else 9) + pad_stages
    stages = layer_stages
    # Where `down` sits once swiglu may or may not be in front of it. Every
    # stage after it -- the pads, and the five outside the layer loop -- is
    # numbered from here, so there is one place to change.
    down_stage = 7 if fuse_swiglu else 8
    # Around the layers: embed at the front, then the final norm, the lm head
    # and Fleet's two-stage argmax (argmax_partial_layer + argmax_reduce_layer,
    # builder.py:799-811). Their slots sit past the layers' in the same
    # per-step block.
    extras = 5
    per_step = layers * stages + extras
    events = per_step * steps
    # How many per-chiplet task queues the scheduler keeps, and therefore how
    # many it probes before it can conclude a stage is drained. Every probe is
    # an atomic claim plus a broadcast of the result to the workgroup, which
    # past one wavefront means LDS and two barriers -- and a stage pays that
    # `dies` times whatever it computes. Measured on MI350X, dropping 16 to 8
    # took 15.42 to 10.57 ms/token, and only the stages that probe moved: the
    # single-task rmsnorms and the argmax reduce did not shift at all.
    #
    # It must be at least the device's chiplet count. `%mydie` is
    # `air.chiplet_id % dies`, so if two chiplets alias onto one queue the
    # two-level flush compares arrivals against `air.chiplet_dim_blocks` --
    # this chiplet's workgroups -- and the count no longer matches what shows
    # up. MI300X and MI350X are both 8 XCDs.
    maxdies = dies
    # The static claim tightens this from "at least the chiplet count" to
    # "exactly it". Pieces are partitioned over die indices 0..dies-1, so a die
    # index that no workgroup reports keeps its pieces, nobody computes them,
    # the stage's event never reaches its total and the launch hangs. With
    # `dies` larger than the hardware, the indices past the end are exactly
    # that. Half of it is checkable here; the other half is a property of the
    # part and is why `dies` defaults to 8.
    if static_claim:
        assert workers >= dies, (
            f"--static-claim partitions the pieces over {dies} chiplets and "
            f"there are only {workers} workgroups, so some chiplet gets none "
            "and its pieces are never computed. Raise --workers to at least "
            "--dies, or pass --dynamic-claim"
        )
    qslots = (stages + extras) * maxdies * 2
    QUT = f"memref<{steps}x{layers}x{qslots}xi32>"
    slots = steps * per_step * maxdies
    flushword = 2 * slots
    # A device-scope atomic per die per stage, 2056 of them a step, whose only
    # reader is a host-side print. It is an instrument, not part of the
    # protocol -- which is why it is off unless asked for. Priced at 0.3% of
    # the launch, which is also a price for one device-scope atomic per die
    # per boundary and so is worth knowing on its own.
    # One i32 accumulator per stage class for --timers. Qwen is a pile of
    # operators, and the only honest way to say which one costs what is to time
    # each on the device -- not to infer it by subtracting deliberately-broken
    # builds, which measures "cost of the program without this stage" and does
    # not sum to the total when the stages meet at a rendezvous.
    nclass = 2 * (stages + extras)  # body and rendezvous wait, per class
    timerbase = flushword + 1
    flush_count = (
        "              %cnt{l}_{stage} = llvm.atomicrmw add %flushP, %one_s "
        'syncscope("") monotonic : !llvm.ptr, i32\n'
        if count_flushes
        else ""
    )
    locwords = timerbase + (nclass + 1 if timers else 0)
    # One workgroup owns the clock. Timing from all of them and summing would
    # overflow i32 and would also count the same wall-clock window `workers`
    # times; one workgroup's view of a stage is that stage's duration, waiting
    # at the rendezvous included, which is exactly its contribution to the
    # critical path.
    timer_id = (
        ""
        if not timers
        else """        %isDie0 = arith.cmpi eq, %mydie, %c0_s : index
        %isRank0 = arith.cmpi eq, %myrank2, %c0_s : index
        %isWG0 = arith.andi %isDie0, %isRank0 : i1"""
    )
    # Names in stage order: nine per layer, then the five outside the loop.
    timer_names = [
        "rmsnorm.attn",
        "qkv",
        "rope+kv_append",
        "attention",
        "o_proj",
        "rmsnorm.mlp",
        "gate_up",
        *([] if fuse_swiglu else ["swiglu"]),
        "down",
        *[f"pad{i}" for i in range(pad_stages)],
        "embed",
        "final_norm",
        "lm_head",
        "argmax_partial",
        "argmax_reduce",
    ]
    # The sum of the stages is what workgroup 0 spent inside them. It is not
    # the same thing as how long the kernel ran, and the difference is exactly
    # the cost that no per-operator table can show. Two more clock reads say
    # what it is.
    timer_launch0 = (
        ""
        if not timers
        else '        %tl0 = llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"() : () -> i64'
    )
    timer_launch1 = (
        ""
        if not timers
        else f"""          %tl1 = llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"() : () -> i64
          %tld = arith.subi %tl1, %tl0 : i64
          %tldi = arith.trunci %tld : i64 to i32
          %isTL = arith.andi %isWG0, %isLead : i1
          scf.if %isTL {{
            %tlk = arith.constant {timerbase + nclass} : index
            %tlo = memref.load %sloc[%tlk] : memref<{locwords}xi32>
            %tln = arith.addi %tlo, %tldi : i32
            memref.store %tln, %sloc[%tlk] : memref<{locwords}xi32>
          }}"""
    )
    timer_report = (
        ""
        if not timers
        else (
            '    vector.print str "--- per-operator device ticks (100 MHz), '
            'workgroup 0, summed over layers and steps:"\n'
            + f"\n    %tlk = arith.constant {timerbase + nclass} : index\n"
            f"    %tlv = memref.load %Loc[%tlk] : memref<{locwords}xi32>\n"
            '    vector.print str "  WHOLE LAUNCH, workgroup 0"\n'
            "    vector.print %tlv : i32\n"
            + "\n".join(
                f"    %tk{i} = arith.constant {timerbase + i} : index\n"
                f"    %tv{i} = memref.load %Loc[%tk{i}] : memref<{locwords}xi32>\n"
                f"    %wk{i} = arith.constant {timerbase + nclass // 2 + i} : index\n"
                f"    %wv{i} = memref.load %Loc[%wk{i}] : memref<{locwords}xi32>\n"
                f'    vector.print str "  {nm} body"\n'
                f"    vector.print %tv{i} : i32\n"
                f'    vector.print str "  {nm} wait"\n'
                f"    vector.print %wv{i} : i32"
                for i, nm in enumerate(timer_names)
            )
        )
    )
    # Every layer stage but 0 and 5 splits into pieces, and so does every pad.
    strided_stages = 7 + pad_stages
    # embed, lm head and the partial argmax are split by piece too; the final
    # norm and the argmax reduce are single-task.
    naive = workers * (strided_stages * layers + 3) * steps
    flush_report = (
        f"""
    %cflush = arith.constant {flushword} : index
    %flushes = memref.load %Loc[%cflush] : memref<{locwords}xi32>
    vector.print str "device-scope event flushes = "
    vector.print %flushes : i32
    %cnaive = arith.constant {naive} : i32
    vector.print str "what signalling per worker would have been = "
    vector.print %cnaive : i32"""
        if count_flushes
        else ""
    )

    s_qkv = _scale(MOD_QKV, dim)
    s_o = _scale(MOD_O, dim)
    s_g = _scale(MOD_G, dim)
    s_u = _scale(MOD_U, dim)
    s_d = _scale(MOD_D, inter)
    # Only used to build a synthetic KV prefix; there is none when cache == 0.
    s_k = _scale(MOD_K, cache, _math.sqrt(cache)) if cache else 0.0
    s_v = _scale(MOD_V, cache, _math.sqrt(cache)) if cache else 0.0

    o = []
    w = o.append

    w(f"""// Generated by test/gpu/megakernel_gen/gen.py -- do not edit.
//   layers={layers} dim={dim} inter={inter} heads={heads} kv_heads={kv_heads}
//   head_dim={hd} tasks/stage={tasks} workers={workers} tokens={tokens}
//   kv prefix={cache} attention length={total} waves/workgroup={waves}
//
// A decode chain as a megakernel: one launch, {layers} layers, every stage
// boundary an event rather than a return to the host.
module {{
{lds_globals}
{'  func.func private @air_load_weights(!llvm.ptr, i64, i64) -> ()' if W else ''}
  func.func @main() {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cdim = arith.constant {dim} : index
    %cinter = arith.constant {inter} : index
    %c2inter = arith.constant {2 * inter} : index
    %cqkvo = arith.constant {qkvo} : index
    %cqw = arith.constant {qw} : index
    %fred_qw = arith.constant {float(qw):.6e} : f32
    %chd = arith.constant {hd} : index
    %c2hd = arith.constant {2 * hd} : index
    %ch2 = arith.constant {h2} : index
    %cheads = arith.constant {heads} : index
    %ckvh = arith.constant {kv_heads} : index
    %cpre = arith.constant {cache} : index
    %ctotal = arith.constant {total} : index
    %cgroup = arith.constant {group} : index
    %cwin = arith.constant {steps * tokens} : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %c3i = arith.constant 3 : i32
    %cvocabi = arith.constant {vocab} : i32
    %fzero = arith.constant 0.0 : f32
    %fone = arith.constant 1.0 : f32
    %eps = arith.constant 1.0e-6 : f32
    %fdim = arith.constant {float(dim):.6e} : f32
    %fhd = arith.constant {float(hd):.6e} : f32
    %fred_d = arith.constant {float(dim):.6e} : f32
    %fred_i = arith.constant {float(inter):.6e} : f32
    %fred_p = arith.constant {float(cache):.6e} : f32
    %c2f = arith.constant 2.0 : f32
    %c3f = arith.constant 3.0 : f32
    %c5f = arith.constant 5.0 : f32
    %c7f = arith.constant 7.0 : f32
    %c11f = arith.constant 11.0 : f32
    %c13f = arith.constant 13.0 : f32
    %quarter = arith.constant 2.500000e-01 : f32
    %eighth = arith.constant 1.250000e-01 : f32
    %sixteenth = arith.constant 6.250000e-02 : f32
    %half = arith.constant 5.000000e-01 : f32
    %inv65536 = arith.constant 1.52587890625e-05 : f32
    %clayers = arith.constant {layers} : index
    %ctok = arith.constant {tokens} : index
    %csteps = arith.constant {steps} : index
    %cplen = arith.constant {prompt_len} : index
    %cmaxseq = arith.constant {maxseq} : index

    %X = memref.alloc() : {AT}
    %X0 = memref.alloc() : {AT}
    %Rv = memref.alloc() : {AT}
    %QKV = memref.alloc() : {QT}
    %Sc = memref.alloc() : {SCT}
    %Av = memref.alloc() : {QWT}
    %Aov = memref.alloc() : {AT}
    %Xa = memref.alloc() : {AT}
    %GU = memref.alloc() : {GT}
    %Actv = memref.alloc() : {IT}
    %Wqkv = memref.alloc() : {WQT}
    %Wo = memref.alloc() : {WT}
    %Wgu = memref.alloc() : {WGT}
    %Wd = memref.alloc() : {WDT}
    %Kc = memref.alloc() : {KVT}
    %Vc = memref.alloc() : {KVT}
    %N1 = memref.alloc() : {NT}
    %N2 = memref.alloc() : {NT}
    %QKN = memref.alloc() : {QKNT}
    %RO = memref.alloc() : {ROT}
    %ref = memref.alloc() : {AT}
    %Emb = memref.alloc() : {EMT}
    %Wlm = memref.alloc() : {LMT}
    %Nf = memref.alloc() : {NFT}
    %Tok = memref.alloc() : {TKT}
    %Lg = memref.alloc() : {LGT}
    %PV = memref.alloc() : {PVT}
    %PI = memref.alloc() : {PIT}
    %Xf = memref.alloc() : {AT}
    %Tok0 = memref.alloc() : {TKT}
    %LgD = memref.alloc() : {LGT}
    // The prompt length reaches the device as data, not as a folded constant:
    // how many tokens a step has is a runtime quantity in Fleet too
    // (qo_indptr_buffer, read inside the task -- see
    // multitoken_paged_attention_mfma_mi300.cuh:78-83), because prefill and
    // decode run the same static task graph with different trip counts.
    %Plen = memref.alloc() : {PLT}
    %pleni = arith.constant {prompt_len} : i32
    memref.store %pleni, %Plen[%c0] : {PLT}

    // Each token gets a different *pattern*, not the same pattern shifted by a
    // constant. A constant offset does not survive: rmsnorm divides most of it
    // out, so the queries come out within a few percent of each other and a
    // token reading another token's scores is indistinguishable from correct.
    scf.for %m = %c0 to %ctok step %c1 {{
      scf.for %i = %c0 to %cdim step %c1 {{
        %ii = arith.index_cast %i : index to i32
        %fi = arith.sitofp %ii : i32 to f32
        %mm = arith.index_cast %m : index to i32
        %fm = arith.sitofp %mm : i32 to f32
        %fi3 = arith.mulf %fi, %c3f : f32
        %fm7 = arith.mulf %fm, %c7f : f32
        %fim = arith.addf %fi3, %fm7 : f32
        %v0 = arith.remf %fim, %c11f : f32
        %v = arith.mulf %v0, %sixteenth : f32
        %v2 = arith.addf %v, %eighth : f32
        memref.store %v2, %X[%m, %i] : {AT}
        memref.store %v2, %X0[%m, %i] : {AT}
        memref.store %v2, %ref[%m, %i] : {AT}
        memref.store %fzero, %Rv[%m, %i] : {AT}
        memref.store %fzero, %Aov[%m, %i] : {AT}
        memref.store %fzero, %Xa[%m, %i] : {AT}
      }}
      scf.for %i = %c0 to %cqkvo step %c1 {{
        memref.store %fzero, %QKV[%m, %i] : {QT}
      }}
      scf.for %i = %c0 to %cqw step %c1 {{
        memref.store %fzero, %Av[%m, %i] : {QWT}
      }}
      scf.for %i = %c0 to %c2inter step %c1 {{
        memref.store %fzero, %GU[%m, %i] : {GT}
      }}
      scf.for %i = %c0 to %cinter step %c1 {{
        memref.store %fzero, %Actv[%m, %i] : {IT}
      }}
      scf.for %h = %c0 to %cheads step %c1 {{
        scf.for %t = %c0 to %ctotal step %c1 {{
          memref.store %fzero, %Sc[%m, %h, %t] : {SCT}
        }}
      }}
    }}

    // Norm weights. Not all ones -- a weight of one is a weight that is not
    // being tested. QKN holds the q head norm in [0, hd) and the k head norm
    // in [hd, 2hd); those two are Qwen3's, and Qwen2 has neither.
    scf.for %lz = %c0 to %clayers step %c1 {{
      %ll = arith.index_cast %lz : index to i32
      %fl = arith.sitofp %ll : i32 to f32
      scf.for %i = %c0 to %cdim step %c1 {{
        %ii = arith.index_cast %i : index to i32
        %fi = arith.sitofp %ii : i32 to f32
        %a0 = arith.addf %fi, %fl : f32
        %r0 = arith.remf %a0, %c3f : f32
        %d0 = arith.subf %r0, %fone : f32
        %s0 = arith.mulf %d0, %quarter : f32
        %n1v = arith.addf %fone, %s0 : f32
        memref.store %n1v, %N1[%lz, %i] : {NT}
        %a1 = arith.addf %a0, %fone : f32
        %r1 = arith.remf %a1, %c5f : f32
        %d1 = arith.subf %r1, %c2f : f32
        %s1 = arith.mulf %d1, %eighth : f32
        %n2v = arith.addf %fone, %s1 : f32
        memref.store %n2v, %N2[%lz, %i] : {NT}
      }}
      scf.for %d = %c0 to %chd step %c1 {{
        %dd = arith.index_cast %d : index to i32
        %fd = arith.sitofp %dd : i32 to f32
        %b0 = arith.addf %fd, %fl : f32
        %q0 = arith.remf %b0, %c3f : f32
        %q1 = arith.subf %q0, %fone : f32
        %q2 = arith.mulf %q1, %quarter : f32
        %qw = arith.addf %fone, %q2 : f32
        memref.store %qw, %QKN[%lz, %d] : {QKNT}
        %b1 = arith.addf %b0, %fone : f32
        %k0 = arith.remf %b1, %c3f : f32
        %k1 = arith.subf %k0, %fone : f32
        %k2 = arith.mulf %k1, %quarter : f32
        %kw = arith.addf %fone, %k2 : f32
        %dk = arith.addi %d, %chd : index
        memref.store %kw, %QKN[%lz, %dk] : {QKNT}
      }}
    }}

    // RoPE tables, cos in [0, hd) and sin in [hd, 2hd). Both halves of a head
    // use the same frequency, which is what makes rotate-half a rotation.
    %lnbase = arith.constant {lnrope:.10e} : f32
    scf.for %t = %c0 to %ctotal step %c1 {{
      %tt = arith.index_cast %t : index to i32
      %ft = arith.sitofp %tt : i32 to f32
      scf.for %d = %c0 to %c2hd step %c1 {{
        %dm = arith.remui %d, %ch2 : index
        %dmm = arith.index_cast %dm : index to i32
        %fdm = arith.sitofp %dmm : i32 to f32
        %two_d = arith.mulf %fdm, %c2f : f32
        %ratio = arith.divf %two_d, %fhd : f32
        %negr = arith.mulf %ratio, %lnbase : f32
        %negr2 = arith.negf %negr : f32
        %freq = math.exp %negr2 : f32
        %ang = arith.mulf %ft, %freq : f32
        %isSin = arith.cmpi uge, %d, %chd : index
        %cv = math.cos %ang : f32
        %sv = math.sin %ang : f32
        %val = arith.select %isSin, %sv, %cv : f32
        memref.store %val, %RO[%t, %d] : {ROT}
      }}
    }}

    // The vocabulary ends of the model. Fleet has these too: embed_layer at
    // the front (builder.py:755) and rmsnorm + lm head + a two-stage argmax at
    // the back (:772-811).
    %cvocab = arith.constant {vocab} : index
    %ctasks_h = arith.constant {tasks} : index
""")

    # Synthetic embedding and lm head. With --weights these are read from
    # the checkpoint instead; see the loader below.
    if not W:
        w(f"""
        // Embedding and lm head come from an integer hash rather than a modular
        // pattern. `f(a*i + b*v) mod m` makes every column a shift of one pattern:
        // the columns repeat with period m, pair up as exact negatives, and two of
        // them come out identical, so logits tie exactly and argmax is decided by
        // index order instead of by the hidden state. Measured on the modular
        // version: 254 distinct columns out of 256, rank 127, and a top-1/top-2 gap
        // of exactly zero. The hash gives 256 distinct columns and a worst
        // off-diagonal correlation of 0.34.
        %hk1 = arith.constant 374761393 : i32
        %hk2 = arith.constant 668265263 : i32
        %hk3 = arith.constant 1274126177 : i32
        %hs13 = arith.constant 13 : i32
        %hs16 = arith.constant 16 : i32
        %hmask = arith.constant 65535 : i32
        %hseedE = arith.constant 12345 : i32
        %hseedL = arith.constant 98765 : i32
        scf.for %v = %c0 to %cvocab step %c1 {{
          %vv = arith.index_cast %v : index to i32
          scf.for %i = %c0 to %cdim step %c1 {{
            %ii = arith.index_cast %i : index to i32
            %m1 = arith.muli %ii, %hk1 : i32
            %m2 = arith.muli %vv, %hk2 : i32
            %h0 = arith.addi %m1, %m2 : i32
            %h1 = arith.addi %h0, %hseedE : i32
            %r1 = arith.shrui %h1, %hs13 : i32
            %h2 = arith.xori %h1, %r1 : i32
            %h3 = arith.muli %h2, %hk3 : i32
            %r2 = arith.shrui %h3, %hs16 : i32
            %h4 = arith.xori %h3, %r2 : i32
            %hm = arith.andi %h4, %hmask : i32
            %hf = arith.uitofp %hm : i32 to f32
            %u0 = arith.mulf %hf, %inv65536 : f32
            %u1 = arith.subf %u0, %half : f32
            %u2 = arith.mulf %u1, %quarter : f32
            %u3 = arith.addf %u2, %eighth : f32
            memref.store %u3, %Emb[%v, %i] : {EMT}
          }}
        }}
        scf.for %i = %c0 to %cdim step %c1 {{
          %ii = arith.index_cast %i : index to i32
          %fi = arith.sitofp %ii : i32 to f32
          %n0 = arith.remf %fi, %c3f : f32
          %n1 = arith.subf %n0, %fone : f32
          %n2 = arith.mulf %n1, %quarter : f32
          %n3 = arith.addf %fone, %n2 : f32
          memref.store %n3, %Nf[%i] : {NFT}
          %iw = arith.index_cast %i : index to i32
          scf.for %v = %c0 to %cvocab step %c1 {{
            %vv = arith.index_cast %v : index to i32
            %m1 = arith.muli %iw, %hk1 : i32
            %m2 = arith.muli %vv, %hk2 : i32
            %h0 = arith.addi %m1, %m2 : i32
            %h1 = arith.addi %h0, %hseedL : i32
            %r1 = arith.shrui %h1, %hs13 : i32
            %h2 = arith.xori %h1, %r1 : i32
            %h3 = arith.muli %h2, %hk3 : i32
            %r2 = arith.shrui %h3, %hs16 : i32
            %h4 = arith.xori %h3, %r2 : i32
            %hm = arith.andi %h4, %hmask : i32
            %hf = arith.uitofp %hm : i32 to f32
            memref.store %hf, %Wlm[%i, %v] : {LMT}
          }}
        }}
        // lm head, centred along its reduction axis like every other weight
        %slmc = arith.constant {_math.sqrt(12.0) / 65536.0 / _math.sqrt(dim):.10e} : f32
        scf.for %v = %c0 to %cvocab step %c1 {{
          %sm = scf.for %i = %c0 to %cdim step %c1
              iter_args(%a = %fzero) -> (f32) {{
            %x = memref.load %Wlm[%i, %v] : {LMT}
            %a2 = arith.addf %a, %x : f32
            scf.yield %a2 : f32
          }}
          %mu = arith.divf %sm, %fred_d : f32
          scf.for %i = %c0 to %cdim step %c1 {{
            %x = memref.load %Wlm[%i, %v] : {LMT}
            %y = arith.subf %x, %mu : f32
            %z = arith.mulf %y, %slmc : f32
            memref.store %z, %Wlm[%i, %v] : {LMT}
          }}
        }}
""")

    w(f"""
    // The prompt, and room for what each step produces.
    %csteps1 = arith.constant {steps + 1} : index
    %csliceVh = arith.constant {slice_v} : index
    scf.for %m = %c0 to %cmaxseq step %c1 {{
      memref.store %zero, %Tok[%m] : {TKT}
    }}
{prompt_init}
    scf.for %m = %c0 to %ctok step %c1 {{
      scf.for %v = %c0 to %cvocab step %c1 {{
        memref.store %fzero, %Lg[%m, %v] : {LGT}
      }}
      scf.for %k = %c0 to %ctasks_h step %c1 {{
        memref.store %fzero, %PV[%m, %k] : {PVT}
        memref.store %zero, %PI[%m, %k] : {PIT}
      }}
      scf.for %i = %c0 to %cdim step %c1 {{
        memref.store %fzero, %Xf[%m, %i] : {AT}
      }}
    }}
""")

    # Raw weight patterns; the centring pass below is what makes them usable.
    # --weights replaces all of this with a read from the checkpoint.
    if not W:
        w(f"""
    scf.for %l = %c0 to %clayers step %c1 {{
      %ll = arith.index_cast %l : index to i32
      %fl = arith.sitofp %ll : i32 to f32
      scf.for %i = %c0 to %cdim step %c1 {{
        %ii = arith.index_cast %i : index to i32
        %fi = arith.sitofp %ii : i32 to f32
        scf.for %n = %c0 to %cqkvo step %c1 {{
          %nn = arith.index_cast %n : index to i32
          %fn = arith.sitofp %nn : i32 to f32
          %a = arith.subf %fi, %fn : f32
          %a2 = arith.addf %a, %fl : f32
          %a3 = arith.addf %a2, %fone : f32
          %rv = arith.remf %a3, %c{MOD_QKV}f : f32
          memref.store %rv, %Wqkv[%l, %i, %n] : {WQT}
        }}

        scf.for %p = %c0 to %cinter step %c1 {{
          %pp = arith.index_cast %p : index to i32
          %fp = arith.sitofp %pp : i32 to f32
          %e = arith.subf %fi, %fp : f32
          %e2 = arith.addf %e, %fl : f32
          %rg = arith.remf %e2, %c{MOD_G}f : f32
          memref.store %rg, %Wgu[%l, %i, %p] : {WGT}
          %e3 = arith.addf %e2, %c3f : f32
          %ru = arith.remf %e3, %c{MOD_U}f : f32
          %pu = arith.addi %p, %cinter : index
          memref.store %ru, %Wgu[%l, %i, %pu] : {WGT}
        }}
      }}
      scf.for %a = %c0 to %cqw step %c1 {{
        %aa = arith.index_cast %a : index to i32
        %fa = arith.sitofp %aa : i32 to f32
        scf.for %j = %c0 to %cdim step %c1 {{
          %jj = arith.index_cast %j : index to i32
          %fj = arith.sitofp %jj : i32 to f32
          %d = arith.subf %fa, %fj : f32
          %d2 = arith.addf %d, %fl : f32
          %d3 = arith.addf %d2, %fone : f32
          %ro = arith.remf %d3, %c{MOD_O}f : f32
          memref.store %ro, %Wo[%l, %a, %j] : {WT}
        }}
      }}
      scf.for %p = %c0 to %cinter step %c1 {{
        %pp = arith.index_cast %p : index to i32
        %fp = arith.sitofp %pp : i32 to f32
        scf.for %j = %c0 to %cdim step %c1 {{
          %jj = arith.index_cast %j : index to i32
          %fj = arith.sitofp %jj : i32 to f32
          %f = arith.subf %fp, %fj : f32
          %f2 = arith.addf %f, %fl : f32
          %rd = arith.remf %f2, %c{MOD_D}f : f32
          memref.store %rd, %Wd[%l, %p, %j] : {WDT}
        }}
      }}
      // The prefix of the KV cache (empty when --cache 0, which is what a real
      // run uses: every entry then comes from the projection). The window
      // entries are written by the
      // kernel, from the projection, so only [0, prefix) is filled here.
      // K varies with t*t, not t: with t+d every K row is a cyclic shift of
      // one pattern, so q.K[t] moves with t only through a shift correlation,
      // the softmax comes out nearly uniform, and the attention output stops
      // depending on the query at all.
      scf.for %t = %c0 to %cpre step %c1 {{
        %tt = arith.index_cast %t : index to i32
        %ft = arith.sitofp %tt : i32 to f32
        %ft2 = arith.mulf %ft, %ft : f32
        scf.for %hk = %c0 to %ckvh step %c1 {{
          %hh = arith.index_cast %hk : index to i32
          %fh = arith.sitofp %hh : i32 to f32
          scf.for %d = %c0 to %chd step %c1 {{
            %dd = arith.index_cast %d : index to i32
            %fd = arith.sitofp %dd : i32 to f32
            %fd3 = arith.mulf %fd, %c3f : f32
            %ks = arith.addf %ft2, %fd3 : f32
            %ks2 = arith.addf %ks, %fl : f32
            %ks3 = arith.addf %ks2, %fh : f32
            %kr = arith.remf %ks3, %c{MOD_K}f : f32
            memref.store %kr, %Kc[%l, %t, %hk, %d] : {KVT}
            %vs = arith.subf %ft, %fd : f32
            %vs2 = arith.addf %vs, %fone : f32
            %vs3 = arith.addf %vs2, %fh : f32
            %vr = arith.remf %vs3, %c{MOD_V}f : f32
            memref.store %vr, %Vc[%l, %t, %hk, %d] : {KVT}
          }}
        }}
      }}
    }}
""")

    def centre(buf, mtype, red_const, red_len_const, out_const, scale, tag):
        """Subtract each column's mean along the reduction axis, then scale.

        `x % mod` lands in [0, mod), so an uncentred weight matrix is entirely
        non-negative and a matmul over it sums non-negative terms rather than
        taking a random walk. Subtracting a constant is not enough: the columns
        only sum to zero if the reduction length is a multiple of the modulus,
        and it is not. The leftover DC term is multiplied by the mean of the
        activation and by the reduction length, which is how the attention
        output came to swamp the residual it was meant to perturb.
        """
        return f"""
    scf.for %l = %c0 to %clayers step %c1 {{
      scf.for %o_{tag} = %c0 to {out_const} step %c1 {{
        %sum_{tag} = scf.for %k_{tag} = %c0 to {red_const} step %c1
            iter_args(%acc_{tag} = %fzero) -> (f32) {{
          %v_{tag} = memref.load {buf}[%l, %k_{tag}, %o_{tag}] : {mtype}
          %a_{tag} = arith.addf %acc_{tag}, %v_{tag} : f32
          scf.yield %a_{tag} : f32
        }}
        %mu_{tag} = arith.divf %sum_{tag}, {red_len_const} : f32
        scf.for %k2_{tag} = %c0 to {red_const} step %c1 {{
          %v2_{tag} = memref.load {buf}[%l, %k2_{tag}, %o_{tag}] : {mtype}
          %d_{tag} = arith.subf %v2_{tag}, %mu_{tag} : f32
          %s_{tag} = arith.mulf %d_{tag}, {scale} : f32
          memref.store %s_{tag}, {buf}[%l, %k2_{tag}, %o_{tag}] : {mtype}
        }}
      }}
    }}"""

    if not W:
        for nm, val in (
            ("sqkv", s_qkv),
            ("so", s_o),
            ("sg", s_g),
            ("su", s_u),
            ("sd", s_d),
            ("sk", s_k),
            ("sv", s_v),
        ):
            w(f"\n    %{nm}c = arith.constant {val:.10e} : f32")
        w(centre("%Wqkv", WQT, "%cdim", "%fred_d", "%cqkvo", "%sqkvc", "wq"))
        w(centre("%Wo", WT, "%cqw", "%fred_qw", "%cdim", "%soc", "wo"))
        # Wgu holds the gate half and the up half side by side and they come
        # from different moduli, so the scale is selected per column.
        w(f"""
    scf.for %l = %c0 to %clayers step %c1 {{
      scf.for %o_gu = %c0 to %c2inter step %c1 {{
        %isup = arith.cmpi uge, %o_gu, %cinter : index
        %scl = arith.select %isup, %suc, %sgc : f32
        %sum_gu = scf.for %k_gu = %c0 to %cdim step %c1
            iter_args(%acc_gu = %fzero) -> (f32) {{
          %v_gu = memref.load %Wgu[%l, %k_gu, %o_gu] : {WGT}
          %a_gu = arith.addf %acc_gu, %v_gu : f32
          scf.yield %a_gu : f32
        }}
        %mu_gu = arith.divf %sum_gu, %fred_d : f32
        scf.for %k2_gu = %c0 to %cdim step %c1 {{
          %v2_gu = memref.load %Wgu[%l, %k2_gu, %o_gu] : {WGT}
          %d_gu = arith.subf %v2_gu, %mu_gu : f32
          %s_gu = arith.mulf %d_gu, %scl : f32
          memref.store %s_gu, %Wgu[%l, %k2_gu, %o_gu] : {WGT}
        }}
      }}
    }}""")
        w(centre("%Wd", WDT, "%cinter", "%fred_i", "%cdim", "%sdc", "wd"))
        # The KV prefix is 4-D and only its prefix rows exist, so it gets its
        # own loop rather than the helper's.
        w(f"""
    scf.for %l = %c0 to %clayers step %c1 {{
      scf.for %hk = %c0 to %ckvh step %c1 {{
        scf.for %d = %c0 to %chd step %c1 {{
          %sk_s = scf.for %t = %c0 to %cpre step %c1
              iter_args(%a = %fzero) -> (f32) {{
            %v = memref.load %Kc[%l, %t, %hk, %d] : {KVT}
            %a2 = arith.addf %a, %v : f32
            scf.yield %a2 : f32
          }}
          %sv_s = scf.for %t = %c0 to %cpre step %c1
              iter_args(%a = %fzero) -> (f32) {{
            %v = memref.load %Vc[%l, %t, %hk, %d] : {KVT}
            %a2 = arith.addf %a, %v : f32
            scf.yield %a2 : f32
          }}
          %muk = arith.divf %sk_s, %fred_p : f32
          %muv = arith.divf %sv_s, %fred_p : f32
          scf.for %t = %c0 to %cpre step %c1 {{
            %kv = memref.load %Kc[%l, %t, %hk, %d] : {KVT}
            %kd = arith.subf %kv, %muk : f32
            %ks = arith.mulf %kd, %skc : f32
            memref.store %ks, %Kc[%l, %t, %hk, %d] : {KVT}
            %vv = memref.load %Vc[%l, %t, %hk, %d] : {KVT}
            %vd = arith.subf %vv, %muv : f32
            %vs = arith.mulf %vd, %svc : f32
            memref.store %vs, %Vc[%l, %t, %hk, %d] : {KVT}
          }}
        }}
      }}
    }}""")

    if W:
        # Read the checkpoint. One extern, a raw pointer and a float offset;
        # weights_loader.c does the file IO and dies on a short read rather
        # than leaving a half-filled buffer to look like a numerical
        # difference later.
        def load(buf, mtype, name, count):
            # The count is this run's buffer size, not the manifest's: with
            # --layers below the checkpoint's, a per-layer tensor is a prefix of
            # what the file holds, and reading the file's count would run off
            # the end of the buffer.
            t = W["tensors"][name]
            assert count <= t["count"], f"{name}: {count} > {t['count']}"
            return f"""
    %pb_{name} = memref.extract_aligned_pointer_as_index {buf} : {mtype} -> index
    %pi_{name} = arith.index_cast %pb_{name} : index to i64
    %pp_{name} = llvm.inttoptr %pi_{name} : i64 to !llvm.ptr
    %of_{name} = arith.constant {t["offset"]} : i64
    %cn_{name} = arith.constant {count} : i64
    func.call @air_load_weights(%pp_{name}, %of_{name}, %cn_{name})
        : (!llvm.ptr, i64, i64) -> ()"""

        w("\n    // Qwen3 weights, straight from the checkpoint.")
        for buf, mtype, name, count in (
            ("%Emb", EMT, "Emb", vocab * dim),
            ("%Wlm", LMT, "Wlm", dim * vocab),
            ("%Nf", NFT, "Nf", dim),
            ("%N1", NT, "N1", layers * dim),
            ("%N2", NT, "N2", layers * dim),
            ("%QKN", QKNT, "QKN", layers * 2 * hd),
            ("%Wqkv", WQT, "Wqkv", layers * dim * qkvo),
            ("%Wo", WT, "Wo", layers * qw * dim),
            ("%Wgu", WGT, "Wgu", layers * dim * 2 * inter),
            ("%Wd", WDT, "Wd", layers * inter * dim),
        ):
            w(load(buf, mtype, name, count))

    w(f"""
    // Narrow the weights to the precision they arrived in. Every one of these
    // came from a bf16 checkpoint and was widened by weights.py, so the
    // truncation is exact -- and it halves what the device has to read, which
    // is what the decode is actually waiting on.
    %WqkvB = memref.alloc() : {WQTB}
    scf.for %l = %c0 to %clayers step %c1 {{
      scf.for %i = %c0 to %cdim step %c1 {{
        scf.for %j = %c0 to %cqkvo step %c1 {{
          %v = memref.load %Wqkv[%l, %i, %j] : {WQT}
          %b = arith.truncf %v : f32 to bf16
          memref.store %b, %WqkvB[%l, {wix("%i", "%j")}] : {WQTB}
        }}
      }}
    }}
    %WoB = memref.alloc() : {WTB}
    scf.for %l = %c0 to %clayers step %c1 {{
      scf.for %i = %c0 to %cqw step %c1 {{
        scf.for %j = %c0 to %cdim step %c1 {{
          %v = memref.load %Wo[%l, %i, %j] : {WT}
          %b = arith.truncf %v : f32 to bf16
          memref.store %b, %WoB[%l, {wix("%i", "%j")}] : {WTB}
        }}
      }}
    }}
    %WguB = memref.alloc() : {WGTB}
    scf.for %l = %c0 to %clayers step %c1 {{
      scf.for %i = %c0 to %cdim step %c1 {{
        scf.for %j = %c0 to %c2inter step %c1 {{
          %v = memref.load %Wgu[%l, %i, %j] : {WGT}
          %b = arith.truncf %v : f32 to bf16
          memref.store %b, %WguB[%l, {wix("%i", "%j")}] : {WGTB}
        }}
      }}
    }}
    %WdB = memref.alloc() : {WDTB}
    scf.for %l = %c0 to %clayers step %c1 {{
      scf.for %i = %c0 to %cinter step %c1 {{
        scf.for %j = %c0 to %cdim step %c1 {{
          %v = memref.load %Wd[%l, %i, %j] : {WDT}
          %b = arith.truncf %v : f32 to bf16
          memref.store %b, %WdB[%l, {wix("%i", "%j")}] : {WDTB}
        }}
      }}
    }}
    %EmbB = memref.alloc() : {EMTB}
    scf.for %i = %c0 to %cvocab step %c1 {{
      scf.for %j = %c0 to %cdim step %c1 {{
        %v = memref.load %Emb[%i, %j] : {EMT}
        %b = arith.truncf %v : f32 to bf16
        memref.store %b, %EmbB[%i, %j] : {EMTB}
      }}
    }}
    %WlmB = memref.alloc() : {LMTB}
    scf.for %i = %c0 to %cdim step %c1 {{
      scf.for %j = %c0 to %cvocab step %c1 {{
        %v = memref.load %Wlm[%i, %j] : {LMT}
        %b = arith.truncf %v : f32 to bf16
        memref.store %b, %WlmB[%j, %i] : {LMTB}
      }}
    }}
""")

    # ---- host reference ----
    w(f"""

    %invsqrthd = arith.constant {invsqrthd:.8e} : f32
    %negbig = arith.constant -1.000000e30 : f32
    %rq = memref.alloc() : memref<{qkvo}xf32>
    %rsc = memref.alloc() : memref<{total}xf32>
    %ra = memref.alloc() : memref<{qw}xf32>
    %rao = memref.alloc() : memref<{dim}xf32>
    %rxa = memref.alloc() : memref<{dim}xf32>
    %rxr = memref.alloc() : memref<{dim}xf32>
    %rgu = memref.alloc() : memref<{2 * inter}xf32>
    %ract = memref.alloc() : memref<{inter}xf32>
    %seqend = scf.for %sp = %c0 to %csteps step %c1
        iter_args(%seq = %c0) -> (index) {{
     // A step consumes as many tokens as are left of the prompt, capped at the
     // window, and one once the prompt is used up -- Fleet's
     // prepare_next_batch, which is `prompt_length - step` clamped for prefill
     // requests and 1 for decode requests (persistent_kernel.cuh:441-450). So
     // the slot the step appends at and the length it attends both move by the
     // step's own token count, not by a fixed stride.
     %rem = arith.subi %cplen, %seq : index
     %isPre = arith.cmpi sgt, %rem, %c0 : index
     %capped = arith.minsi %rem, %ctok : index
     %nat = arith.select %isPre, %capped, %c1 : index
     %wbase = arith.addi %cpre, %seq : index
     %curlen = arith.addi %wbase, %nat : index
     // embed: the step's input is the next slice of the token stream, which
     // for anything past the prompt is what the previous step produced
     scf.for %m = %c0 to %nat step %c1 {{
       %tp = arith.addi %seq, %m : index
       %tk = memref.load %Tok[%tp] : {TKT}
       %tki = arith.index_cast %tk : i32 to index
       scf.for %i = %c0 to %cdim step %c1 {{
         %embh = memref.load %EmbB[%tki, %i] : {EMTB}
         %ev = arith.extf %embh : bf16 to f32
         memref.store %ev, %ref[%m, %i] : {AT}
       }}
     }}
     scf.for %l = %c0 to %clayers step %c1 {{
      // 0, 1, 2 for every token before any attention, because token m's scores
      // read what tokens before it appended.
      scf.for %m = %c0 to %nat step %c1 {{
       %ss = scf.for %i = %c0 to %cdim step %c1
           iter_args(%s = %fzero) -> (f32) {{
         %v = memref.load %ref[%m, %i] : {AT}
         %sq2 = arith.mulf %v, %v : f32
         %s2 = arith.addf %s, %sq2 : f32
         scf.yield %s2 : f32
       }}
       %mean = arith.divf %ss, %fdim : f32
       %me = arith.addf %mean, %eps : f32
       %rms = math.sqrt %me : f32
       scf.for %i = %c0 to %cdim step %c1 {{
         %v = memref.load %ref[%m, %i] : {AT}
         %nv = arith.divf %v, %rms : f32
         %nw = memref.load %N1[%l, %i] : {NT}
         %rv = arith.mulf %nv, %nw : f32
         memref.store %rv, %Rv[%m, %i] : {AT}
       }}
       scf.for %n = %c0 to %cqkvo step %c1 {{
         %acc = scf.for %i = %c0 to %cdim step %c1
             iter_args(%s = %fzero) -> (f32) {{
           %rv = memref.load %Rv[%m, %i] : {AT}
           %wb = memref.load %WqkvB[%l, {wix("%i", "%n")}] : {WQTB}
           %wv = arith.extf %wb : bf16 to f32
           %mu = arith.mulf %rv, %wv : f32
           %s2 = arith.addf %s, %mu : f32
           scf.yield %s2 : f32
         }}
         memref.store %acc, %rq[%n] : memref<{qkvo}xf32>
       }}
       %pos = arith.addi %wbase, %m : index
       // q heads: per-head rmsnorm then rope, in place
       scf.for %h = %c0 to %cheads step %c1 {{
         %hb = arith.muli %h, %chd : index
         %qs = scf.for %d = %c0 to %chd step %c1
             iter_args(%s = %fzero) -> (f32) {{
           %hi = arith.addi %hb, %d : index
           %v = memref.load %rq[%hi] : memref<{qkvo}xf32>
           %sq2 = arith.mulf %v, %v : f32
           %s2 = arith.addf %s, %sq2 : f32
           scf.yield %s2 : f32
         }}
         %qm = arith.divf %qs, %fhd : f32
         %qm2 = arith.addf %qm, %eps : f32
         %qr = math.sqrt %qm2 : f32
         scf.for %d = %c0 to %chd step %c1 {{
           %hi = arith.addi %hb, %d : index
           %v = memref.load %rq[%hi] : memref<{qkvo}xf32>
           %nv = arith.divf %v, %qr : f32
           %nw = memref.load %QKN[%l, %d] : {QKNT}
           %o2 = arith.mulf %nv, %nw : f32
           memref.store %o2, %rq[%hi] : memref<{qkvo}xf32>
         }}
         scf.for %d = %c0 to %ch2 step %c1 {{
           %i1 = arith.addi %hb, %d : index
           %dh = arith.addi %d, %ch2 : index
           %i2 = arith.addi %hb, %dh : index
           %v1 = memref.load %rq[%i1] : memref<{qkvo}xf32>
           %v2 = memref.load %rq[%i2] : memref<{qkvo}xf32>
           %cs = memref.load %RO[%pos, %d] : {ROT}
           %dsin = arith.addi %d, %chd : index
           %sn = memref.load %RO[%pos, %dsin] : {ROT}
           %a1 = arith.mulf %v1, %cs : f32
           %b1 = arith.mulf %v2, %sn : f32
           %o1 = arith.subf %a1, %b1 : f32
           %a2 = arith.mulf %v2, %cs : f32
           %b2 = arith.mulf %v1, %sn : f32
           %o2 = arith.addf %a2, %b2 : f32
           memref.store %o1, %rq[%i1] : memref<{qkvo}xf32>
           memref.store %o2, %rq[%i2] : memref<{qkvo}xf32>
         }}
       }}
       // k heads: same, then appended to the cache. v is appended unchanged.
       scf.for %hk = %c0 to %ckvh step %c1 {{
         %kb0 = arith.muli %cheads, %chd : index
         %khb = arith.muli %hk, %chd : index
         %kb = arith.addi %kb0, %khb : index
         %ks = scf.for %d = %c0 to %chd step %c1
             iter_args(%s = %fzero) -> (f32) {{
           %hi = arith.addi %kb, %d : index
           %v = memref.load %rq[%hi] : memref<{qkvo}xf32>
           %sq2 = arith.mulf %v, %v : f32
           %s2 = arith.addf %s, %sq2 : f32
           scf.yield %s2 : f32
         }}
         %km = arith.divf %ks, %fhd : f32
         %km2 = arith.addf %km, %eps : f32
         %kr = math.sqrt %km2 : f32
         scf.for %d = %c0 to %chd step %c1 {{
           %hi = arith.addi %kb, %d : index
           %v = memref.load %rq[%hi] : memref<{qkvo}xf32>
           %nv = arith.divf %v, %kr : f32
           %dk = arith.addi %d, %chd : index
           %nw = memref.load %QKN[%l, %dk] : {QKNT}
           %o2 = arith.mulf %nv, %nw : f32
           memref.store %o2, %rq[%hi] : memref<{qkvo}xf32>
         }}
         scf.for %d = %c0 to %ch2 step %c1 {{
           %i1 = arith.addi %kb, %d : index
           %dh = arith.addi %d, %ch2 : index
           %i2 = arith.addi %kb, %dh : index
           %v1 = memref.load %rq[%i1] : memref<{qkvo}xf32>
           %v2 = memref.load %rq[%i2] : memref<{qkvo}xf32>
           %cs = memref.load %RO[%pos, %d] : {ROT}
           %dsin = arith.addi %d, %chd : index
           %sn = memref.load %RO[%pos, %dsin] : {ROT}
           %a1 = arith.mulf %v1, %cs : f32
           %b1 = arith.mulf %v2, %sn : f32
           %o1 = arith.subf %a1, %b1 : f32
           %a2 = arith.mulf %v2, %cs : f32
           %b2 = arith.mulf %v1, %sn : f32
           %o2 = arith.addf %a2, %b2 : f32
           memref.store %o1, %rq[%i1] : memref<{qkvo}xf32>
           memref.store %o2, %rq[%i2] : memref<{qkvo}xf32>
         }}
         %vb0 = arith.addi %cheads, %ckvh : index
         %vb1 = arith.muli %vb0, %chd : index
         %vb = arith.addi %vb1, %khb : index
         scf.for %d = %c0 to %chd step %c1 {{
           %ki = arith.addi %kb, %d : index
           %kv = memref.load %rq[%ki] : memref<{qkvo}xf32>
           memref.store %kv, %Kc[%l, %pos, %hk, %d] : {KVT}
           %vi = arith.addi %vb, %d : index
           %vv = memref.load %rq[%vi] : memref<{qkvo}xf32>
           memref.store %vv, %Vc[%l, %pos, %hk, %d] : {KVT}
         }}
       }}
       // keep q for the attention pass below
       scf.for %h = %c0 to %cheads step %c1 {{
         %hb = arith.muli %h, %chd : index
         scf.for %d = %c0 to %chd step %c1 {{
           %hi = arith.addi %hb, %d : index
           %v = memref.load %rq[%hi] : memref<{qkvo}xf32>
           %oi = arith.addi %hb, %d : index
           memref.store %v, %QKV[%m, %oi] : {QT}
         }}
       }}
      }}
      // 3 onwards, now that every token has appended
      scf.for %m = %c0 to %nat step %c1 {{
       %pos = arith.addi %wbase, %m : index
       scf.for %h = %c0 to %cheads step %c1 {{
         %hb = arith.muli %h, %chd : index
         %hk = arith.divui %h, %cgroup : index
         %mxs = scf.for %t = %c0 to %curlen step %c1
             iter_args(%mv = %negbig) -> (f32) {{
           %dot = scf.for %d = %c0 to %chd step %c1
               iter_args(%s = %fzero) -> (f32) {{
             %hi = arith.addi %hb, %d : index
             %qv = memref.load %QKV[%m, %hi] : {QT}
             %kv = memref.load %Kc[%l, %t, %hk, %d] : {KVT}
             %mu = arith.mulf %qv, %kv : f32
             %s2 = arith.addf %s, %mu : f32
             scf.yield %s2 : f32
           }}
           %scv = arith.mulf %dot, %invsqrthd : f32
           // causal over the window: token m sees the prefix and the window
           // entries up to and including its own
           %ok = arith.cmpi ule, %t, %pos : index
           %scm = arith.select %ok, %scv, %negbig : f32
           memref.store %scm, %rsc[%t] : memref<{total}xf32>
           %m2 = arith.maxnumf %mv, %scm : f32
           scf.yield %m2 : f32
         }}
         %sum = scf.for %t = %c0 to %curlen step %c1
             iter_args(%sm = %fzero) -> (f32) {{
           %v = memref.load %rsc[%t] : memref<{total}xf32>
           %d = arith.subf %v, %mxs : f32
           %e = math.exp %d : f32
           memref.store %e, %rsc[%t] : memref<{total}xf32>
           %s2 = arith.addf %sm, %e : f32
           scf.yield %s2 : f32
         }}
         scf.for %d = %c0 to %chd step %c1 {{
           %acc = scf.for %t = %c0 to %curlen step %c1
               iter_args(%s = %fzero) -> (f32) {{
             %e = memref.load %rsc[%t] : memref<{total}xf32>
             %pv = arith.divf %e, %sum : f32
             %vv = memref.load %Vc[%l, %t, %hk, %d] : {KVT}
             %mu = arith.mulf %pv, %vv : f32
             %s2 = arith.addf %s, %mu : f32
             scf.yield %s2 : f32
           }}
           %oi = arith.addi %hb, %d : index
           memref.store %acc, %ra[%oi] : memref<{qw}xf32>
         }}
       }}
       // 4: ao = a @ Wo
       scf.for %j = %c0 to %cdim step %c1 {{
         %acc = scf.for %i = %c0 to %cqw step %c1
             iter_args(%s = %fzero) -> (f32) {{
           %av = memref.load %ra[%i] : memref<{qw}xf32>
           %wb = memref.load %WoB[%l, {wix("%i", "%j")}] : {WTB}
           %wv = arith.extf %wb : bf16 to f32
           %mu = arith.mulf %av, %wv : f32
           %s2 = arith.addf %s, %mu : f32
           scf.yield %s2 : f32
         }}
         memref.store %acc, %rao[%j] : memref<{dim}xf32>
       }}
       // 5: the residual around attention, then the norm the MLP reads. The
       // residual that stage 8 closes is the *unnormalised* one: a decoder
       // layer normalises what the MLP sees and leaves the stream alone.
       %ssa = scf.for %j = %c0 to %cdim step %c1
           iter_args(%s = %fzero) -> (f32) {{
         %xv = memref.load %ref[%m, %j] : {AT}
         %av = memref.load %rao[%j] : memref<{dim}xf32>
         %xa = arith.addf %xv, %av : f32
         memref.store %xa, %rxr[%j] : memref<{dim}xf32>
         %sqa = arith.mulf %xa, %xa : f32
         %s2 = arith.addf %s, %sqa : f32
         scf.yield %s2 : f32
       }}
       %meana = arith.divf %ssa, %fdim : f32
       %mea = arith.addf %meana, %eps : f32
       %rmsa = math.sqrt %mea : f32
       scf.for %j = %c0 to %cdim step %c1 {{
         %xv = memref.load %rxr[%j] : memref<{dim}xf32>
         %nv = arith.divf %xv, %rmsa : f32
         %nw = memref.load %N2[%l, %j] : {NT}
         %xn = arith.mulf %nv, %nw : f32
         memref.store %xn, %rxa[%j] : memref<{dim}xf32>
       }}
       // 6: gu = xa @ Wgu
       scf.for %p = %c0 to %c2inter step %c1 {{
         %acc = scf.for %i = %c0 to %cdim step %c1
             iter_args(%s = %fzero) -> (f32) {{
           %xv = memref.load %rxa[%i] : memref<{dim}xf32>
           %wb = memref.load %WguB[%l, {wix("%i", "%p")}] : {WGTB}
           %wv = arith.extf %wb : bf16 to f32
           %mu = arith.mulf %xv, %wv : f32
           %s2 = arith.addf %s, %mu : f32
           scf.yield %s2 : f32
         }}
         memref.store %acc, %rgu[%p] : memref<{2 * inter}xf32>
       }}
       // 7: act = silu(gate) * up
       scf.for %p = %c0 to %cinter step %c1 {{
         %gv = memref.load %rgu[%p] : memref<{2 * inter}xf32>
         %pu = arith.addi %p, %cinter : index
         %uv = memref.load %rgu[%pu] : memref<{2 * inter}xf32>
         %ng = arith.negf %gv : f32
         %eg = math.exp %ng : f32
         %de = arith.addf %fone, %eg : f32
         %si = arith.divf %gv, %de : f32
         %av = arith.mulf %si, %uv : f32
         memref.store %av, %ract[%p] : memref<{inter}xf32>
       }}
       // 8: x = xa + act @ Wd
       scf.for %j = %c0 to %cdim step %c1 {{
         %acc = scf.for %p = %c0 to %cinter step %c1
             iter_args(%s = %fzero) -> (f32) {{
           %av = memref.load %ract[%p] : memref<{inter}xf32>
           %wb = memref.load %WdB[%l, {wix("%p", "%j")}] : {WDTB}
           %wv = arith.extf %wb : bf16 to f32
           %mu = arith.mulf %av, %wv : f32
           %s2 = arith.addf %s, %mu : f32
           scf.yield %s2 : f32
         }}
         %xv = memref.load %rxr[%j] : memref<{dim}xf32>
         %nx = arith.addf %xv, %acc : f32
         memref.store %nx, %ref[%m, %j] : {AT}
       }}
      }}
     }}
     // final norm, lm head, then argmax the way Fleet does it: a partial per
     // piece, then a reduce over the pieces
     scf.for %m = %c0 to %nat step %c1 {{
       %fs = scf.for %i = %c0 to %cdim step %c1
           iter_args(%a = %fzero) -> (f32) {{
         %v = memref.load %ref[%m, %i] : {AT}
         %q = arith.mulf %v, %v : f32
         %a2 = arith.addf %a, %q : f32
         scf.yield %a2 : f32
       }}
       %fm = arith.divf %fs, %fdim : f32
       %fme = arith.addf %fm, %eps : f32
       %fr = math.sqrt %fme : f32
       scf.for %i = %c0 to %cdim step %c1 {{
         %v = memref.load %ref[%m, %i] : {AT}
         %nv = arith.divf %v, %fr : f32
         %nw = memref.load %Nf[%i] : {NFT}
         %o = arith.mulf %nv, %nw : f32
         memref.store %o, %Xf[%m, %i] : {AT}
       }}
       scf.for %v = %c0 to %cvocab step %c1 {{
         %acc = scf.for %i = %c0 to %cdim step %c1
             iter_args(%a = %fzero) -> (f32) {{
           %xv = memref.load %Xf[%m, %i] : {AT}
           %wb = memref.load %WlmB[%v, %i] : {LMTB}
           %wv = arith.extf %wb : bf16 to f32
           %mu = arith.mulf %xv, %wv : f32
           %a2 = arith.addf %a, %mu : f32
           scf.yield %a2 : f32
         }}
         memref.store %acc, %Lg[%m, %v] : {LGT}
       }}
       // partial: best in each piece
       scf.for %k = %c0 to %ctasks_h step %c1 {{
         %v0 = arith.muli %k, %csliceVh : index
         %bi:2 = scf.for %jj = %c0 to %csliceVh step %c1
             iter_args(%bv = %negbig, %bx = %zero) -> (f32, i32) {{
           %v = arith.addi %v0, %jj : index
           %vokh = arith.cmpi ult, %v, %cvocab : index
           %lv = scf.if %vokh -> (f32) {{
             %lvr = memref.load %Lg[%m, %v] : {LGT}
             scf.yield %lvr : f32
           }} else {{
             scf.yield %negbig : f32
           }}
           %gt = arith.cmpf ogt, %lv, %bv : f32
           %nv2 = arith.select %gt, %lv, %bv : f32
           %vi = arith.index_cast %v : index to i32
           %nx = arith.select %gt, %vi, %bx : i32
           scf.yield %nv2, %nx : f32, i32
         }}
         memref.store %bi#0, %PV[%m, %k] : {PVT}
         memref.store %bi#1, %PI[%m, %k] : {PIT}
       }}
       // reduce: best across pieces, ties to the lower vocabulary index
       %rd:2 = scf.for %k = %c0 to %ctasks_h step %c1
           iter_args(%bv = %negbig, %bx = %zero) -> (f32, i32) {{
         %pv = memref.load %PV[%m, %k] : {PVT}
         %pi = memref.load %PI[%m, %k] : {PIT}
         %gt = arith.cmpf ogt, %pv, %bv : f32
         %nv2 = arith.select %gt, %pv, %bv : f32
         %nx = arith.select %gt, %pi, %bx : i32
         scf.yield %nv2, %nx : f32, i32
       }}
       // A sequence has one next token: the argmax of its last position. The
       // earlier rows of a prefill chunk predict tokens the prompt already
       // supplies, and Fleet drops those too -- it only copies an output token
       // back into the stream when the slot is at or past the prompt
       // (persistent_kernel.cuh:396).
       %lastm = arith.subi %nat, %c1 : index
       %isLastRow = arith.cmpi eq, %m, %lastm : index
       scf.if %isLastRow {{
         %nxt = arith.addi %seq, %nat : index
         %past = arith.cmpi sge, %nxt, %cplen : index
         scf.if %past {{
           memref.store %rd#1, %Tok[%nxt] : {TKT}
         }}
       }}
     }}
     %seqn = arith.addi %seq, %nat : index
     scf.yield %seqn : index
    }}
    // The reference used these as scratch; hand the device zeroed copies.
    scf.for %m = %c0 to %ctok step %c1 {{
      scf.for %i = %c0 to %cdim step %c1 {{
        memref.store %fzero, %Rv[%m, %i] : {AT}
      }}
      scf.for %i = %c0 to %cqkvo step %c1 {{
        memref.store %fzero, %QKV[%m, %i] : {QT}
      }}
      scf.for %i = %c0 to %cqw step %c1 {{
        memref.store %fzero, %Av[%m, %i] : {QWT}
      }}
    }}
    // The reference appended its own k and v; the device must start from a
    // cache holding only the prefix.
    scf.for %l = %c0 to %clayers step %c1 {{
      scf.for %t = %c0 to %cwin step %c1 {{
        %tp = arith.addi %cpre, %t : index
        scf.for %hk = %c0 to %ckvh step %c1 {{
          scf.for %d = %c0 to %chd step %c1 {{
            memref.store %fzero, %Kc[%l, %tp, %hk, %d] : {KVT}
            memref.store %fzero, %Vc[%l, %tp, %hk, %d] : {KVT}
          }}
        }}
      }}
    }}
""")

    w(f"""
    %Q = memref.alloc() : {QUT}
    %cq = arith.constant {qslots} : index
    scf.for %sp = %c0 to %csteps step %c1 {{
      scf.for %l = %c0 to %clayers step %c1 {{
        scf.for %k = %c0 to %cq step %c1 {{
          memref.store %zero, %Q[%sp, %l, %k] : {QUT}
        }}
      }}
    }}
    // Two-level event counting (persistent_kernel.cuh:1226-1251): workers add
    // into a counter only their own die touches, and the last one out flushes
    // the die's whole share to the device counter.
    %Loc = memref.alloc() : memref<{locwords}xi32>
    %cloc = arith.constant {locwords} : index
    scf.for %i = %c0 to %cloc step %c1 {{
      memref.store %zero, %Loc[%i] : memref<{locwords}xi32>
    }}
    %E = memref.alloc() : memref<{events}xi32>
    %ce = arith.constant {events} : index
    scf.for %i = %c0 to %ce step %c1 {{
      memref.store %zero, %E[%i] : memref<{events}xi32>
    }}

    %dX = gpu.alloc () : {AT}
    %dRv = gpu.alloc () : {AT}
    %dQKV = gpu.alloc () : {QT}
    %dSc = gpu.alloc () : {SCT}
    %dAv = gpu.alloc () : {QWT}
    %dAov = gpu.alloc () : {AT}
    %dXa = gpu.alloc () : {AT}
    %dGU = gpu.alloc () : {GT}
    %dActv = gpu.alloc () : {IT}
    %dWqkv = gpu.alloc () : {WQTB}
    %dWo = gpu.alloc () : {WTB}
    %dWgu = gpu.alloc () : {WGTB}
    %dWd = gpu.alloc () : {WDTB}
    %dKc = gpu.alloc () : {KVT}
    %dVc = gpu.alloc () : {KVT}
    %dN1 = gpu.alloc () : {NT}
    %dN2 = gpu.alloc () : {NT}
    %dQKN = gpu.alloc () : {QKNT}
    %dRO = gpu.alloc () : {ROT}
    %dEmb = gpu.alloc () : {EMTB}
    %dWlm = gpu.alloc () : {LMTB}
    %dNf = gpu.alloc () : {NFT}
    %dTok = gpu.alloc () : {TKT}
    %dLg = gpu.alloc () : {LGT}
    %dPV = gpu.alloc () : {PVT}
    %dPI = gpu.alloc () : {PIT}
    %dQ = gpu.alloc () : {QUT}
    %dLoc = gpu.alloc () : memref<{locwords}xi32>
    %dE = gpu.alloc () : memref<{events}xi32>
    %dPlen = gpu.alloc () : {PLT}
    gpu.memcpy %dPlen, %Plen : {PLT}, {PLT}
    gpu.memcpy %dX, %X : {AT}, {AT}
    gpu.memcpy %dRv, %Rv : {AT}, {AT}
    gpu.memcpy %dQKV, %QKV : {QT}, {QT}
    gpu.memcpy %dSc, %Sc : {SCT}, {SCT}
    gpu.memcpy %dAv, %Av : {QWT}, {QWT}
    gpu.memcpy %dAov, %Aov : {AT}, {AT}
    gpu.memcpy %dXa, %Xa : {AT}, {AT}
    gpu.memcpy %dGU, %GU : {GT}, {GT}
    gpu.memcpy %dActv, %Actv : {IT}, {IT}
    gpu.memcpy %dWqkv, %WqkvB : {WQTB}, {WQTB}
    gpu.memcpy %dWo, %WoB : {WTB}, {WTB}
    gpu.memcpy %dWgu, %WguB : {WGTB}, {WGTB}
    gpu.memcpy %dWd, %WdB : {WDTB}, {WDTB}
    gpu.memcpy %dKc, %Kc : {KVT}, {KVT}
    gpu.memcpy %dVc, %Vc : {KVT}, {KVT}
    gpu.memcpy %dN1, %N1 : {NT}, {NT}
    gpu.memcpy %dN2, %N2 : {NT}, {NT}
    gpu.memcpy %dQKN, %QKN : {QKNT}, {QKNT}
    gpu.memcpy %dRO, %RO : {ROT}, {ROT}
    gpu.memcpy %dEmb, %EmbB : {EMTB}, {EMTB}
    gpu.memcpy %dWlm, %WlmB : {LMTB}, {LMTB}
    gpu.memcpy %dNf, %Nf : {NFT}, {NFT}
    gpu.memcpy %dLg, %Lg : {LGT}, {LGT}
    gpu.memcpy %dPV, %PV : {PVT}, {PVT}
    gpu.memcpy %dPI, %PI : {PIT}, {PIT}
    gpu.memcpy %dQ, %Q : {QUT}, {QUT}
    gpu.memcpy %dLoc, %Loc : memref<{locwords}xi32>, memref<{locwords}xi32>
    gpu.memcpy %dE, %E : memref<{events}xi32>, memref<{events}xi32>

    // --repeat runs the chain more than once so an external clock has
    // something to measure. Timing it from inside with mgpuEventRecord does not
    // work: the events go on the stream created here, while gpu-to-llvm puts
    // the kernel on one of its own, so mgpuEventElapsedTime returns
    // hipErrorInvalidHandle. The 4k GEMM test has the same problem.
    %crep = arith.constant {repeat} : index
    scf.for %rep = %c0 to %crep step %c1 {{
      scf.for %sp = %c0 to %csteps step %c1 {{
        scf.for %l = %c0 to %clayers step %c1 {{
          scf.for %k = %c0 to %cq step %c1 {{
            memref.store %zero, %Q[%sp, %l, %k] : {QUT}
          }}
        }}
      }}
      scf.for %i = %c0 to %ce step %c1 {{
        memref.store %zero, %E[%i] : memref<{events}xi32>
      }}
      scf.for %i = %c0 to %cloc step %c1 {{
        memref.store %zero, %Loc[%i] : memref<{locwords}xi32>
      }}
      gpu.memcpy %dQ, %Q : {QUT}, {QUT}
      gpu.memcpy %dLoc, %Loc : memref<{locwords}xi32>, memref<{locwords}xi32>
      gpu.memcpy %dE, %E : memref<{events}xi32>, memref<{events}xi32>
      // The chain rewrites x in place, and each token always writes the same
      // cache slot, so only x has to be restored.
      // Only the prompt is restored: the chain embeds it, and every later
      // token is produced on the device.
      scf.for %m = %c0 to %cmaxseq step %c1 {{
        %keep = arith.cmpi ult, %m, %cplen : index
        %cur = memref.load %Tok[%m] : {TKT}
        %nv = arith.select %keep, %cur, %zero : i32
        memref.store %nv, %Tok0[%m] : {TKT}
      }}
      gpu.memcpy %dTok, %Tok0 : {TKT}, {TKT}
      func.call @chain(%dQ, %dE, %dX, %dRv, %dQKV, %dSc, %dAv, %dAov, %dXa,
                       %dGU, %dActv, %dWqkv, %dWo, %dWgu, %dWd, %dKc, %dVc,
                       %dN1, %dN2, %dQKN, %dRO, %dEmb, %dWlm, %dNf, %dTok,
                       %dLg, %dPV, %dPI, %dLoc, %dPlen)
        : ({QUT}, memref<{events}xi32>, {AT}, {AT},
           {QT}, {SCT}, {QWT}, {AT}, {AT}, {GT}, {IT},
           {WQTB}, {WTB}, {WGTB}, {WDTB}, {KVT}, {KVT}, {NT}, {NT}, {QKNT}, {ROT},
           {EMTB}, {LMTB}, {NFT}, {TKT}, {LGT}, {PVT}, {PIT},
           memref<{locwords}xi32>, {PLT}) -> ()
    }}

    gpu.memcpy %X, %dX : {AT}, {AT}

    %tol = arith.constant 2.0e-2 : f32
    %bad = scf.for %m = %c0 to %ctok step %c1
        iter_args(%bo = %zero) -> (i32) {{
      %bi = scf.for %i = %c0 to %cdim step %c1
          iter_args(%b = %bo) -> (i32) {{
        %got = memref.load %X[%m, %i] : {AT}
        %want = memref.load %ref[%m, %i] : {AT}
        %d = arith.subf %got, %want : f32
        %ad = math.absf %d : f32
        %aw = math.absf %want : f32
        %sc = arith.maxnumf %aw, %fone : f32
        %rel = arith.divf %ad, %sc : f32
        %ok = arith.cmpf ole, %rel, %tol : f32
        %inc = arith.select %ok, %zero, %one : i32
        %b2 = arith.addi %b, %inc : i32
        scf.yield %b2 : i32
      }}
      scf.yield %bi : i32
    }}
    vector.print str "layers = "
    %nl = arith.constant {layers} : i32
    vector.print %nl : i32
    vector.print str "tokens = "
    %ntk = arith.constant {tokens} : i32
    vector.print %ntk : i32
    vector.print str "steps = "
    %nst = arith.constant {steps} : i32
    vector.print %nst : i32
    vector.print str "query heads = "
    %nh = arith.constant {heads} : i32
    vector.print %nh : i32
    gpu.memcpy %Loc, %dLoc : memref<{locwords}xi32>, memref<{locwords}xi32>
{timer_report}{flush_report}
    vector.print str "output elements differing from the reference = "
    vector.print %bad : i32
    // The token check is exact, not a tolerance: argmax turns the whole chain
    // into a discrete answer, so a token either matches or it does not.
    gpu.memcpy %Tok0, %dTok : {TKT}, {TKT}
    // Only the generated tail is compared: the prompt was handed to both sides.
    %tbad = scf.for %m = %cplen to %cmaxseq step %c1
        iter_args(%b = %zero) -> (i32) {{
      %g = memref.load %Tok0[%m] : {TKT}
      %wt = memref.load %Tok[%m] : {TKT}
      %eq = arith.cmpi eq, %g, %wt : i32
      %inc = arith.select %eq, %zero, %one : i32
      %b2 = arith.addi %b, %inc : i32
      scf.yield %b2 : i32
    }}
    // The logits are where the vocabulary tail is actually checked. The token
    // on top of them is a much blunter instrument: this chain is contractive,
    // the hidden state varies by tens of percent between steps, and argmax over
    // a vocabulary needs more than that to move -- so a token can match while
    // the logits under it are wrong. The float comparison catches that; the
    // token comparison catches the argmax stages on top.
    gpu.memcpy %LgD, %dLg : {LGT}, {LGT}
    %lbad = scf.for %m = %c0 to %ctok step %c1
        iter_args(%bo = %zero) -> (i32) {{
      %bi = scf.for %v = %c0 to %cvocab step %c1
          iter_args(%b = %bo) -> (i32) {{
        %g = memref.load %LgD[%m, %v] : {LGT}
        %wv = memref.load %Lg[%m, %v] : {LGT}
        %d = arith.subf %g, %wv : f32
        %ad = math.absf %d : f32
        %aw = math.absf %wv : f32
        %scl = arith.maxnumf %aw, %fone : f32
        %rel = arith.divf %ad, %scl : f32
        %ok = arith.cmpf ole, %rel, %tol : f32
        %inc = arith.select %ok, %zero, %one : i32
        %b2 = arith.addi %b, %inc : i32
        scf.yield %b2 : i32
      }}
      scf.yield %bi : i32
    }}
    vector.print str "logit elements differing from the reference = "
    vector.print %lbad : i32
    // Print the reference token ids so an outside model can check them: the
    // device agreeing with the host reference only proves they match.
    vector.print str "reference tokens:"
    scf.for %m = %cplen to %cmaxseq step %c1 {{
      %tv = memref.load %Tok[%m] : {TKT}
      vector.print %tv : i32
    }}
    vector.print str "tokens differing from the reference = "
    vector.print %tbad : i32
    %tot0 = arith.addi %bad, %tbad : i32
    %tot = arith.addi %tot0, %lbad : i32
    vector.print str "total differences = "
    vector.print %tot : i32
    return
  }}
""")

    # Only the packed form needs these: the arrival bit that lives in the

    # top half of the word, and the shift that reads it back.

    packconsts = (
        ""
        if not pack_arrival
        else "        %eightL = arith.constant 8 : i64\n"
        "        %arrbit = arith.constant 4294967296 : i64\n"
        "        %c32L = arith.constant 32 : i64\n"
    )

    layer_consts = "\n".join(
        f"        %L{i} = arith.constant {i} : index" for i in range(layers)
    )
    w(f"""
  func.func @chain(%Q: {QUT}, %E: memref<{events}xi32>,
                   %X: {AT}, %Rv: {AT}, %QKV: {QT}, %Sc: {SCT},
                   %Av: {QWT}, %Aov: {AT}, %Xa: {AT}, %GU: {GT}, %Actv: {IT},
                   %Wqkv: {WQTB}, %Wo: {WTB}, %Wgu: {WGTB}, %Wd: {WDTB},
                   %Kc: {KVT}, %Vc: {KVT}, %N1: {NT}, %N2: {NT},
                   %QKN: {QKNT}, %RO: {ROT}, %Emb: {EMTB}, %Wlm: {LMTB},
                   %Nf: {NFT}, %Tok: {TKT}, %Lg: {LGT}, %PV: {PVT},
                   %PI: {PIT}, %Loc: memref<{locwords}xi32>, %Pl: {PLT}) {{
    %c1 = arith.constant 1 : index
    %cw = arith.constant {workers} : index
    air.launch (%bx, %by) in (%nbx=%cw, %nby=%c1)
        args(%q=%Q, %eb=%E, %x=%X, %r=%Rv, %qkv=%QKV, %scb=%Sc, %av=%Av,
             %aov=%Aov, %xab=%Xa, %gub=%GU, %actb=%Actv, %wqkv=%Wqkv,
             %wo=%Wo, %wgu=%Wgu, %wd=%Wd, %kc=%Kc, %vc=%Vc, %n1=%N1,
             %n2=%N2, %qkn=%QKN, %ro=%RO, %emb=%Emb, %wlm=%Wlm, %nf=%Nf,
             %tok=%Tok, %lg=%Lg, %pvb=%PV, %pib=%PI, %loc=%Loc, %plb=%Pl)
        : {QUT}, memref<{events}xi32>, {AT}, {AT},
          {QT}, {SCT}, {QWT}, {AT}, {AT}, {GT}, {IT},
          {WQTB}, {WTB}, {WGTB}, {WDTB}, {KVT}, {KVT}, {NT}, {NT}, {QKNT}, {ROT},
          {EMTB}, {LMTB}, {NFT}, {TKT}, {LGT}, {PVT}, {PIT},
          memref<{locwords}xi32>, {PLT} {{
      air.segment @worker args(%sq=%q, %se=%eb, %sx=%x, %sr=%r, %sqkv=%qkv,
                               %ssc=%scb, %sav=%av, %saov=%aov, %sxa=%xab,
                               %sgu=%gub, %sact=%actb, %swqkv=%wqkv,
                               %swo=%wo, %swgu=%wgu, %swd=%wd, %skc=%kc,
                               %svc=%vc, %sn1=%n1, %sn2=%n2, %sqkn=%qkn,
                               %sro=%ro, %semb=%emb, %swlm=%wlm, %snf=%nf,
                               %stok=%tok, %slg=%lg, %spv=%pvb, %spi=%pib,
                               %sloc=%loc, %splen=%plb)
          : {QUT}, memref<{events}xi32>, {AT}, {AT},
            {QT}, {SCT}, {QWT}, {AT}, {AT}, {GT}, {IT},
            {WQTB}, {WTB}, {WGTB}, {WDTB}, {KVT}, {KVT}, {NT}, {NT}, {QKNT}, {ROT},
            {EMTB}, {LMTB}, {NFT}, {TKT}, {LGT}, {PVT}, {PIT},
            memref<{locwords}xi32>, {PLT} {{
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %ctasks = arith.constant {tasks} : index
        %ctasksD = arith.constant {tasks_d} : index
        %ctasksV = arith.constant {tasks_v} : index
        %csliceDW = arith.constant {slice_dw} : index
        %csliceD = arith.constant {slice_d} : index
        %csliceI = arith.constant {slice_i} : index
        %cslice2I = arith.constant {slice_2i} : index
        %csliceQ = arith.constant {slice_q} : index
        %cdim_s = arith.constant {dim} : index
        %cinter_s = arith.constant {inter} : index
        %chd_s = arith.constant {hd} : index
        %ch2_s = arith.constant {h2} : index
        %cheads_s = arith.constant {heads} : index
        %ckvh_s = arith.constant {kv_heads} : index
        %cgroup_s = arith.constant {group} : index
        %cpre_s = arith.constant {cache} : index
        %ctotal_s = arith.constant {total} : index
        %csteps_s = arith.constant {steps} : index
        %cevstep = arith.constant {4 * per_step} : i64
        %clocstep = arith.constant {4 * per_step * maxdies} : i64
        %cvocab_s = arith.constant {vocab} : index
        %csliceV = arith.constant {slice_v} : index
        %negbigI = arith.constant -2147483648 : i32
        %cqw_s = arith.constant {qw} : index
        %ckbase = arith.constant {heads * hd} : index
        %cvbase = arith.constant {(heads + kv_heads) * hd} : index
        %one_s = arith.constant 1 : i32
        %zero_s = arith.constant 0 : i32
        %ctok_s = arith.constant {tokens} : index
        %ctasks_i = arith.constant {tasks} : i32
        %ctasksD_i = arith.constant {tasks_d} : i32
        %ctasksV_i = arith.constant {tasks_v} : i32
        %cheads_i = arith.constant {heads} : i32
        %n1_s = arith.constant 1 : i32
        %fzero_s = arith.constant 0.0 : f32
        %fone_s = arith.constant 1.0 : f32
        %eps_s = arith.constant 1.0e-6 : f32
        %fdim_s = arith.constant {float(dim):.6e} : f32
        %fhd_s = arith.constant {float(hd):.6e} : f32
        %invsqrthd_s = arith.constant {invsqrthd:.8e} : f32
        %negbig_s = arith.constant -1.000000e30 : f32
        %true = arith.constant true
        %false = arith.constant false
        // Stands for "the scan found nothing". It is fed through the same
        // claim path as a real claim so that the workgroup reaches the same
        // out-of-range test either way; see strided_stage.
        %nopiece_s = arith.constant {1 << 24} : i32
        %cmaxm1 = arith.constant {maxdies - 1} : index

        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index
        // %tx_s is the lane, and %nlane the wave, because the block is exactly
        // one wavefront: air.herd at the bottom is 1x1 and the AIR compute
        // model makes one herd tile one wavefront (AIRToROCDLPass.cpp:1066),
        // so blockDim.x == wave size. The protocol below leans on that --
        // rocdl.readfirstlane broadcasts within a wave, and only because the
        // wave is the whole block does that reach every thread that claimed
        // nothing. A wider herd would need the broadcast to go through LDS.
        //
        // A constant rather than gpu.block_dim x so that this and the width of
        // the butterfly reduction cannot drift apart: see the note on `wave`
        // at the top of emit().
        %nlane = arith.constant {wave} : index
        %cwaves = arith.constant {waves} : index
        %nthr = arith.constant {nthreads} : index
        %wid = arith.divui %tx_s, %nlane : index
        %lid = arith.remui %tx_s, %nlane : index
        %isW0 = arith.cmpi eq, %wid, %c0_s : index
        // Lane 0 of *this* wave, as opposed to thread 0 of the workgroup. The
        // event spins are per wave -- every wave has to see the counter move,
        // and rocdl.readfirstlane only ever reaches the wave it ran in -- so a
        // spin guarded by %isLead would leave every wave but the first reading
        // the else value forever.
        %isL0 = arith.cmpi eq, %lid, %c0_s : index
{lds_handles}
        // Which die this workgroup is on. Work is claimed per die so that the
        // slices a die computes are the ones its own cache is holding.
        %mydie_raw = air.chiplet_id
        %cmaxdies = arith.constant {maxdies} : index
        %mydie = arith.remui %mydie_raw, %cmaxdies : index
{layer_consts}
        %locbase = memref.extract_aligned_pointer_as_index %sloc : memref<{locwords}xi32> -> index
        %locbi = arith.index_cast %locbase : index to i64
        %locp = llvm.inttoptr %locbi : i64 to !llvm.ptr
        %fourL = arith.constant 4 : i64
{packconsts}        %cflushW = arith.constant {4 * flushword} : i64
        %flushP = llvm.getelementptr %locp[%cflushW] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %myrank2 = air.chiplet_block_id
{timer_id}
        %mycnt = air.chiplet_dim_blocks
{timer_launch0}
        %mycnt_i = arith.index_cast %mycnt : index to i32
        %evbase = memref.extract_aligned_pointer_as_index %se : memref<{events}xi32> -> index
        %evi = arith.index_cast %evbase : index to i64
        %evptr = llvm.inttoptr %evi : i64 to !llvm.ptr

        scf.execute_region {{
          // Every lane runs this, not just the lead one. The scheduler is
          // still per workgroup -- the claims and the signals below are taken
          // by the lead lane and broadcast -- but the task bodies are the
          // wave's work, which is the only way an MFMA or an LDS staging step
          // could ever be reached: both are wavefront instructions.
          //
          // One launch runs the whole decode. The task graph is the same every
          // step; what changes is the iteration it belongs to. Fleet versions
          // task identity the same way -- TaskId is
          // (iteration_num << 32) | position_index
          // (persistent_kernel.cuh:263) -- so a step gets fresh event and queue
          // slots without anything having to be reset between steps.
          %seqend = scf.for %step = %c0_s to %csteps_s step %c1_s
              iter_args(%seq = %c0_s) -> (index) {{
            %stepi = arith.index_cast %step : index to i64
            %stepev = arith.muli %stepi, %cevstep : i64
            %steploc = arith.muli %stepi, %clocstep : i64
            // How many tokens this step actually carries. It is read from a
            // buffer, not folded from %step, because that is the whole point:
            // one static task graph serves a prefill chunk of `tokens` and a
            // decode step of 1, exactly as Fleet's does -- prepare_next_batch
            // writes qo_indptr_buffer on the device and the tasks read their
            // trip counts out of it (persistent_kernel.cuh:441-450,
            // multitoken_paged_attention_mfma_mi300.cuh:78-83).
            %plv = memref.load %splen[%c0_s] : {PLT}
            %plen_s = arith.index_cast %plv : i32 to index
            %rem_s = arith.subi %plen_s, %seq : index
            %isPre_s = arith.cmpi sgt, %rem_s, %c0_s : index
            %capped_s = arith.minsi %rem_s, %ctok_s : index
            %nat = arith.select %isPre_s, %capped_s, %c1_s : index
            %wbase = arith.addi %cpre_s, %seq : index
            %curlen = arith.addi %wbase, %nat : index
            // How many pieces a stage has this step. The grid is (token, n) and
            // the token extent is this step's active count, not the window
            // width: at a decode step those differ by a factor of `tokens`, and
            // every piece that exists costs a returning atomic on a counter the
            // whole device shares whether or not it has anything to do. A
            // window of 5 decoding one token was claiming five pieces for every
            // one that computed.
            //
            // This is also the shape Fleet has. Its task graph is static in the
            // sense that matters -- the set of tasks per layer is fixed at
            // build time -- but a task is an output tile and the token loop
            // lives inside it with a trip count read at runtime from
            // qo_indptr_buffer (gang_linear_mi300.cuh:51,
            // multitoken_paged_attention_mfma_mi300.cuh:78-83). Here the token
            // is part of the piece id, which is what makes the M-major sweep
            // expressible; making its extent runtime keeps that and drops the
            // pieces that were never going to do anything.
            %nat_i = arith.index_cast %nat : index to i32
            %ntasks_t = arith.muli %nat_i, %ctasks_i : i32
            %ntasksD_t = arith.muli %nat_i, %ctasksD_i : i32
            %ntasksV_t = arith.muli %nat_i, %ctasksV_i : i32
            %nheads_t = arith.muli %nat_i, %cheads_i : i32""")

    # The event wait, and who pays for it.
    #
    # Every wave polling the same counter is eight times the traffic on the one
    # cache line the whole device is contending for, and on gfx9 an acquire
    # load is `global_load` + `s_waitcnt` + `buffer_inv sc0 sc1` -- which
    # invalidates the CU's vector cache *and* the XCD's L2. Doing that at full
    # issue rate from 1024 waves leaves the L2 permanently cold.
    #
    # So one wave waits and a barrier releases the workgroup, which it was
    # going to meet at before the next claim anyway; and the poll is relaxed
    # with a single acquire fence once the wait is over, which is the standard
    # idiom and the shape Fleet's own loop has -- __ATOMIC_RELAXED inside, an
    # acquire fence outside (persistent_kernel.cuh:944-966).
    #
    # That fence used to be taken by every wave, and it was 7.42 of the 12.28
    # us a stage boundary then cost -- 60% of the boundary, and 1.9 of the
    # 5.44 ms a token then took. `acquire_once`, the default, has the one wave
    # that waited take it inside the `scf.if` it waited in, before the barrier
    # that releases the other seven. Whole model, 28 layers, 128 workers, six
    # steps, minimum of two runs: 3 265 096 launch ticks to 2 209 060, **1.48x**.
    #
    # That is sound because every wave of a workgroup is on one CU and one XCD,
    # so one wave's `buffer_inv sc0 sc1` empties the vector cache and the L2
    # that all of them read through, and the barrier orders the rest of the
    # workgroup's loads after it. A line refilled in between can only be fresh:
    # the producers wrote back past L2 before they signalled, so anyone
    # fetching that address afterwards gets the new value.
    #
    # It is believed because this is the shape megakernel_gen_contend exists to
    # catch -- a workgroup whose waves disagree about memory -- and the bug
    # that test was written for passed 3/3 at 128 workers and failed 3/3 at
    # 256. So: the suite 21/21, the contending shape five more times at 256
    # workers with `total differences = 0` every time, and the model at
    # 256/256, 512/128, 128/128 and 64/64 all agreeing with numpy token for
    # token. `--acquire-per-wave` restores the old form byte for byte.
    #
    # `acquire_agent` asks for `buffer_inv sc1` rather than `buffer_inv
    # sc0 sc1`, which is the scope the protocol actually needs -- every
    # workgroup here is on one device, and system scope additionally orders
    # against the host, which nothing in a decode step does. It measures as
    # nothing, both ways round: 3 262 472 against a 3 265 096 baseline per
    # wave, and 2 215 288 against 2 214 204 with `acquire_once` on. Kept
    # because it costs nothing to keep and says what the right scope is.
    acq = (
        f'          llvm.fence syncscope("{"agent" if acquire_agent else ""}") acquire'
    )
    if waves == 1:
        spin_open, spin_close, spin_end = "", "", ""
        spin_order = "acquire"
    elif acquire_once:
        spin_open = "          scf.if %isW0 {"
        spin_close = "  " + acq + "\n          }"
        spin_end = "          gpu.barrier"
        spin_order = "monotonic"
    else:
        spin_open = "          scf.if %isW0 {"
        spin_close = "          }"
        spin_end = "          gpu.barrier\n" + acq
        spin_order = "monotonic"
    # What --pad-strip 1 leaves behind: the same rendezvous with no fence at
    # all, wherever the fence would have been.
    spin_close_nf = "          }"
    spin_end_nf = "          gpu.barrier"

    # What the waiting wave does between polls.
    #
    # The poll is a system-scope load, which on gfx9 is `global_load_dword
    # ... sc0 sc1`: it may not be answered from this CU's vector cache or this
    # XCD's L2, because the workgroup that will set the word is on another XCD
    # and the L2s are not coherent with each other. So every poll is a memory
    # transaction, and with the loop as tight as the hardware will run it,
    # 128 workgroups are issuing them continuously against one cache line.
    #
    # The rendezvous is 2.65 us of a 5.52 us stage boundary, 0.68 of the
    # 3.68 ms a token then took, and a single uncached read does not cost
    # 2.65 us.
    # `s_sleep n` idles the wave for n*64 clocks -- 30 ns a unit at 2.1 GHz --
    # which is small against one poll's latency and large against the interval
    # between polls, so it trades detection latency the loop was not using for
    # traffic on the line everyone is watching.
    #
    # It buys 0.8%, and that is the useful part of the result. Swept at 128
    # workers, minimum of two runs a point, the two runs of a point never
    # more than 0.31% apart:
    #
    #   sleep      0       1       2       4       8      16
    #   vs 0    0.00%  +0.56%  +0.26%  -0.11%  -0.67%  -0.80%
    #
    # If the rendezvous were 128 workgroups queueing behind one cache line,
    # cutting the traffic by a factor of sixteen would not be worth 0.8%. So
    # it is not congestion, it is **latency**: the interval between the last
    # worker's release landing and a poller's next read seeing it, which no
    # amount of polling harder can shorten. 16 is kept because it is free and
    # measured, but the way to spend 0.68 ms/token of rendezvous is to have
    # fewer of them, not to poll them better.
    spin_wait = "" if not spin_sleep else f"            rocdl.s.sleep {spin_sleep}\n"

    # Emitted before every event signal once the workgroup is more than one
    # wave; see the note at the signal site.
    #
    # KNOWN HOLE, measured but not yet closed. `gpu.barrier` is `s_barrier`
    # preceded by at most `s_waitcnt lgkmcnt(0)` -- checked in the emitted ISA,
    # where most of them carry no `s_waitcnt` at all and none carries `vmcnt`.
    # So the barrier orders LDS and control flow and says nothing about whether
    # a wave's global stores have left the wave, while the signal below is one
    # thread's release and `vmcnt` is per wave. A workgroup can therefore
    # announce "my piece is readable" with seven waves' stores still in flight.
    # The window is short -- the stores were issued before the barrier and the
    # reader has an event, a barrier and a fence to get through -- and it has
    # never been caught, but it is not closed by anything here.
    #
    # Closing it with an agent-scope release fence in every thread works and
    # cost 17.5% of the device ticks at 128/128 when it was tried (rope and
    # attention both doubled), because agent scope on this part writes back L2
    # per wave. A workgroup-scope fence should emit the `s_waitcnt vmcnt(0)`
    # and not the writeback, which is all that is needed here -- the lead
    # thread's system-scope release below already does the L2 flush. That is
    # still the thing to measure.
    #
    # Two cautions for whoever does. The 17.5% is from the build that took the
    # acquire fence in every wave, which was 1.48x slower overall, so it is a
    # share of a different program and the agent-scope variant has to be
    # re-measured rather than compared against. And this is the writer's half
    # of the protocol, independent of `acquire_once` above, which changed only
    # the reader's -- fixing one says nothing about the other.
    stage_bar = "" if waves == 1 else "          gpu.barrier"

    def lane_reduce(src, dst, op, tag, indent, stride, steps, ty="f32"):
        """Butterfly over the lanes `stride` apart -- 2**steps of them.

        wave_reduce is the stride-1 case. Used where a wave's lanes are split
        two ways, some across output columns and the rest across the reduction:
        the lanes sharing a column are the ones differing only in the bits above
        log2(stride), so xor-ing exactly those bits adds their partials up and
        leaves every one of them holding the total.
        """
        pad = " " * indent
        if steps == 0:
            return f"{pad}{dst} = arith.addf {src}, %fzero_s : {ty}"
        out, cur = [], src
        for k in range(steps):
            nxt = dst if k == steps - 1 else f"%lr{tag}{k}"
            out.append(
                f"{pad}%lro{tag}{k} = arith.constant {stride << k} : i32\n"
                f"{pad}%lrw{tag}{k} = arith.constant {wave} : i32\n"
                f"{pad}%lrs{tag}{k}, %lrp{tag}{k} = gpu.shuffle xor {cur}, "
                f"%lro{tag}{k}, %lrw{tag}{k} : {ty}\n"
                f"{pad}{nxt} = {op} {cur}, %lrs{tag}{k} : {ty}"
            )
            cur = nxt
        return "\n".join(out)

    # The running end of the chain: the clock value that the next stage will
    # call its own start. Reset per generated program, threaded by emission
    # order -- every stage string is built exactly once and in the order it
    # appears in the step body, so the name is always already in scope.
    tprev = [None]

    def timer_begin(l, stage):
        """Only the first stage of the step body reads the clock.

        Every later stage starts where the one before it ended, and that is
        the whole point. A window needs both of its edges nailed to the
        program, and only one of the two was: the end read sits immediately
        after the stage's `gpu.barrier` and `llvm.fence`, which the backend
        will not schedule across, while a start read at the top of a stage has
        nothing around it but address arithmetic and sinks into the body it is
        supposed to be measuring. Two independent reads per stage therefore
        measured an interval strictly inside the stage, and the shortfall did
        not go anywhere -- it was simply not attributed. Summed over the 257
        stage instances of a step it came to 41% of the launch.

        Chaining removes the question. Consecutive windows share an edge, so
        the classes telescope: whatever the total is, it equals the last read
        minus the first, however the scheduler places the reads in between. A
        read that drifts now moves time from one class to its neighbour
        instead of deleting it, which is a bias that shows up as a suspicious
        class rather than as a missing 41%. It also halves the number of
        clock reads.
        """
        if not timers or not stage_timers:
            return ""
        if tprev[0] is not None:
            return ""
        tprev[0] = f"%tb{l}_{stage}"
        return (
            f"          %tb{l}_{stage} = llvm.call_intrinsic "
            f'"llvm.amdgcn.s.memrealtime"() : () -> i64'
        )

    def timer_mid(l, stage):
        """Between the body and the rendezvous.

        A stage does not end when this workgroup finishes its pieces; it ends
        when the slowest of the 128 does. Splitting here separates the work
        from the wait, which is the difference between "the GEMM is slow" and
        "the GEMM is fine and the stage is bounded by a straggler".
        """
        if not timers or not stage_timers:
            return ""
        return (
            f"          %tm{l}_{stage} = llvm.call_intrinsic "
            f'"llvm.amdgcn.s.memrealtime"() : () -> i64'
        )

    def timer_end(l, stage):
        if not timers or not stage_timers:
            return ""
        beg = tprev[0]
        assert beg is not None, (
            f"stage ({l}, {stage}) ended a timing window that never started -- "
            "timer_begin must be interpolated before timer_end in the same "
            "stage template, and stage strings must be built in emission order"
        )
        tprev[0] = f"%te{l}_{stage}"
        return f"""          %te{l}_{stage} = llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"() : () -> i64
          %dw{l}_{stage} = arith.subi %te{l}_{stage}, %tm{l}_{stage} : i64
          %dwi{l}_{stage} = arith.trunci %dw{l}_{stage} : i64 to i32
          %dt{l}_{stage} = arith.subi %tm{l}_{stage}, {beg} : i64
          %dti{l}_{stage} = arith.trunci %dt{l}_{stage} : i64 to i32
          %isT{l}_{stage} = arith.andi %isWG0, %isLead : i1
          scf.if %isT{l}_{stage} {{
            %tc{l}_{stage} = arith.constant {timerbase + stage} : index
            %to{l}_{stage} = memref.load %sloc[%tc{l}_{stage}] : memref<{locwords}xi32>
            %tn{l}_{stage} = arith.addi %to{l}_{stage}, %dti{l}_{stage} : i32
            memref.store %tn{l}_{stage}, %sloc[%tc{l}_{stage}] : memref<{locwords}xi32>
            %wc{l}_{stage} = arith.constant {timerbase + nclass // 2 + stage} : index
            %wo{l}_{stage} = memref.load %sloc[%wc{l}_{stage}] : memref<{locwords}xi32>
            %wn{l}_{stage} = arith.addi %wo{l}_{stage}, %dwi{l}_{stage} : i32
            memref.store %wn{l}_{stage}, %sloc[%wc{l}_{stage}] : memref<{locwords}xi32>
          }}"""

    def wave_reduce(src, dst, op, tag, indent, ty="f32"):
        """Combine one value per lane into one value the whole wave has.

        gpu.subgroup_reduce says this in a single op, but nothing in the
        pipeline lowers it -- the patterns exist only behind
        `--test-gpu-subgroup-reduce-lowering` -- while convert-gpu-to-rocdl
        does lower gpu.shuffle. So the butterfly is written out: log2(wave)
        exchanges, each lane adding what it got from the lane one bit away.

        Every lane ends with the total, which is the property the callers use:
        the divisor of an rmsnorm and the max of a softmax are needed by all of
        them, and this way none of it has to be broadcast afterwards.
        """
        pad = " " * indent
        out = []
        cur = src
        for k in range(wave_steps):
            nxt = dst if k == wave_steps - 1 else f"%wv{tag}{k}"
            out.append(
                f"{pad}%wo{tag}{k} = arith.constant {1 << k} : i32\n"
                f"{pad}%ww{tag}{k} = arith.constant {wave} : i32\n"
                f"{pad}%ws{tag}{k}, %wp{tag}{k} = gpu.shuffle xor {cur}, "
                f"%wo{tag}{k}, %ww{tag}{k} : {ty}\n"
                f"{pad}{nxt} = {op} {cur}, %ws{tag}{k} : {ty}"
            )
            cur = nxt
        return "\n".join(out)

    def block_sum(src, dst, tag, indent):
        """Combine one value per *thread* into one value the whole block has.

        wave_reduce first, so what crosses waves is one number per wave
        rather than one per lane, then a slot each in LDS and every wave adds
        the same `waves` of them. Reading the slots redundantly is cheaper
        than reducing them on one wave and broadcasting: it is `waves` LDS
        reads either way, and this way there is no third barrier.
        """
        pad = " " * indent
        if waves == 1:
            return wave_reduce(src, dst, "arith.addf", tag, indent)
        return (
            f"{wave_reduce(src, '%bw' + tag, 'arith.addf', tag, indent)}\n"
            f"{pad}gpu.barrier\n"
            f"{pad}scf.if %isL0 {{\n"
            f"{pad}  memref.store %bw{tag}, %ldsr[%wid] : "
            f"{RT}\n"
            f"{pad}}}\n"
            f"{pad}gpu.barrier\n"
            f"{pad}{dst} = scf.for %bi{tag} = %c0_s to %cwaves step %c1_s\n"
            f"{pad}    iter_args(%ba{tag} = %fzero_s) -> (f32) {{\n"
            f"{pad}  %bv{tag} = memref.load %ldsr[%bi{tag}] : "
            f"{RT}\n"
            f"{pad}  %bn{tag} = arith.addf %ba{tag}, %bv{tag} : f32\n"
            f"{pad}  scf.yield %bn{tag} : f32\n"
            f"{pad}}}"
        )

    # A stage whose work splits into independent pieces. Each die has its own
    # head and its own stride of pieces, so what a die touches is what its cache
    # already holds. A die that runs out steals from the others, which keeps
    # this correct when the dispatcher does not use every die: locality is a
    # preference here, not an assumption.
    #
    # With `tokens` > 1 a piece is an (m, n) pair and the order matters. Fleet
    # sweeps M fast and N slow (gang_linear_mi300.cuh:75-76, full-M-major so
    # its window W is m_tiles):
    #
    #     m_tile = local % win_h        n_tile = local / win_h
    #
    # so `win_h` consecutive claims land on the same n -- the same block of
    # weight rows -- with different tokens. The block is read into the die's
    # cache once and serves all of them. Applied here to the die's own claim
    # counter rather than a global tile id, because that counter is what a
    # die's workgroups share. At tokens == 1 this is m = 0, n = k.
    def single_claim(l, stage, lc):
        """Which one workgroup runs a stage that cannot be split.

        The dynamic form races for it: every workgroup takes an atomic on one
        counter and the one that reads zero wins, then broadcasts the answer to
        its own waves. That is `workers` atomics on a single address and two
        barriers each, to decide something that has no inputs -- so under the
        static claim the answer is simply "the first workgroup of the first
        chiplet", which every thread already knows.
        """
        if static_claim:
            return (
                f"          %d0{l}_{stage} = arith.cmpi eq, %mydie, %c0_s : index\n"
                f"          %r0{l}_{stage} = arith.cmpi eq, %myrank2, %c0_s : index\n"
                f"          %mine{l}_{stage} = arith.andi %d0{l}_{stage}, %r0{l}_{stage} : i1"
            )
        return (
            f"          %cll{l}_{stage} = scf.if %isLead -> (i32) {{\n"
            f"            %a = memref.atomic_rmw addi %one_s, %sq[%step, {lc}, "
            f"%hsingle{stage}_{l}] : (i32, {QUT}) -> i32\n"
            f"            scf.yield %a : i32\n"
            f"          }} else {{\n"
            f"            scf.yield %zero_s : i32\n"
            f"          }}\n"
            + bcast_i32(f"%cll{l}_{stage}", f"%cl{l}_{stage}", f"s{l}_{stage}", 10)
            + f"\n          %mine{l}_{stage} = arith.cmpi eq, %cl{l}_{stage}, %zero_s : i32"
        )

    # The two ways a workgroup can find out which pieces are its. Kept as
    # templates rather than branches inside strided_stage so the two are
    # readable side by side: one is a protocol, the other is arithmetic.
    DYNAMIC_CLAIM = """          %t{l}_{stage}:2 = scf.while (%go = %true, %outer = %zero_s) : (i1, i32) -> (i1, i32) {{
            scf.condition(%go) %go, %outer : i1, i32
          }} do {{
          ^bb0(%g: i1, %acc: i32):
              // One claim for the whole workgroup, and the entire search for it
              // is the lead thread's -- it reads the drained flags, takes the
              // atomics and comes back with a queue and a piece, or with
              // nothing. Letting every lane claim would be a different
              // scheduler: the queue counts workgroups and so does the
              // two-level event flush below, which arrives once per workgroup
              // and compares against air.chiplet_dim_blocks.
              //
              // The search being one thread's is not only about the count. The
              // flags are written by other workgroups while this one reads
              // them, so a plain load of one is not workgroup-uniform however
              // uniform its address is: two waves can read it either side of
              // the write and disagree. Branching on that disagreement was a
              // real bug -- the branch led to the broadcast below, which is two
              // gpu.barriers, so the waves of one workgroup met different
              // barriers and read each other's claims. Measured: 28 layers at
              // 256 workers came out wrong three runs out of three, and right
              // three out of three with the flag test removed. Here nothing
              // outside the lead thread ever looks at a flag, and every wave
              // reaches the same barriers because there is no longer a branch
              // between them.
              %sc{l}_{stage}:2 = scf.if %isLead -> (i32, i32) {{
                %fnd:3 = scf.for %pp = %c0_s to %cmaxdies step %c1_s
                    iter_args(%got = %false, %gd = %zero_s, %gk = %nopiece_s) -> (i1, i32, i32) {{
                  %step1:3 = scf.if %got -> (i1, i32, i32) {{
                    scf.yield %got, %gd, %gk : i1, i32, i32
                  }} else {{
                    // Own die first, then the others in order.
                    %draw = arith.addi %mydie, %pp : index
                    %ds = arith.remui %draw, %cmaxdies : index
                    %d2 = arith.muli %ds, %c2_s : index
                    %hidx = arith.addi %hb{l}_{stage}, %d2 : index
                    %fidx = arith.addi %hidx, %c1_s : index
                    // Skipping a drained queue costs one load, against an
                    // atomic and a trip round the broadcast to discover the
                    // same thing.
                    %flg = memref.load %sq[%step, {lc}, %fidx] : {QUT}
                    %drained = arith.cmpi ne, %flg, %zero_s : i32
                    %step2:3 = scf.if %drained -> (i1, i32, i32) {{
                      scf.yield %false, %zero_s, %nopiece_s : i1, i32, i32
                    }} else {{
                      %a = memref.atomic_rmw addi %one_s, %sq[%step, {lc}, %hidx] : (i32, {QUT}) -> i32
                      %ka = arith.index_cast %a : i32 to index
                      %isN1a = arith.cmpi eq, %nat, %c1_s : index
                      %kna = scf.if %isN1a -> (index) {{
                        scf.yield %ka : index
                      }} else {{
                        %kda = arith.divui %ka, %nat : index
                        scf.yield %kda : index
                      }}
                      %ksa = arith.muli %kna, %cmaxdies : index
                      %ixa = arith.addi %ds, %ksa : index
                      %hasa = arith.cmpi ult, %ixa, {count_expr} : index
                      %step3:3 = scf.if %hasa -> (i1, i32, i32) {{
                        %dsi = arith.index_cast %ds : index to i32
                        scf.yield %true, %dsi, %a : i1, i32, i32
                      }} else {{
                        // Last one out says so, so the next workgroup round
                        // reads a word instead of taking an atomic.
                        memref.store %one_s, %sq[%step, {lc}, %fidx] : {QUT}
                        scf.yield %false, %zero_s, %nopiece_s : i1, i32, i32
                      }}
                      scf.yield %step3#0, %step3#1, %step3#2 : i1, i32, i32
                    }}
                    scf.yield %step2#0, %step2#1, %step2#2 : i1, i32, i32
                  }}
                  scf.yield %step1#0, %step1#1, %step1#2 : i1, i32, i32
                }}
                scf.yield %fnd#2, %fnd#1 : i32, i32
              }} else {{
                scf.yield %zero_s, %zero_s : i32, i32
              }}
{bcast}
              %d = arith.index_cast %cd : i32 to index
              %k = arith.index_cast %cl : i32 to index
              // M fast, N slow: the token moves every claim, the weight block
              // only every %nat claims. The extent is this step's active token
              // count, so at a decode step the grid is one token deep and every
              // piece claimed is a piece that computes -- see the note where
              // %ntasks_t is built.
              // %nat is a runtime value, so these are a software divide --
              // roughly thirty instructions of v_rcp_f32 and Newton steps, on
              // a path taken once per die probe per stage, which is the same
              // order as the MACs a lane does in the piece it wins. At a decode
              // step %nat is 1 and the answer is (0, k); the branch is
              // wave-uniform, so it is an s_cbranch and the divide is simply
              // not executed.
              %isN1 = arith.cmpi eq, %nat, %c1_s : index
              %mkn:2 = scf.if %isN1 -> (index, index) {{
                scf.yield %c0_s, %k : index, index
              }} else {{
                %mr = arith.remui %k, %nat : index
                %kd = arith.divui %k, %nat : index
                scf.yield %mr, %kd : index, index
              }}
              %m = arith.addi %mkn#0, %c0_s : index
              %kn = arith.addi %mkn#1, %c0_s : index
              %kstride = arith.muli %kn, %cmaxdies : index
              %ix = arith.addi %d, %kstride : index
              %has = arith.cmpi ult, %ix, {count_expr} : index
              %acc2 = scf.if %has -> i32 {{
{open_body}
{body}
{close_body}
                %n = arith.addi %acc, %one_s : i32
                scf.yield %n : i32
              }} else {{
                scf.yield %acc : i32
              }}
              scf.yield %has, %acc2 : i1, i32
          }}
          %pcount{l}_{stage} = arith.addi %t{l}_{stage}#1, %zero_s : i32
"""

    STATIC_CLAIM = """          // Static claim: no queue, no atomic, no broadcast.
          //
          // The dynamic queue costs more than the work it hands out. Per layer
          // step at 128 workers, swiglu spends 14.7 us in its body on about a
          // thousandth of qkv's arithmetic, and what it is paying for is one
          // successful claim atomic, one failing one, two LDS broadcasts and
          // their four barriers -- on a counter that sixteen workgroups a die
          // are hammering, so the atomics serialise and the last workgroup
          // waits behind all the others to be told which single piece is its.
          //
          // None of that is needed to decide which piece is whose. The chiplet
          // reporting protocol already told this workgroup its rank among the
          // workgroups sharing its chiplet and how many there are, both
          // workgroup uniform and both already paid for, so which pieces are
          // die d's is a choice this makes rather than a fact it is told, and
          // the partition is disjoint and complete by arithmetic rather than
          // by a protocol.
          //
          // Which choice is the whole of {claim_name}, and it is worth 3.2%
          // for nothing: same instruction stream, same partition, two
          // multiplies with their operands swapped.
          //
          // A piece is a strip of adjacent output columns and the weight's
          // fastest axis is the column, so a piece is `slice * 2` adjacent
          // bytes of every row it reads, against a 128-byte line. Giving die
          // d a contiguous block of pieces rather than those congruent to d
          // was meant to stop the eight pieces that share a line landing on
          // eight XCDs with eight private L2s. That prediction was wrong and
          // the measurement says so: it made `down` and `o_proj`, the two
          // stages with 16 bytes a piece a row and therefore the whole of
          // that eightfold duplication, 1.8% faster and 0.3% slower. The
          // MALL absorbs cross-XCD duplication.
          //
          // What moved was gate_up, 13.2%, and gate_up is the one stage in
          // this model whose piece -- 48 columns, 96 bytes -- does not divide
          // a line, so three pieces in four straddle one. Blocking makes a
          // die's range 16 * 96 = 1536 bytes, twelve whole lines, and the
          // straddles internal to it. Alignment, not duplication.
          //
          // What it gives up is stealing: a die the dispatcher put no
          // workgroups on keeps its pieces, nobody computes them, the stage's
          // event never reaches its total and the launch hangs. That is the
          // same failure mode air.launch's co-residency guarantee already has,
          // and it is loud rather than silent.
          //
          // The token loop is inside the piece loop, which is the M-fast
          // traversal the dynamic path needed a software divide to express:
          // one weight block, then every token against it.
          %ndie{l}_{stage} = arith.addi {count_expr}, %cmaxm1 : index
          %nper{l}_{stage} = arith.divui %ndie{l}_{stage}, %cmaxdies : index
          %tst{l}_{stage} = scf.for %kn = %myrank2 to %nper{l}_{stage} step %mycnt
              iter_args(%accs = %zero_s) -> (i32) {{
{claim_ix}
            %has = arith.cmpi ult, %ix, {count_expr} : index
            %accn = scf.if %has -> i32 {{
              %a2 = scf.for %m = %c0_s to %nat step %c1_s
                  iter_args(%accm = %accs) -> (i32) {{
{open_body}
{body}
{close_body}
                %nm = arith.addi %accm, %one_s : i32
                scf.yield %nm : i32
              }}
              scf.yield %a2 : i32
            }} else {{
              scf.yield %accs : i32
            }}
            scf.yield %accn : i32
          }}
          %pcount{l}_{stage} = arith.addi %tst{l}_{stage}, %zero_s : i32
"""

    def strided_stage(
        l,
        stage,
        ev,
        count_expr,
        total_const,
        body,
        slot=None,
        lc=None,
        lanes=False,
        strip=0,
    ):
        slot = (l * stages + stage) if slot is None else slot
        lc = f"%L{l}" if lc is None else lc
        # How much of the workgroup the body uses.
        #   False  -- the lead thread alone; the body is a reduction or an
        #             in-place update that has not been spread yet.
        #   True   -- one wave, striding its outer loop from %lid by %nlane.
        #   "block"-- the body handles every wave itself, and may barrier.
        # The middle case has to be pinned to wave 0 once there is more than
        # one wave. Letting all of them run it is not the harmless duplication
        # it looks like: rope reads two halves of a head and writes both back,
        # so a second wave arriving late reads what the first already rotated
        # and rotates it again.
        if lanes in ("block", "threads"):
            open_body, close_body = "", ""
        elif lanes:
            open_body = "" if waves == 1 else "                  scf.if %isW0 {"
            close_body = "" if waves == 1 else "                  }"
        else:
            open_body = "                  scf.if %isLead {"
            close_body = "                  }"
        claim_block = (STATIC_CLAIM if static_claim else DYNAMIC_CLAIM).format(
            l=l,
            stage=stage,
            lc=lc,
            QUT=QUT,
            count_expr=count_expr,
            claim_name="--blocked-claim" if blocked_claim else "--round-robin-claim",
            claim_ix=(
                f"            %kstride = arith.muli %mydie, %nper{l}_{stage} : index\n"
                f"            %ix = arith.addi %kn, %kstride : index"
                if blocked_claim
                else "            %kstride = arith.muli %kn, %cmaxdies : index\n"
                "            %ix = arith.addi %mydie, %kstride : index"
            ),
            body=body,
            open_body=open_body,
            close_body=close_body,
            bcast=bcast_i32x2(
                "%sc" + str(l) + "_" + str(stage) + "#0",
                "%sc" + str(l) + "_" + str(stage) + "#1",
                "%cl",
                "%cd",
                f"c{l}_{stage}",
                14,
            ),
        )
        # A stage boundary costs 5.52 us -- the slope of the launch clock
        # against --pad-stages, measured when a token took 3.68 ms, of which
        # it was 1.42. Still the largest single thing in the program, so it
        # is worth knowing which part of it is which -- and worth re-running
        # this ladder, because two of the three pieces it found have been cut
        # since and the shares below are from before that.
        #
        # `strip` takes the boundary apart. It is only ever used on a pad
        # stage, and a pad stage is exactly the right place for it: its body is
        # empty, so nothing it fails to publish is ever read, and its event
        # word is read by nothing but its own spin -- every stage waits on its
        # own event and no other. So each level below can be removed with the
        # model still producing the same six tokens, and the difference between
        # two slopes is the price of what was removed.
        #
        # Measured on the current default, 28 layers at 128 workers, three
        # pads, minimum of two runs per point:
        #
        #   level  a pad stage contains          us/inst  the piece removed
        #     0    the whole boundary               5.52
        #     1    no acquire fence                 4.65   the fence     0.87
        #     2    and no spin, no release barrier  1.99   spin+barrier  2.65
        #     3    and no atomics                  -0.09   four atomics  2.09
        #     4    and no claim                            the claim     free
        #
        # Level 3 is indistinguishable from no pad at all, which is what says
        # the pieces account for the whole boundary rather than most of it.
        # Two counters a die, or one word holding both.
        #
        # The die's workers have to agree on two things before the last of
        # them can fire the event: how many pieces the die did, and how many
        # workers have arrived. Kept apart, that is an add to each and then --
        # for whoever turns out to be last -- an acquire load to read the sum
        # back, three memory operations on the boundary's critical path.
        #
        # Packed, the arrival count lives in the top 32 bits of the same word
        # the piece count is accumulated in, so adding `(1 << 32) + pieces`
        # does both at once and the value the atomic gives back carries the
        # answer to both questions: the top half says how many arrived before
        # me, and the bottom half plus my own pieces is the die's total. Nobody
        # will add after the last arriver, so that total is final and the load
        # is not needed. Three memory operations become one.
        #
        # The counts cannot collide: pieces a die does are bounded by `tasks`
        # and arrivals by the workers on a die, so neither half carries into
        # the other. The release stays where it was, on the one atomic, which
        # is stronger than before -- the count and the arrival now become
        # visible together rather than in that order.
        #
        # Same bytes as the two i32 arrays it replaces: `slots` 8-byte words
        # over the [0, 8*slots) the two of them had, so the flush word and the
        # timers keep their offsets. Every term of the offset is a multiple of
        # 8, which the i64 atomic needs.
        # Where the die's counter lives. Packed, it is one 8-byte word where
        # the two 4-byte ones were, so the step stride doubles -- %steploc
        # counts i32 words' worth of bytes -- and the second array goes away.
        locptrs = (
            f"""          %lw{l}_{stage} = arith.constant {8 * slot * maxdies} : i64
          %myd{l}_{stage} = arith.index_cast %mydie : index to i64
          %myd8{l}_{stage} = arith.muli %myd{l}_{stage}, %eightL : i64
          %lb{l}_{stage} = arith.addi %lw{l}_{stage}, %myd8{l}_{stage} : i64
          %sl2{l}_{stage} = arith.addi %steploc, %steploc : i64
          %lo{l}_{stage} = arith.addi %lb{l}_{stage}, %sl2{l}_{stage} : i64
          %pkP{l}_{stage} = llvm.getelementptr %locp[%lo{l}_{stage}] : (!llvm.ptr, i64) -> !llvm.ptr, i8"""
            if pack_arrival
            else f"""          %lw{l}_{stage} = arith.constant {4 * slot * maxdies} : i64
          %myd{l}_{stage} = arith.index_cast %mydie : index to i64
          %myd4{l}_{stage} = arith.muli %myd{l}_{stage}, %fourL : i64
          %lb{l}_{stage} = arith.addi %lw{l}_{stage}, %myd4{l}_{stage} : i64
          %lo{l}_{stage} = arith.addi %lb{l}_{stage}, %steploc : i64
          %locP{l}_{stage} = llvm.getelementptr %locp[%lo{l}_{stage}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %aw{l}_{stage} = arith.constant {4 * slots} : i64
          %ao{l}_{stage} = arith.addi %aw{l}_{stage}, %lo{l}_{stage} : i64
          %arrP{l}_{stage} = llvm.getelementptr %locp[%ao{l}_{stage}] : (!llvm.ptr, i64) -> !llvm.ptr, i8"""
        )
        packed = (
            f"""            %pk{l}_{stage} = arith.extui %pcount{l}_{stage} : i32 to i64
            %pv{l}_{stage} = arith.addi %pk{l}_{stage}, %arrbit : i64
            %old{l}_{stage} = llvm.atomicrmw add %pkP{l}_{stage}, %pv{l}_{stage} syncscope("agent") release : !llvm.ptr, i64
            %arh{l}_{stage} = arith.shrui %old{l}_{stage}, %c32L : i64
            %ar{l}_{stage} = arith.trunci %arh{l}_{stage} : i64 to i32
            %last{l}_{stage} = arith.subi %mycnt_i, %one_s : i32
            %amLast{l}_{stage} = arith.cmpi eq, %ar{l}_{stage}, %last{l}_{stage} : i32
            scf.if %amLast{l}_{stage} {{
              %pre{l}_{stage} = arith.trunci %old{l}_{stage} : i64 to i32
              %tot{l}_{stage} = arith.addi %pre{l}_{stage}, %pcount{l}_{stage} : i32"""
            if pack_arrival
            else f"""            %la{l}_{stage} = llvm.atomicrmw add %locP{l}_{stage}, %pcount{l}_{stage} syncscope("agent") monotonic : !llvm.ptr, i32
            // Release on the arrival so the accumulate above is visible to
            // whoever turns out to be last.
            %ar{l}_{stage} = llvm.atomicrmw add %arrP{l}_{stage}, %one_s syncscope("agent") release : !llvm.ptr, i32
            %last{l}_{stage} = arith.subi %mycnt_i, %one_s : i32
            %amLast{l}_{stage} = arith.cmpi eq, %ar{l}_{stage}, %last{l}_{stage} : i32
            scf.if %amLast{l}_{stage} {{
              %tot{l}_{stage} = llvm.load %locP{l}_{stage} atomic syncscope("agent") acquire {{alignment = 4 : i64}} : !llvm.ptr -> i32"""
        )
        sig_block = (
            ""
            if strip >= 3
            else f"""          scf.if %isLead {{
{packed}
              %sig{l}_{stage} = llvm.atomicrmw add %p{l}_{stage}, %tot{l}_{stage} syncscope("") release : !llvm.ptr, i32
{flush_count.format(l=l, stage=stage)}            }}
          }}"""
        )
        spin_block = (
            ""
            if strip >= 2
            else f"""{spin_open}
          scf.while : () -> () {{
            %seen_l{l}_{stage} = scf.if %isL0 -> (i32) {{
              %v = llvm.load %p{l}_{stage} atomic syncscope("") {spin_order} {{alignment = 4 : i64}} : !llvm.ptr -> i32
              scf.yield %v : i32
            }} else {{
              scf.yield %zero_s : i32
            }}
            %seen = rocdl.readfirstlane %seen_l{l}_{stage} : i32
            %notYet = arith.cmpi ult, %seen, {total_const} : i32
            scf.condition(%notYet)
          }} do {{
{spin_wait}            scf.yield
          }}
{spin_close_nf if strip == 1 else spin_close}
{spin_end_nf if (strip == 1 and spin_end) else spin_end}"""
        )
        return f"""
          // stage {stage}
{timer_begin(l, stage)}
          %ec{l}_{stage} = arith.constant {4 * ev} : i64
          %eo{l}_{stage} = arith.addi %stepev, %ec{l}_{stage} : i64
          %p{l}_{stage} = llvm.getelementptr %evptr[%eo{l}_{stage}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %hb{l}_{stage} = arith.constant {stage * maxdies * 2} : index
{"" if strip >= 4 else claim_block}          // Two-level: add into a counter only this die touches, then let the
          // last worker on the die flush the die's whole share once. The
          // instructions are the same as signalling per worker; what changes is
          // how many times the device-scope one runs.
{locptrs}
          // Signalling is the lead thread's job alone -- one arrival per
          // workgroup is the whole premise of the reduction. Within a wave the
          // release covers what the other lanes stored, because vmcnt counts
          // the wave's memory operations whatever exec mask they issued under.
          // Across waves it does not: wave 0's s_waitcnt says nothing about
          // wave 5's stores, and neither does s_barrier. So every wave takes
          // its own release fence and only then do they meet, which is what
          // makes the event mean "this workgroup's output is readable" rather
          // than "wave 0's is" -- see the note on stage_bar.
{stage_bar}
{sig_block}
          // See the note on spin_open: one wave waits, a barrier releases the
          // workgroup, and the acquire happens once rather than per poll.
{timer_mid(l, stage)}
{spin_block}
{timer_end(l, stage)}"""

    def single_stage(l, stage, ev, body, slot=None, lc=None, lanes=False):
        slot = (l * stages + stage) if slot is None else slot
        lc = f"%L{l}" if lc is None else lc
        # As in strided_stage: `lanes` means the body strides its own loops
        # from %tx_s and closes them with gpu.subgroup_reduce. A body that
        # does not runs on the lead lane alone. The signal stays on the lead
        # lane either way -- the event counts tasks, and the task is one.
        # "block" means the body spreads itself over every wave and reduces
        # across them, so it must not be pinned to wave 0 -- the barriers
        # inside it are barriers the whole workgroup has to reach.
        if lanes == "block":
            w0open, w0close = "", ""
        else:
            w0open = "" if waves == 1 else "            scf.if %isW0 {"
            w0close = "" if waves == 1 else "            }"
        body_block = (
            f"""{w0open}
            scf.for %m = %c0_s to %nat step %c1_s {{
{body}
            }}
{w0close}
{stage_bar}
            scf.if %isLead {{
              %sig{l}_{stage} = llvm.atomicrmw add %p{l}_{stage}, %n1_s syncscope("") release : !llvm.ptr, i32
            }}"""
            if lanes
            else f"""            scf.if %isLead {{
              scf.for %m = %c0_s to %nat step %c1_s {{
{body}
              }}
              %sig{l}_{stage} = llvm.atomicrmw add %p{l}_{stage}, %n1_s syncscope("") release : !llvm.ptr, i32
            }}"""
        )
        return f"""
          // stage {stage} -- one task: a reduction over the whole row, so it
          // cannot be split by output slice the way the matmuls can.
{timer_begin(l, stage)}
          %hsingle{stage}_{l} = arith.constant {stage * maxdies * 2} : index
          %ec{l}_{stage} = arith.constant {4 * ev} : i64
          %eo{l}_{stage} = arith.addi %stepev, %ec{l}_{stage} : i64
          %p{l}_{stage} = llvm.getelementptr %evptr[%eo{l}_{stage}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
{single_claim(l, stage, lc)}
          scf.if %mine{l}_{stage} {{
{body_block}
          }}
{timer_mid(l, stage)}
{spin_open}
          scf.while : () -> () {{
            %seenl{l}_{stage} = scf.if %isL0 -> (i32) {{
              %v = llvm.load %p{l}_{stage} atomic syncscope("") {spin_order} {{alignment = 4 : i64}} : !llvm.ptr -> i32
              scf.yield %v : i32
            }} else {{
              scf.yield %zero_s : i32
            }}
            %seen = rocdl.readfirstlane %seenl{l}_{stage} : i32
            %notYet = arith.cmpi ult, %seen, %n1_s : i32
            scf.condition(%notYet)
          }} do {{
{spin_wait}            scf.yield
          }}
{spin_close}
{spin_end}
{timer_end(l, stage)}"""

    # The weights are read once per layer and never again inside a decode step,
    # and they are far larger than anything else in flight, so they look like
    # exactly what should be streamed past the cache rather than kept in it.
    # Measured, they are 7% faster kept: 44.7 ms/token with `nt` on the weight
    # loads against 41.7 without, slope over 99 extra launches.
    #
    # That is also closer to what Fleet does here, not further from it. Fleet's
    # non-temporal weight loads live in the gang_ksplit path
    # (gang_ksplit_linear_mi300.cuh:88, amd_buffer_coherence_enum(18) = nt|sc1),
    # which its batch-1 build does not compile -- `fleet.s` contains no `nt` at
    # all, while this chain was emitting 74. So the default is off and --nt
    # turns it back on; the lowering itself is covered by the ISA gate in
    # mlir/test/Conversion/AIRToROCDL/air_nontemporal.mlir either way.
    ntw = " {{nontemporal = true}}" if nt_weights else ""

    def dot_loop(
        dst,
        start,
        red_c,
        red_num,
        step_c,
        step_num,
        lhs,
        lhsty,
        wmat,
        wty,
        l,
        jname,
        tag,
        indent,
        lhs_silu=None,
        staged=False,
        ltype=None,
    ):
        """The dot product one lane owns, with `unroll` loads in flight.

        `lhs_silu` makes the reduction build its own left-hand side: element
        k is `silu(lhs[k]) * lhs[k + lhs_silu]` rather than `lhs[k]`. That is
        the whole of the SwiGLU fusion, and the reason it is on this side.
        The weights are not touched -- the strip each piece reads and the
        order it reads it in are exactly what they were -- and the only thing
        that changes is that the activation arrives as a pair to be combined
        instead of as a value some earlier stage combined.

        Fleet fuses on this side too. `silu_mul_linear` computes
        `silu(gate) * up @ weight^T`, taking `gate = input[:, :K]` and
        `up = input[:, K:]` (silu_mul_linear_mi300.cuh:39-40, 114-115): the
        consumer of the activation, with the gate and up halves left end to
        end, which is also why there is nothing to gain by interleaving them.

        Rolled, this loop asks for one weight and waits for it. The arithmetic
        says that is the whole story of the matmuls: at 128 workers gate_up
        gives each thread 2688 weights over 28 layers and takes 2.13 ms/token,
        which is about 1660 cycles a weight against fourteen instructions of
        work -- one full memory latency per iteration, overlapped with nothing.
        A wave has 128 bytes outstanding where it would need tens of kilobytes
        to hold the machine's bandwidth open.

        `unroll` loads before the first `extf` forces that many to be in
        flight: the wave issues them back to back and only then takes the
        `s_waitcnt`, which is the one lever here that does not need more
        workgroups. The accumulate stays a single chain in the original order,
        so the result is bit for bit what the rolled loop produced and a
        difference in the output is a bug rather than a rounding change.

        Falls back to rolled whenever the trip count does not divide, which is
        checked here rather than assumed: a remainder loop would double the
        code for the cases that do not arise in this model.
        """
        pad = " " * indent
        u = 1
        if unroll > 1 and step_num > 0 and red_num % step_num == 0:
            while u * 2 <= unroll and red_num % (step_num * u * 2) == 0:
                u *= 2

        # One element of the left-hand side. Staged, that is an LDS read and
        # the SwiGLU -- if there is one -- has already been applied on the way
        # in, so it is one read either way and the transcendental is paid once
        # per element rather than once per output column.
        def lhsval(idx, sfx):
            if staged:
                return f"{pad}  %lv{tag}{sfx} = memref.load %ldsl[{idx}] : {ltype}"
            if lhs_silu is None:
                return f"{pad}  %lv{tag}{sfx} = memref.load {lhs}[%m, {idx}] : {lhsty}"
            return (
                f"{pad}  %lg{tag}{sfx} = memref.load {lhs}[%m, {idx}] : {lhsty}\n"
                f"{pad}  %ui{tag}{sfx} = arith.addi {idx}, {lhs_silu} : index\n"
                f"{pad}  %lu{tag}{sfx} = memref.load {lhs}[%m, %ui{tag}{sfx}] : {lhsty}"
            )

        def lhscomb(sfx):
            if staged or lhs_silu is None:
                return []
            return [
                f"{pad}  %ng{tag}{sfx} = arith.negf %lg{tag}{sfx} : f32",
                f"{pad}  %eg{tag}{sfx} = math.exp %ng{tag}{sfx} : f32",
                f"{pad}  %de{tag}{sfx} = arith.addf %fone_s, %eg{tag}{sfx} : f32",
                f"{pad}  %si{tag}{sfx} = arith.divf %lg{tag}{sfx}, %de{tag}{sfx} : f32",
                f"{pad}  %lv{tag}{sfx} = arith.mulf %si{tag}{sfx}, %lu{tag}{sfx} : f32",
            ]

        if u == 1:
            body = [
                f"{pad}{dst} = scf.for %i{tag} = {start} to {red_c} step {step_c}",
                f"{pad}    iter_args(%sacc{tag} = %fzero_s) -> (f32) {{",
                lhsval(f"%i{tag}", ""),
                *lhscomb(""),
                f"{pad}  %wb{tag} = memref.load {wmat}[%L{l}, {wix(f'%i{tag}', jname)}]{ntw} : {wty}",
                f"{pad}  %wv{tag} = arith.extf %wb{tag} : bf16 to f32",
                f"{pad}  %mp{tag} = arith.mulf %lv{tag}, %wv{tag} : f32",
                f"{pad}  %s2{tag} = arith.addf %sacc{tag}, %mp{tag} : f32",
                f"{pad}  scf.yield %s2{tag} : f32",
                f"{pad}}}",
            ]
            return "\n".join(body)
        out = [f"{pad}%cSU{tag} = arith.constant {step_num * u} : index"]
        for k in range(1, u):
            out.append(f"{pad}%cO{tag}_{k} = arith.constant {step_num * k} : index")
        out.append(
            f"{pad}{dst} = scf.for %i{tag} = {start} to {red_c} step %cSU{tag}\n"
            f"{pad}    iter_args(%sacc{tag} = %fzero_s) -> (f32) {{"
        )
        # Every load first, so the wave has all of them outstanding before the
        # first extf makes it wait.
        #
        # Under [output][reduction] with a contiguous slice these `u` loads
        # are `u` adjacent bf16, and there is nothing to do about that here:
        # the backend already merges them. Checked in the ISA -- the emitted
        # gfx950 assembly for this model has 206 `global_load_dwordx4` and one
        # `global_load_ushort`, and writing them as an explicit `vector.load`
        # of `vector<8xbf16>` produces assembly with exactly the same counts
        # and a launch clock 0.17% apart, which is noise. So Little's law,
        # which wanted sixteen bytes outstanding a lane rather than two, was
        # already satisfied by the layout change; it is not what is left.
        for k in range(u):
            idx = f"%i{tag}" if k == 0 else f"%ik{tag}_{k}"
            if k:
                out.append(f"{pad}  {idx} = arith.addi %i{tag}, %cO{tag}_{k} : index")
            out.append(
                f"{pad}  %wb{tag}_{k} = memref.load {wmat}[%L{l}, {wix(idx, jname)}]{ntw} : {wty}"
            )
        # Then every left-hand side load, for the same reason: what is being
        # bought here is loads in flight, and with the SwiGLU fused that is
        # two an element rather than one. The combining waits until all of
        # them have been asked for.
        for k in range(u):
            idx = f"%i{tag}" if k == 0 else f"%ik{tag}_{k}"
            out.append(lhsval(idx, f"_{k}"))
        for k in range(u):
            out.extend(lhscomb(f"_{k}"))
        prev = f"%sacc{tag}"
        for k in range(u):
            out.append(
                f"{pad}  %wv{tag}_{k} = arith.extf %wb{tag}_{k} : bf16 to f32\n"
                f"{pad}  %mp{tag}_{k} = arith.mulf %lv{tag}_{k}, %wv{tag}_{k} : f32\n"
                f"{pad}  %s2{tag}_{k} = arith.addf {prev}, %mp{tag}_{k} : f32"
            )
            prev = f"%s2{tag}_{k}"
        out.append(f"{pad}  scf.yield {prev} : f32\n{pad}}}")
        return "\n".join(out)

    def matmul_stage(
        l,
        stage,
        ev,
        out,
        outty,
        lhs,
        lhsty,
        wmat,
        wty,
        slice_c,
        red_c,
        red_num,
        slice_num,
        residual=None,
        count_c="%ctasks",
        total_c="%ntasks_t",
        lhs_silu=None,
    ):
        # Read the reduction operand into LDS once, then let every lane read
        # it from there. The copy is `red_num` elements over all the threads
        # of the workgroup -- 2 elements a thread for a 1024-wide reduction --
        # against the `cols` full passes over it that the lanes were each
        # making out of global memory.
        #
        # A SwiGLU, if this stage has one, is applied here: once per element
        # on the way in, rather than once per element per output column. That
        # is the whole reason Fleet can fuse it for free
        # (silu_mul_linear_mi300.cuh:34, 190-192) and this generator could
        # not.
        #
        # Barriers on both sides: the buffer is reused by every stage and by
        # every piece a workgroup claims, so the copy must not start until the
        # last reader has finished and must not be read until it is done.
        staged = stage_lhs
        if staged and lhs_silu is not None:
            copy_body = f"""                  %sg = memref.load {lhs}[%m, %sj] : {lhsty}
                  %sui = arith.addi %sj, {lhs_silu} : index
                  %su = memref.load {lhs}[%m, %sui] : {lhsty}
                  %sng = arith.negf %sg : f32
                  %seg = math.exp %sng : f32
                  %sde = arith.addf %fone_s, %seg : f32
                  %ssi = arith.divf %sg, %sde : f32
                  %sv = arith.mulf %ssi, %su : f32"""
        else:
            copy_body = (
                f"""                  %sv = memref.load {lhs}[%m, %sj] : {lhsty}"""
            )
        stage_copy = (
            ""
            if not staged
            else f"""                gpu.barrier
                scf.for %sj = %tx_s to {red_c} step %nthr {{
{copy_body}
                  memref.store %sv, %ldsl[%sj] : {LT}
                }}
                gpu.barrier
"""
        )
        store = (
            f"""
                  %rv = memref.load {residual}[%m, %j] : {outty}
                  %a2 = arith.addf %rv, %a : f32
                  memref.store %a2, {out}[%m, %j] : {outty}"""
            if residual
            else f"""
                  memref.store %a, {out}[%m, %j] : {outty}"""
        )
        # Same store, but on the unclamped column, and indented for the
        # extra scf.if the multi-wave path wraps it in.
        store_blk = (
            f"""
                      %rv = memref.load {residual}[%m, %j2] : {outty}
                      %a2 = arith.addf %rv, %a : f32
                      memref.store %a2, {out}[%m, %j2] : {outty}"""
            if residual
            else f"""
                      memref.store %a, {out}[%m, %j2] : {outty}"""
        )
        # A lane per output column. Splitting the columns rather than the
        # reduction needs no cross-lane anything -- the columns are
        # independent, and each lane keeps its own accumulator in a register --
        # and it is the split that reads the weights coalesced: %j is the
        # fastest axis of {wmat}, so lane n and lane n+1 ask for adjacent
        # words of the same cache line. Splitting %i instead would have every
        # lane on a different row, one line each.
        if waves == 1:
            return strided_stage(
                l,
                stage,
                ev,
                count_c,
                total_c,
                f"""{stage_copy}                %j0 = arith.muli %ix, {slice_c} : index
                scf.for %jj = %tx_s to {slice_c} step %nlane {{
                  %j = arith.addi %j0, %jj : index
{dot_loop("%a", "%c0_s", red_c, red_num, "%c1_s", 1, lhs, lhsty, wmat, wty,
          l, "%j", f"w1_{l}_{stage}", 18, lhs_silu=lhs_silu,
          staged=staged, ltype=LT)}{store}
                }}""",
                lanes=True,
            )

        # With more than one wave the columns alone cannot keep them busy: a
        # slice is `width / tasks` wide and at 128 tasks that is 8 for anything
        # dim-wide, against 64 lanes. So the waves divide the reduction instead
        # and the lanes keep the columns, which leaves the weight reads
        # coalesced exactly as above -- only now wave w reads rows w, w+waves,
        # ... of the same 64-column strip.
        #
        # The column loop counts blocks rather than striding by the lane id, so
        # every thread runs the same number of iterations and reaches the same
        # barriers; a lane-dependent trip count around a gpu.barrier hangs the
        # workgroup. Columns past the end are computed on a clamped index and
        # thrown away, which keeps every load in bounds.
        cols = 1
        while cols * 2 <= min(slice_num, wave):
            cols *= 2
        klanes = wave // cols
        ksteps = klanes.bit_length() - 1
        nblk = (slice_num + cols - 1) // cols
        # A slice is `width / tasks` columns -- 8 for anything dim-wide at 128
        # tasks -- so a lane per column left 56 of 64 lanes idle and used 32 of
        # every 128-byte line. Splitting the lanes two ways, `cols` across the
        # columns and the remaining `klanes` across the reduction, keeps all 64
        # busy whatever the slice is; the lanes sharing a column differ only in
        # the bits above log2(cols), so an xor butterfly over those bits folds
        # their partials together. Across the four matmuls that is 3.7x of
        # arithmetic that was being thrown away.
        #
        # How a lane's share of the reduction is laid out follows the weight
        # layout and has to, or the layout buys nothing. Strided -- lane k
        # takes k, k + waves*klanes, ... -- is right for [reduction][output],
        # where consecutive reduction indices are a full width apart anyway
        # and the coalescing that matters is across the lanes of a column
        # group. Under [output][reduction] the reduction is the contiguous
        # axis, so a lane wants a contiguous block of it: lane k takes
        # [k*chunk, (k+1)*chunk) and its `unroll` loads in flight are `unroll`
        # adjacent bf16 of one line rather than `unroll` separate lines. That
        # is the lm head's access pattern, which is the fast one.
        #
        # It only works when the split divides the reduction; when it does
        # not, fall back to strided rather than emit a remainder loop, and say
        # so at generation time rather than quietly computing a wrong sum.
        kways = waves * klanes
        chunked = out_major and red_num % kways == 0
        kslice = (
            f"""                %cKC{l}_{stage} = arith.constant {red_num // kways} : index
                %kbeg{l}_{stage} = arith.muli %ksl{l}_{stage}, %cKC{l}_{stage} : index
                %kend{l}_{stage} = arith.addi %kbeg{l}_{stage}, %cKC{l}_{stage} : index
"""
            if chunked
            else ""
        )
        red_start = f"%kbeg{l}_{stage}" if chunked else f"%ksl{l}_{stage}"
        red_end = f"%kend{l}_{stage}" if chunked else red_c
        red_step = "%c1_s" if chunked else f"%cKS{l}_{stage}"
        red_trip = (red_num // kways) if chunked else red_num
        red_stepn = 1 if chunked else kways
        # How the `waves * klanes` partials of one column meet.
        #
        # They were meeting entirely in LDS: every thread stored its partial
        # and lane c of wave 0 walked all `waves * klanes` of column c in a
        # single `iter_args` chain -- 64 dependent LDS loads for anything
        # dim-wide at 128 tasks, once per column block. That chain, not the
        # weight traffic, is what set the spread in achieved bandwidth across
        # these four stages: dividing its length by the weights a lane loads
        # in the same block orders all five matmul classes correctly, over a
        # 31x range, where line coverage and bytes in flight did not.
        #
        #   class    epilogue  weights/lane  ratio   GB/s
        #   lm_head       152          2432  0.062   1078
        #   gate_up        32           128  0.250    734
        #   qkv            16            64  0.250    678
        #   down           64            48  1.333    363
        #   o_proj         64            32  2.000    316
        #
        # The lanes sharing a column differ only in the bits above
        # log2(cols), so `ksteps` xor exchanges fold their partials in
        # registers and leave one partial a wave to go through LDS. The chain
        # becomes `waves` long whatever `klanes` is, and only the `cols` lanes
        # of each wave that hold the fold need store.
        kfold = "".join(
            f"""                  %kx{l}_{stage}_{k} = arith.constant {cols << k} : i32
                  %kw{l}_{stage}_{k} = arith.constant {wave} : i32
                  %kv{l}_{stage}_{k}, %kp{l}_{stage}_{k} = gpu.shuffle xor %kacc{l}_{stage}_{k}, %kx{l}_{stage}_{k}, %kw{l}_{stage}_{k} : f32
                  %kacc{l}_{stage}_{k + 1} = arith.addf %kacc{l}_{stage}_{k}, %kv{l}_{stage}_{k} : f32
""" for k in range(ksteps)
        )
        kfold = (
            f"                  %kacc{l}_{stage}_0 = arith.addf %part0, %fzero_s : f32\n"
            + kfold
            + f"                  %part = arith.addf %kacc{l}_{stage}_{ksteps}, %fzero_s : f32\n"
        )
        if not fold_klanes:
            kfold = "                  %part = arith.addf %part0, %fzero_s : f32\n"
        # Slot w*cols + c, so wave 0 reads `waves` of them at stride `cols`.
        kstore = f"""                  %kin{l}_{stage} = arith.cmpi ult, %lid, %cC{l}_{stage} : index
                  scf.if %kin{l}_{stage} {{
                    %kwo2{l}_{stage} = arith.muli %wid, %cC{l}_{stage} : index
                    %ksi{l}_{stage} = arith.addi %kwo2{l}_{stage}, %lid : index
                    memref.store %part, %ldsr[%ksi{l}_{stage}] : {RT}
                  }}
"""
        kread = f"""                      %a = scf.for %blw = %c0_s to %cwaves step %c1_s
                          iter_args(%blacc = %fzero_s) -> (f32) {{
                        %blo = arith.muli %blw, %cC{l}_{stage} : index
                        %bli = arith.addi %blo, %lid : index
                        %blv = memref.load %ldsr[%bli] : {RT}
                        %bln = arith.addf %blacc, %blv : f32
                        scf.yield %bln : f32
                      }}
"""
        if not fold_klanes:
            kstore = f"                  memref.store %part, %ldsr[%tx_s] : {RT}\n"
            kread = f"""                      %a = scf.for %blw = %c0_s to %cwaves step %c1_s
                          iter_args(%blacc = %fzero_s) -> (f32) {{
                        %blo = arith.muli %blw, %nlane : index
                        %bin = scf.for %blk = %c0_s to %cKL{l}_{stage} step %c1_s
                            iter_args(%bkacc = %blacc) -> (f32) {{
                          %bko = arith.muli %blk, %cC{l}_{stage} : index
                          %bkb = arith.addi %blo, %bko : index
                          %bli = arith.addi %bkb, %lid : index
                          %blv = memref.load %ldsr[%bli] : {RT}
                          %bln = arith.addf %bkacc, %blv : f32
                          scf.yield %bln : f32
                        }}
                        scf.yield %bin : f32
                      }}
"""
        return strided_stage(
            l,
            stage,
            ev,
            count_c,
            total_c,
            f"""{stage_copy}                %j0 = arith.muli %ix, {slice_c} : index
                %cC{l}_{stage} = arith.constant {cols} : index
                %cKS{l}_{stage} = arith.constant {waves * klanes} : index
                %cKL{l}_{stage} = arith.constant {klanes} : index
                %cnblk{l}_{stage} = arith.constant {nblk} : index
                %cslast{l}_{stage} = arith.constant {slice_num - 1} : index
                %col{l}_{stage} = arith.remui %lid, %cC{l}_{stage} : index
                %kln{l}_{stage} = arith.divui %lid, %cC{l}_{stage} : index
                %kwo{l}_{stage} = arith.muli %wid, %cKL{l}_{stage} : index
                %ksl{l}_{stage} = arith.addi %kwo{l}_{stage}, %kln{l}_{stage} : index
{kslice}                scf.for %jb = %c0_s to %cnblk{l}_{stage} step %c1_s {{
                  %jbo = arith.muli %jb, %cC{l}_{stage} : index
                  %jj = arith.addi %jbo, %col{l}_{stage} : index
                  %jok = arith.cmpi ult, %jj, {slice_c} : index
                  %jcl = arith.minsi %jj, %cslast{l}_{stage} : index
                  %j = arith.addi %j0, %jcl : index
{dot_loop("%part0", red_start, red_end, red_trip,
          red_step, red_stepn, lhs, lhsty, wmat, wty,
          l, "%j", f"wm_{l}_{stage}", 18, lhs_silu=lhs_silu,
          staged=staged, ltype=LT)}
{kfold}                  // The wave partials for one column live at the same lane of
                  // every wave, so the slot is just the thread id and wave 0
                  // walks them with a fixed stride.
                  gpu.barrier
{kstore}                  gpu.barrier
                  scf.if %isW0 {{
                    // Lane c of wave 0 finishes column c: the butterfly left
                    // every lane sharing a column holding that column's wave
                    // total, so slot w*wave + c is wave w's share of column c.
                    %inC{l}_{stage} = arith.cmpi ult, %lid, %cC{l}_{stage} : index
                    scf.if %inC{l}_{stage} {{
{kread}                      scf.if %jok {{
                        %j2 = arith.addi %j0, %jj : index{store_blk}
                      }}
                    }}
                  }}
                }}""",
            lanes="block",
        )

    # Per-head rmsnorm then rope, in place in the qkv buffer. Rope reads both
    # halves of a head before writing either, which is why it is a second loop
    # over h2 rather than folded into the first.
    def head_norm_rope(base, wofs, l, tag):
        """Per-head rmsnorm then rope, in place in the qkv buffer.

        Qwen3 normalises q and k per head before rotating them and Qwen2 does
        not (kv_cache_update_mi300.cuh:130). Rope reads both halves of a head
        before writing either, which is why it is a second loop rather than
        folded into the first; both halves use the same frequency, which is
        what makes rotate-half a rotation (tasks/ampere/norm.cuh:101-118).
        """
        return f"""                  %hp_{tag} = scf.for %hdi = %tx_s to %chd_s step %nlane
                      iter_args(%sa_{tag} = %fzero_s) -> (f32) {{
                    %hi_{tag} = arith.addi {base}, %hdi : index
                    %hv_{tag} = memref.load %sqkv[%m, %hi_{tag}] : {QT}
                    %hq_{tag} = arith.mulf %hv_{tag}, %hv_{tag} : f32
                    %hn_{tag} = arith.addf %sa_{tag}, %hq_{tag} : f32
                    scf.yield %hn_{tag} : f32
                  }}
{wave_reduce("%hp_" + tag, "%hs_" + tag, "arith.addf", "h" + tag, 18)}
                  %hm_{tag} = arith.divf %hs_{tag}, %fhd_s : f32
                  %hme_{tag} = arith.addf %hm_{tag}, %eps_s : f32
                  %hr_{tag} = math.sqrt %hme_{tag} : f32
                  scf.for %hdi = %tx_s to %chd_s step %nlane {{
                    %wi_{tag} = arith.addi %hdi, {wofs} : index
                    %hj_{tag} = arith.addi {base}, %hdi : index
                    %hw_{tag} = memref.load %sqkv[%m, %hj_{tag}] : {QT}
                    %nv_{tag} = arith.divf %hw_{tag}, %hr_{tag} : f32
                    %nw_{tag} = memref.load %sqkn[%L{l}, %wi_{tag}] : {QKNT}
                    %ov_{tag} = arith.mulf %nv_{tag}, %nw_{tag} : f32
                    memref.store %ov_{tag}, %sqkv[%m, %hj_{tag}] : {QT}
                  }}
                  scf.for %hdi = %tx_s to %ch2_s step %nlane {{
                    %i1_{tag} = arith.addi {base}, %hdi : index
                    %dh_{tag} = arith.addi %hdi, %ch2_s : index
                    %i2_{tag} = arith.addi {base}, %dh_{tag} : index
                    %v1_{tag} = memref.load %sqkv[%m, %i1_{tag}] : {QT}
                    %v2_{tag} = memref.load %sqkv[%m, %i2_{tag}] : {QT}
                    %cs_{tag} = memref.load %sro[%ropepos, %hdi] : {ROT}
                    %ds_{tag} = arith.addi %hdi, %chd_s : index
                    %sn_{tag} = memref.load %sro[%ropepos, %ds_{tag}] : {ROT}
                    %x1_{tag} = arith.mulf %v1_{tag}, %cs_{tag} : f32
                    %y1_{tag} = arith.mulf %v2_{tag}, %sn_{tag} : f32
                    %o1_{tag} = arith.subf %x1_{tag}, %y1_{tag} : f32
                    %x2_{tag} = arith.mulf %v2_{tag}, %cs_{tag} : f32
                    %y2_{tag} = arith.mulf %v1_{tag}, %sn_{tag} : f32
                    %o2_{tag} = arith.addf %x2_{tag}, %y2_{tag} : f32
                    memref.store %o1_{tag}, %sqkv[%m, %i1_{tag}] : {QT}
                    memref.store %o2_{tag}, %sqkv[%m, %i2_{tag}] : {QT}
                  }}"""

    base_x = layers * stages

    # embed: this step's input is the token the last step produced. Fleet's
    # embed_layer (builder.py:755) does the same gather.
    w(
        strided_stage(
            "x",
            stages + 0,
            base_x + 0,
            "%ctasks",
            "%ntasks_t",
            f"""                %tp = arith.addi %seq, %m : index
                %tk = memref.load %stok[%tp] : {TKT}
                %tki = arith.index_cast %tk : i32 to index
                %j0 = arith.muli %ix, %csliceD : index
                scf.for %jj = %tx_s to %csliceD step %nthr {{
                  %j = arith.addi %j0, %jj : index
                  %embw = memref.load %semb[%tki, %j] : {EMTB}
                  %ev = arith.extf %embw : bf16 to f32
                  memref.store %ev, %sx[%m, %j] : {AT}
                }}""",
            slot=base_x + 0,
            lc="%L0",
            lanes="threads",
        )
    )

    for l in range(layers):
        base = stages * l
        w(f"\n          // ================= layer {l} =================")

        # 0: r = rmsnorm(x) * n1
        w(
            single_stage(
                l,
                0,
                base + 0,
                f"""              %sp{l} = scf.for %i = %tx_s to %cdim_s step %nthr
                  iter_args(%s = %fzero_s) -> (f32) {{
                %v = memref.load %sx[%m, %i] : {AT}
                %sq2 = arith.mulf %v, %v : f32
                %s2 = arith.addf %s, %sq2 : f32
                scf.yield %s2 : f32
              }}
              // Each thread sums its own stride of the row and the block adds
              // the partials, which every thread then has -- so %rms below is
              // uniform without anything being broadcast. Reassociating a
              // float sum moves the last bits; the host comparison is relative
              // to 2e-2 and the token check is an argmax, and the independent
              // numpy reference already sums in a third order again.
{block_sum("%sp" + str(l), "%ss" + str(l), "n" + str(l), 14)}
              %mean{l} = arith.divf %ss{l}, %fdim_s : f32
              %me{l} = arith.addf %mean{l}, %eps_s : f32
              %rms{l} = math.sqrt %me{l} : f32
              scf.for %i = %tx_s to %cdim_s step %nthr {{
                %v = memref.load %sx[%m, %i] : {AT}
                %nv = arith.divf %v, %rms{l} : f32
                %nw = memref.load %sn1[%L{l}, %i] : {NT}
                %rv = arith.mulf %nv, %nw : f32
                memref.store %rv, %sr[%m, %i] : {AT}
              }}""",
                lanes="block",
            )
        )

        # 1: qkv = r @ Wqkv
        w(
            matmul_stage(
                l,
                1,
                base + 1,
                "%sqkv",
                QT,
                "%sr",
                AT,
                "%swqkv",
                WQTB,
                "%csliceQ",
                "%cdim_s",
                dim,
                slice_q,
            )
        )

        # 2: per-head norm, rope, and the cache append. One piece per
        # (token, query head); the kv heads ride along on the first `kv_heads`
        # of them, which is Fleet's shape too -- the update is per kv head.
        w(
            strided_stage(
                l,
                2,
                base + 2,
                "%cheads_s",
                "%nheads_t",
                f"""                %pos = arith.addi %wbase, %m : index
                %ropepos = arith.addi %wbase, %m : index
                %hb = arith.muli %ix, %chd_s : index
                scf.execute_region {{
{head_norm_rope("%hb", "%c0_s", l, "q")}
                  scf.yield
                }}
                %isKv = arith.cmpi ult, %ix, %ckvh_s : index
                scf.if %isKv {{
                  %khb = arith.muli %ix, %chd_s : index
                  %kb = arith.addi %ckbase, %khb : index
{head_norm_rope("%kb", "%chd_s", l, "k")}
                  %vb = arith.addi %cvbase, %khb : index
                  scf.for %hdi = %tx_s to %chd_s step %nlane {{
                    %ki = arith.addi %kb, %hdi : index
                    %kv = memref.load %sqkv[%m, %ki] : {QT}
                    memref.store %kv, %skc[%L{l}, %pos, %ix, %hdi] : {KVT}
                    %vi = arith.addi %vb, %hdi : index
                    %vv = memref.load %sqkv[%m, %vi] : {QT}
                    memref.store %vv, %svc[%L{l}, %pos, %ix, %hdi] : {KVT}
                  }}
                }}""",
                lanes=True,
            )
        )

        # 3: attention for one (token, query head). Scores, softmax and the
        # weighted sum of V in one task, which is how Fleet packages it
        # (paged_attention_layer is one task per request and kv head).
        #
        # A lane per key position -- what this was -- leaves `curlen` lanes
        # busy, and at a decode step curlen is the sequence so far: ten of the
        # workgroup's five hundred and twelve threads, each running a 128-deep
        # dependent FMA chain down the head. The head is the only axis with
        # width at batch one, so the wave takes one key position and its lanes
        # split the head; the waves take the key positions in turn. Same
        # arithmetic, a 128-deep chain traded for two MACs and a butterfly.
        #
        # At one wave the workgroup is the wave, so the softmax needs neither
        # the guard nor the trip through LDS to reach the threads that use it.
        if waves == 1:
            sm_open, sm_close, sm_pub, sm_get, sumref = "", "", "", "", "%sum"
        else:
            sm_open = "                scf.if %isW0 {"
            sm_close = "                }"
            sm_pub = "                  memref.store %sum, %ldsr[%c0_s] : " f"{RT}"
            sm_get = (
                "                gpu.barrier\n"
                "                %sumb = memref.load %ldsr[%c0_s] : "
                f"{RT}"
            )
            sumref = "%sumb"
        w(
            strided_stage(
                l,
                3,
                base + 3,
                "%cheads_s",
                "%nheads_t",
                f"""                %pos = arith.addi %wbase, %m : index
                %qhb = arith.muli %ix, %chd_s : index
                %hb = arith.muli %ix, %chd_s : index
                %hk = arith.divui %ix, %cgroup_s : index
                scf.for %t = %wid to %curlen step %cwaves {{
                  %dotp = scf.for %hdi = %lid to %chd_s step %nlane
                      iter_args(%s = %fzero_s) -> (f32) {{
                    %hi = arith.addi %qhb, %hdi : index
                    %qv = memref.load %sqkv[%m, %hi] : {QT}
                    %kv = memref.load %skc[%L{l}, %t, %hk, %hdi] : {KVT}
                    %mp = arith.mulf %qv, %kv : f32
                    %s2 = arith.addf %s, %mp : f32
                    scf.yield %s2 : f32
                  }}
{wave_reduce("%dotp", "%dot", "arith.addf", "dt" + str(l), 18)}
                  %scv = arith.mulf %dot, %invsqrthd_s : f32
                  // causal over the window: token m sees the prefix and the
                  // window entries up to and including its own
                  %okm = arith.cmpi ule, %t, %pos : index
                  %scm = arith.select %okm, %scv, %negbig_s : f32
                  // the butterfly left every lane holding the score; one of
                  // them writes it
                  scf.if %isL0 {{
                    memref.store %scm, %ssc[%m, %ix, %t] : {SCT}
                  }}
                }}
                // The softmax is over the whole row, so it waits for every
                // wave's scores -- and it rewrites the slots it reads, which is
                // why it stays on one wave instead of being repeated by all of
                // them racing over the same addresses. curlen is one lane's
                // worth, so this is two loop bodies and a broadcast.
                gpu.barrier
{sm_open}
                  %mxp = scf.for %t = %lid to %curlen step %nlane
                      iter_args(%mv = %negbig_s) -> (f32) {{
                    %v = memref.load %ssc[%m, %ix, %t] : {SCT}
                    %m2 = arith.maxnumf %mv, %v : f32
                    scf.yield %m2 : f32
                  }}
{wave_reduce("%mxp", "%mxs", "arith.maxnumf", "mx" + str(l), 18)}
                  %sump = scf.for %t = %lid to %curlen step %nlane
                      iter_args(%sm = %fzero_s) -> (f32) {{
                    %v = memref.load %ssc[%m, %ix, %t] : {SCT}
                    %dd = arith.subf %v, %mxs : f32
                    %e = math.exp %dd : f32
                    memref.store %e, %ssc[%m, %ix, %t] : {SCT}
                    %s2 = arith.addf %sm, %e : f32
                    scf.yield %s2 : f32
                  }}
{wave_reduce("%sump", "%sum", "arith.addf", "sm" + str(l), 18)}
{sm_pub}
{sm_close}
{sm_get}
                // The last loop needs no reduction at all: a thread owns an
                // output component outright, and there are more threads than
                // components, so the chain is curlen deep rather than curlen
                // times the components a lane was carrying.
                //
                // Nothing here is non-temporal. The kv cache is the one thing
                // in a decode step that is read again -- every head of a kv
                // group reads the same rows, and every step re-reads every row
                // the previous ones wrote -- and the whole cache for this
                // model is about a megabyte over all 28 layers, so it belongs
                // in L2 and nowhere else. It was marked non-temporal, which
                // sends each element to memory and back; this loop is a chain
                // of `curlen` dependent loads, so that was `curlen` memory
                // latencies end to end, per head, per layer, per step.
                scf.for %hdi = %tx_s to %chd_s step %nthr {{
                  %a = scf.for %t = %c0_s to %curlen step %c1_s
                      iter_args(%sacc = %fzero_s) -> (f32) {{
                    %e = memref.load %ssc[%m, %ix, %t] : {SCT}
                    %pv = arith.divf %e, {sumref} : f32
                    %vv = memref.load %svc[%L{l}, %t, %hk, %hdi] : {KVT}
                    %mp = arith.mulf %pv, %vv : f32
                    %s2 = arith.addf %sacc, %mp : f32
                    scf.yield %s2 : f32
                  }}
                  %oi = arith.addi %hb, %hdi : index
                  memref.store %a, %sav[%m, %oi] : {QWT}
                }}""",
                lanes="block",
            )
        )

        # 4: ao = a @ Wo
        w(
            matmul_stage(
                l,
                4,
                base + 4,
                "%saov",
                AT,
                "%sav",
                QWT,
                "%swo",
                WTB,
                "%csliceDW",
                "%cqw_s",
                qw,
                slice_dw,
                count_c="%ctasksD",
                total_c="%ntasksD_t",
            )
        )

        # 5: xa = rmsnorm(x + ao) * n2. The norm in front of the MLP is not
        # decoration: without it nothing renormalises the residual stream --
        # stage 0's rmsnorm feeds only the projection -- so the stream grows or
        # decays geometrically with depth, and since the host comparison is
        # relative, a stream in the thousands hides every error smaller than
        # itself.
        w(
            single_stage(
                l,
                5,
                base + 5,
                f"""              %spa{l} = scf.for %i = %tx_s to %cdim_s step %nthr
                  iter_args(%s = %fzero_s) -> (f32) {{
                %xv = memref.load %sx[%m, %i] : {AT}
                %avv = memref.load %saov[%m, %i] : {AT}
                %xa = arith.addf %xv, %avv : f32
                memref.store %xa, %sxa[%m, %i] : {AT}
                %sqa = arith.mulf %xa, %xa : f32
                %s2 = arith.addf %s, %sqa : f32
                scf.yield %s2 : f32
              }}
{block_sum("%spa" + str(l), "%ssa" + str(l), "a" + str(l), 14)}
              %meana{l} = arith.divf %ssa{l}, %fdim_s : f32
              %mea{l} = arith.addf %meana{l}, %eps_s : f32
              %rmsa{l} = math.sqrt %mea{l} : f32
              // The normed copy goes somewhere else: stage 8 closes the
              // residual onto the unnormalised one, which is what a decoder
              // layer does. %sr is free here -- stage 1 was the last reader.
              //
              // Same stride as the loop above, so the %sxa slot a thread
              // reads here is the one it wrote itself -- nobody waits on
              // anybody, and the block_sum above already met at a barrier.
              scf.for %i = %tx_s to %cdim_s step %nthr {{
                %xv3 = memref.load %sxa[%m, %i] : {AT}
                %nv = arith.divf %xv3, %rmsa{l} : f32
                %nw = memref.load %sn2[%L{l}, %i] : {NT}
                %xn = arith.mulf %nv, %nw : f32
                memref.store %xn, %sr[%m, %i] : {AT}
              }}""",
                lanes="block",
            )
        )

        # 6: gu = xa @ Wgu, gate and up in one matmul as Fleet fuses them.
        #
        # Fused, the piece is a slice of `inter` and holds both halves of it:
        # gate column j and up column j + inter, reduced in the same pass over
        # the same lhs, with the SwiGLU applied where they meet. The stage
        # writes `act` directly and stage 7 is not emitted. Split, the piece
        # is a slice of 2*inter and the two halves land in different pieces,
        # so the elementwise step has to be a stage, and a stage is a
        # boundary, and a boundary is 5.52 us.
        w(
            matmul_stage(
                l,
                6,
                base + 6,
                "%sgu",
                GT,
                "%sr",
                AT,
                "%swgu",
                WGTB,
                "%cslice2I",
                "%cdim_s",
                dim,
                slice_2i,
            )
        )

        # 7: SwiGLU. Elementwise, so a piece is a slice of the intermediate
        # width rather than a reduction. Not emitted when gate_up did it.
        if not fuse_swiglu:
            w(
                strided_stage(
                    l,
                    7,
                    base + 7,
                    "%ctasks",
                    "%ntasks_t",
                    f"""                %j0 = arith.muli %ix, %csliceI : index
                scf.for %jj = %tx_s to %csliceI step %nthr {{
                  %j = arith.addi %j0, %jj : index
                  %gv = memref.load %sgu[%m, %j] : {GT}
                  %ju = arith.addi %j, %cinter_s : index
                  %uv = memref.load %sgu[%m, %ju] : {GT}
                  %ng = arith.negf %gv : f32
                  %eg = math.exp %ng : f32
                  %de = arith.addf %fone_s, %eg : f32
                  %si = arith.divf %gv, %de : f32
                  %actv = arith.mulf %si, %uv : f32
                  memref.store %actv, %sact[%m, %j] : {IT}
                }}""",
                    lanes="threads",
                )
            )

        # 8, or 7 when swiglu was folded in above: x = xa + act @ Wd, the
        # residual folded into the matmul as Fleet folds it
        # (linear_with_residual_layer).
        w(
            matmul_stage(
                l,
                down_stage,
                base + down_stage,
                "%sx",
                AT,
                "%sgu" if fuse_swiglu else "%sact",
                GT if fuse_swiglu else IT,
                "%swd",
                WDTB,
                "%csliceDW",
                "%cinter_s",
                inter,
                slice_dw,
                residual="%sxa",
                count_c="%ctasksD",
                total_c="%ntasksD_t",
                lhs_silu="%cinter_s" if fuse_swiglu else None,
            )
        )

        # 9.., or 8.. when swiglu was folded in: the empty stages, if any were
        # asked for. Same claim, same
        # signal, same rendezvous, no body -- so they change the launch by the
        # cost of a stage boundary and by nothing else, and the token stream
        # they produce is still the right one.
        for p in range(pad_stages):
            w(
                strided_stage(
                    l,
                    down_stage + 1 + p,
                    base + down_stage + 1 + p,
                    "%ctasks",
                    "%ntasks_t",
                    "",
                    strip=pad_strip,
                )
            )

    # final norm, lm head, and Fleet's two-stage argmax
    w(
        single_stage(
            "x",
            stages + 1,
            base_x + 1,
            f"""              %fp = scf.for %i = %tx_s to %cdim_s step %nthr
                  iter_args(%a = %fzero_s) -> (f32) {{
                %v = memref.load %sx[%m, %i] : {AT}
                %fq = arith.mulf %v, %v : f32
                %a2 = arith.addf %a, %fq : f32
                scf.yield %a2 : f32
              }}
{block_sum("%fp", "%fs", "f", 14)}
              %fm = arith.divf %fs, %fdim_s : f32
              %fme = arith.addf %fm, %eps_s : f32
              %fr = math.sqrt %fme : f32
              scf.for %i = %tx_s to %cdim_s step %nthr {{
                %v = memref.load %sx[%m, %i] : {AT}
                %nv = arith.divf %v, %fr : f32
                %nw = memref.load %snf[%i] : {NFT}
                %o = arith.mulf %nv, %nw : f32
                memref.store %o, %sr[%m, %i] : {AT}
              }}""",
            slot=base_x + 1,
            lc="%L0",
            lanes="block",
        )
    )
    w(
        strided_stage(
            "x",
            stages + 2,
            base_x + 2,
            "%ctasksV",
            "%ntasksV_t",
            f"""                %v0 = arith.muli %ix, %csliceV : index
                scf.for %jj = %tx_s to %csliceV step %nthr {{
                  %v = arith.addi %v0, %jj : index
                  %vok = arith.cmpi ult, %v, %cvocab_s : index
                  scf.if %vok {{
                    %a = scf.for %i = %c0_s to %cdim_s step %c1_s
                        iter_args(%sacc = %fzero_s) -> (f32) {{
                      %xv = memref.load %sr[%m, %i] : {AT}
                      %wb = memref.load %swlm[%v, %i] : {LMTB}
                      %wv = arith.extf %wb : bf16 to f32
                      %mp = arith.mulf %xv, %wv : f32
                      %s2 = arith.addf %sacc, %mp : f32
                      scf.yield %s2 : f32
                    }}
                    memref.store %a, %slg[%m, %v] : {LGT}
                  }}
                }}""",
            slot=base_x + 2,
            lc="%L0",
            lanes="threads",
        )
    )
    # argmax_partial_layer: the best in this piece of the vocabulary
    w(
        strided_stage(
            "x",
            stages + 3,
            base_x + 3,
            "%ctasksV",
            "%ntasksV_t",
            f"""                %v0 = arith.muli %ix, %csliceV : index
                %bi:2 = scf.for %jj = %c0_s to %csliceV step %c1_s
                    iter_args(%bv = %negbig_s, %bidx = %zero_s) -> (f32, i32) {{
                  %v = arith.addi %v0, %jj : index
                  %vok = arith.cmpi ult, %v, %cvocab_s : index
                  %lv = scf.if %vok -> (f32) {{
                    %lvr = memref.load %slg[%m, %v] : {LGT}
                    scf.yield %lvr : f32
                  }} else {{
                    scf.yield %negbig_s : f32
                  }}
                  %gt = arith.cmpf ogt, %lv, %bv : f32
                  %nv2 = arith.select %gt, %lv, %bv : f32
                  %vi = arith.index_cast %v : index to i32
                  %nx = arith.select %gt, %vi, %bidx : i32
                  scf.yield %nv2, %nx : f32, i32
                }}
                memref.store %bi#0, %spv[%m, %ix] : {PVT}
                memref.store %bi#1, %spi[%m, %ix] : {PIT}""",
            slot=base_x + 3,
            lc="%L0",
        )
    )
    # argmax_reduce_layer: the best across pieces, ties to the lower index
    w(
        single_stage(
            "x",
            stages + 4,
            base_x + 4,
            f"""              %rd:2 = scf.for %k = %c0_s to %ctasksV step %c1_s
                  iter_args(%bv = %negbig_s, %bidx = %zero_s) -> (f32, i32) {{
                %apv = memref.load %spv[%m, %k] : {PVT}
                %api = memref.load %spi[%m, %k] : {PIT}
                %gt = arith.cmpf ogt, %apv, %bv : f32
                %nv2 = arith.select %gt, %apv, %bv : f32
                %nx = arith.select %gt, %api, %bidx : i32
                scf.yield %nv2, %nx : f32, i32
              }}
              // One next token per sequence: the argmax of the last active
              // position, and only once that slot is past the prompt. Fleet
              // guards the same write (persistent_kernel.cuh:396).
              %lastm = arith.subi %nat, %c1_s : index
              %isLastRow = arith.cmpi eq, %m, %lastm : index
              scf.if %isLastRow {{
                %nxt = arith.addi %seq, %nat : index
                %past = arith.cmpi sge, %nxt, %plen_s : index
                scf.if %past {{
                  memref.store %rd#1, %stok[%nxt] : {TKT}
                }}
              }}""",
            slot=base_x + 4,
            lc="%L0",
        )
    )

    w(f"""
            %seqn = arith.addi %seq, %nat : index
            scf.yield %seqn : index
          }}
{timer_launch1}
          scf.yield
        }}

        // The herd's x extent is the wavefront count: the lowering multiplies
        // it by the wave size to get blockDim.x (AIRToROCDLPass.cpp:1066). At
        // one tile the block is one wave, which is what lets the claim ride on
        // rocdl.readfirstlane; past that it goes through LDS instead.
        air.herd @herd tile (%htx, %hty) in (%ntx=%cwaves, %nty=%c1_s) {{
        }}
      }}
    }}
    return
  }}
}}
""")
    return "".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument(
        "--inter", type=int, default=0, help="MLP intermediate width; defaults to 2*dim"
    )
    ap.add_argument("--heads", type=int, default=4, help="query heads")
    ap.add_argument(
        "--kv-heads",
        type=int,
        default=2,
        help="key/value heads; heads must be a multiple of this",
    )
    ap.add_argument(
        "--tasks",
        type=int,
        default=8,
        help="tasks per matmul stage; dim, inter and the qkv width "
        "must divide by this",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=32,
        help="resident workgroups; must fit the device at once",
    )
    ap.add_argument(
        "--cache", type=int, default=32, help="KV cache prefix the window attends"
    )
    ap.add_argument(
        "--tokens",
        type=int,
        default=1,
        help="tokens in flight (M). 1 is a decode step; >1 is a "
        "prefill or speculative window, and is what makes the "
        "M-major traversal do anything",
    )
    ap.add_argument(
        "--head-dim",
        type=int,
        default=0,
        help="head dim; defaults to dim/heads. Qwen3 states it "
        "separately and it is not dim/heads",
    )
    ap.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help="rope base; Qwen3-0.6B uses 1e6",
    )
    ap.add_argument(
        "--vocab", type=int, default=256, help="vocabulary size; must divide by tasks"
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=1,
        help="decode steps in one launch. Each step appends its "
        "own window to the KV cache and attends everything up "
        "to it, so the attention length is a runtime value",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default=None,
        help="directory holding manifest.json and weights.f32 "
        "written by weights.py; takes every shape but "
        "--layers from the checkpoint config",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="comma-separated prompt token ids; at most --tokens "
        "of them, and they set --prompt-len",
    )
    ap.add_argument(
        "--prompt-len",
        type=int,
        default=0,
        help="how many of the run's tokens are prompt. Steps "
        "consume the prompt --tokens at a time and then "
        "decode one token each. Default (0) makes the whole "
        "run prompt, so every step is a prefill chunk.",
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="how many times to launch the chain, for timing",
    )
    ap.add_argument(
        "--nt",
        action="store_true",
        help="load the weights non-temporally. Off by default: it "
        "measured 7%% slower, and Fleet's batch-1 build emits "
        "no nt either",
    )
    ap.add_argument(
        "--dies",
        type=int,
        default=8,
        help="per-chiplet task queues; must be >= the device's "
        "chiplet count (8 on MI300X and MI350X). Every stage "
        "probes this many queues, so a value larger than the "
        "hardware costs time in every stage",
    )
    ap.add_argument(
        "--dynamic-claim",
        action="store_true",
        help="hand the pieces out from per-chiplet queues instead of "
        "partitioning them by chiplet rank. Measured 1.60x slower on "
        "Qwen3-0.6B at 128 workers, because the queue costs more than the "
        "work it hands out; it is kept because it is the form that tolerates "
        "a chiplet the dispatcher gave no workgroups to",
    )
    ap.add_argument(
        "--reduce-unroll",
        type=int,
        default=8,
        help="how many weight loads a lane issues before it waits "
        "for the first. The matmul inner loop is one memory "
        "latency per iteration overlapped with nothing, and "
        "this is the only way to hold more of the machine's "
        "bandwidth open without more workgroups. Rounded down "
        "to a power of two that divides the trip count, per "
        "stage; the accumulate order is unchanged, so the "
        "output is bit for bit the rolled loop's. Measured "
        "1/2/4/8/16 on MI350X: 8 is the knee and 16 gives the "
        "registers back to spills and lands on 1's numbers",
    )
    ap.add_argument(
        "--pad-stages",
        type=int,
        default=0,
        help="add this many empty stages to every layer. They claim, signal "
        "and wait like any other stage and compute nothing, so the tokens are "
        "unchanged and the launch grows by the cost of a stage boundary times "
        "the number added. The slope of the launch clock against this is how "
        "a boundary was priced at 5.52 us, with the per-stage timers off, so "
        "it does not depend on the instrument that first suggested the "
        "number. Pair it with --pad-strip to price the pieces",
    )
    ap.add_argument(
        "--stage-lhs",
        action="store_true",
        help="read a matmul's reduction operand into LDS once per workgroup "
        "and let every lane read it from there, instead of each lane "
        "streaming the whole of it out of global memory. **14.7%% slower**, "
        "which is what says the redundant reads were never expensive: the "
        "activation is 4 to 12 KB and every one of those loads was an L2 "
        "hit. Keep it for --fuse-swiglu, which needs it and is 3.4%% faster "
        "on top of it, and for the record that the traffic argument is dead",
    )
    ap.add_argument(
        "--fuse-swiglu",
        action="store_true",
        help="apply the SwiGLU inside `down`, on the activation as it is "
        "read, so swiglu is not a stage and a layer has eight rather than "
        "nine -- which is the fusion Fleet does. Correct, and **34.7%% "
        "slower**, because AIR's matmul has every lane load the activation "
        "from global memory: the SwiGLU is then recomputed once per output "
        "column, 3.1M transcendentals a layer instead of 3072. Fleet's "
        "version stages the activation through LDS and applies the SwiGLU "
        "once on the way in. Kept because the missing piece it points at -- "
        "LDS staging of the left-hand side -- is worth more than the fusion",
    )
    ap.add_argument(
        "--split-arrival",
        action="store_true",
        help="keep a die's piece count and its arrival count in separate "
        "words, so signalling is an add to each plus -- for whoever turns "
        "out to be last -- an acquire load to read the sum back, rather than "
        "one 64-bit atomic whose return value answers both. Three memory "
        "operations on the boundary's critical path instead of one, and "
        "2.7%% slower on Qwen3-0.6B at 128 workers",
    )
    ap.add_argument(
        "--spin-sleep",
        type=int,
        default=16,
        help="idle the waiting wave for this many units of 64 clocks between "
        "polls of the event word; 0 is the tightest loop the hardware will "
        "run. Swept 0/1/2/4/8/16 on Qwen3-0.6B at 128 workers: 1 and 2 are "
        "worse, 4 breaks even, 8 and 16 are 0.7-0.8%% better and flattening. "
        "Small, and the size is the point -- see the note at spin_wait",
    )
    ap.add_argument(
        "--acquire-per-wave",
        action="store_true",
        help="take the post-rendezvous acquire fence in every wave after the "
        "barrier instead of once in the wave that waited. That fence is "
        "`buffer_inv sc0 sc1` and it was 60%% of what a stage boundary cost; "
        "taking it once is 1.48x on Qwen3-0.6B at 128 workers. Kept because "
        "it is the form that does not depend on every wave of a workgroup "
        "sharing a cache with the one that fenced",
    )
    ap.add_argument(
        "--acquire-agent",
        action="store_true",
        help="ask for agent scope on the post-rendezvous acquire fence, "
        "`buffer_inv sc1` rather than `buffer_inv sc0 sc1`. Every workgroup "
        "is on one device, so that is the scope the protocol needs -- but it "
        "measures as nothing either way round, so it is not the default and "
        "not worth re-measuring",
    )
    ap.add_argument(
        "--round-robin-claim",
        action="store_true",
        help="give chiplet d the pieces congruent to d rather than a "
        "contiguous block of them. **3.2%% slower**, and not for the reason "
        "the blocked mapping was written for: it was meant to stop eight "
        "XCDs each filling the same 128-byte line to use an eighth of it, "
        "which predicted that down and o_proj would move most. They did not "
        "move at all -- gate_up did, by 13.2%%, and gate_up is the one stage "
        "whose piece is 48 columns, 96 bytes, the one width in this model "
        "that is not a divisor of a line. Blocking makes a die's range 12 "
        "whole lines and the straddles internal. The MALL absorbs the "
        "cross-XCD duplication; misalignment it cannot",
    )
    ap.add_argument(
        "--weights-reduction-major",
        action="store_true",
        help="store the bf16 weights the device reads as [reduction][output] "
        "rather than [output][reduction], and give each lane a strided slice "
        "of the reduction rather than a contiguous one. **5.7%% slower.** "
        "Under it a lane walking its own reduction strides by the full "
        "width, so its `unroll` loads in flight are `unroll` separate cache "
        "lines with two bytes taken from each, and a wave covers `cols * 2` "
        "bytes of every line it touches -- 16 for anything dim-wide at 128 "
        "tasks. The default is Fleet's layout "
        "(linear_ck_mi300.cuh:406-408), which is also the one this program "
        "already gave the lm head alone. The two orders sum the reduction "
        "differently, so their results are not bit for bit each other's; "
        "what says both are right is the token check",
    )
    ap.add_argument(
        "--lds-klanes",
        action="store_true",
        help="send the partials of all `waves * klanes` lanes that share an "
        "output column through LDS, for lane c of wave 0 to walk in one "
        "dependent chain, rather than folding the lanes with an xor "
        "butterfly first and sending one partial a wave. **3.2%% slower**, "
        "and 5.8%% on the two stages whose chain it lengthens most. The "
        "chain is what set the spread in achieved bandwidth across the "
        "matmuls: its length over the weights a lane loads in the same "
        "column block orders all five classes, where cache-line coverage "
        "and bytes in flight did not",
    )
    ap.add_argument(
        "--half-dim-tasks",
        action="store_true",
        help="split o_proj and down `tasks // 2` ways into pieces twice as "
        "wide, idling half the workgroups, rather than `tasks` ways like "
        "every other stage. **6.0%% slower.** It was the better trade under "
        "[reduction][output], where the piece width was also the cache-line "
        "coverage -- 8 columns is 16 bytes of a 128-byte line and 16 columns "
        "is 32. Under [output][reduction] a lane walks a row, the width does "
        "not touch coverage, and the workgroups are worth more than the "
        "width",
    )
    ap.add_argument(
        "--count-flushes",
        action="store_true",
        help="count the device-scope event flushes and print the total next "
        "to what signalling per worker would have cost. One extra "
        "device-scope atomic per die per stage, 2056 a step, read by nothing "
        "but that print -- an instrument, not part of the protocol, so it is "
        "off by default",
    )
    ap.add_argument(
        "--pad-strip",
        type=int,
        default=0,
        help="how much of the stage boundary to leave out of a pad stage. "
        "0 all of it, 1 no acquire fence, 2 no rendezvous, 3 no signal, "
        "4 no claim. A pad stage publishes nothing and nothing waits on its "
        "event, so every level still produces the right tokens, and the "
        "difference between two --pad-stages slopes is the price of what was "
        "removed",
    )
    ap.add_argument(
        "--timers-total-only",
        action="store_true",
        help="with --timers, report only the whole launch and emit no "
        "per-stage clock reads. This is what says the per-stage reads are "
        "nearly free -- 0.5%% of the launch -- and it is the mode to use with "
        "--pad-stages, where the per-class table would only be in the way",
    )
    ap.add_argument(
        "--timers",
        action="store_true",
        help="accumulate per-operator device ticks and print them; "
        "adds two s_memrealtime per stage, so measure without it",
    )
    ap.add_argument(
        "--waves",
        type=int,
        default=1,
        help="wavefronts per workgroup (the herd's x extent). "
        "Above 1 the waves split the reduction of each matmul "
        "and meet in LDS, which is the only way a decode step "
        "can use more of the machine than its output width",
    )
    ap.add_argument(
        "--wave",
        type=int,
        default=64,
        help="lanes per wavefront. A task body is split across "
        "these and its reductions closed over them, so this "
        "must equal what -air-to-rocdl{wave-size=} uses",
    )
    a = ap.parse_args()
    W = None
    if a.weights:
        import json
        from pathlib import Path

        W = json.loads((Path(a.weights) / "manifest.json").read_text())
        c = W["config"]
        # Everything but the layer count comes from the checkpoint; the layer
        # count stays a knob so a short model can be run for speed, which is
        # what Fleet's --num-layers is for too.
        a.dim, a.inter = c["dim"], c["inter"]
        a.heads, a.kv_heads = c["heads"], c["kv_heads"]
        a.head_dim, a.vocab = c["head_dim"], c["vocab"]
        a.rope_theta = c["rope_theta"]
        assert (
            a.layers <= c["layers"]
        ), f"checkpoint has {c['layers']} layers, asked for {a.layers}"
        assert a.cache == 0, (
            "--weights needs --cache 0: a synthetic KV prefix is not something "
            "the model produced, so the tokens would not mean anything"
        )
    prompt = [int(x) for x in a.prompt.split(",")] if a.prompt else None
    sys.stdout.write(
        # By keyword, all of them. This was positional and a flag added in
        # the middle of the signature silently took the value of the one
        # after it -- the generator ran, the tests passed, and the flag was
        # on when it was asked to be off. Thirty-eight arguments is past
        # where a reader can check an order by eye.
        emit(
            layers=a.layers,
            dim=a.dim,
            tasks=a.tasks,
            workers=a.workers,
            repeat=a.repeat,
            cache=a.cache,
            tokens=a.tokens,
            inter=a.inter,
            heads=a.heads,
            kv_heads=a.kv_heads,
            steps=a.steps,
            vocab=a.vocab,
            head_dim=a.head_dim,
            rope_theta=a.rope_theta,
            W=W,
            prompt=prompt,
            prompt_len=a.prompt_len,
            wave=a.wave,
            waves=a.waves,
            nt_weights=a.nt,
            timers=a.timers,
            dies=a.dies,
            unroll=a.reduce_unroll,
            stage_timers=not a.timers_total_only,
            static_claim=not a.dynamic_claim,
            pad_stages=a.pad_stages,
            pad_strip=a.pad_strip,
            acquire_once=not a.acquire_per_wave,
            spin_sleep=a.spin_sleep,
            pack_arrival=not a.split_arrival,
            fuse_swiglu=a.fuse_swiglu,
            stage_lhs=a.stage_lhs,
            acquire_agent=a.acquire_agent,
            blocked_claim=not a.round_robin_claim,
            out_major=not a.weights_reduction_major,
            fold_klanes=not a.lds_klanes,
            full_dim_tasks=not a.half_dim_tasks,
            count_flushes=a.count_flushes,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

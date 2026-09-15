#!/usr/bin/env python3
"""Emit a megakernel decode chain as AIR MLIR.

Qwen has 36 layers of several stages each. Nobody writes that by hand, and the
hand-written tests next to this one stop being a plan the moment the layer
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


def emit(layers: int, dim: int, tasks: int, workers: int, repeat: int = 1,
         cache: int = 32, tokens: int = 1, inter: int = 0,
         heads: int = 4, kv_heads: int = 2, steps: int = 1,
         vocab: int = 256) -> str:
    inter = inter or 2 * dim
    assert dim % heads == 0, "dim must divide evenly into heads"
    assert heads % kv_heads == 0, "heads must be a multiple of kv-heads"
    hd = dim // heads          # head dim
    h2 = hd // 2               # rope pairs
    group = heads // kv_heads  # query heads per kv head
    qkvo = (heads + 2 * kv_heads) * hd
    # The cache holds the prefix plus every window the run will append. Step s
    # writes slots [cache + s*tokens, cache + (s+1)*tokens) and attends
    # everything up to its own, so the attention length is a runtime value.
    total = cache + steps * tokens
    assert hd % 2 == 0, "head dim must be even for rope"
    for n, v in (("dim", dim), ("inter", inter), ("2*inter", 2 * inter),
                 ("qkv out", qkvo), ("vocab", vocab)):
        assert v % tasks == 0, f"{n} ({v}) must divide evenly into tasks"
    slice_d = dim // tasks
    slice_i = inter // tasks
    slice_2i = (2 * inter) // tasks
    slice_q = qkvo // tasks
    slice_v = vocab // tasks
    invsqrthd = hd ** -0.5
    # Activations carry a token dimension; weights do not. That asymmetry is
    # the whole reason M > 1 changes the traversal: a weight block is worth
    # reading once and using `tokens` times.
    AT = f"memref<{tokens}x{dim}xf32>"
    IT = f"memref<{tokens}x{inter}xf32>"
    GT = f"memref<{tokens}x{2 * inter}xf32>"
    QT = f"memref<{tokens}x{qkvo}xf32>"
    SCT = f"memref<{tokens}x{heads}x{total}xf32>"
    WQT = f"memref<{layers}x{dim}x{qkvo}xf32>"
    WT = f"memref<{layers}x{dim}x{dim}xf32>"
    WGT = f"memref<{layers}x{dim}x{2 * inter}xf32>"
    WDT = f"memref<{layers}x{inter}x{dim}xf32>"
    KVT = f"memref<{layers}x{total}x{kv_heads}x{hd}xf32>"
    NT = f"memref<{layers}x{dim}xf32>"
    QKNT = f"memref<{layers}x{2 * hd}xf32>"
    ROT = f"memref<{total}x{2 * hd}xf32>"
    EMT = f"memref<{vocab}x{dim}xf32>"
    LMT = f"memref<{dim}x{vocab}xf32>"
    LGT = f"memref<{tokens}x{vocab}xf32>"
    PVT = f"memref<{tokens}x{tasks}xf32>"
    PIT = f"memref<{tokens}x{tasks}xi32>"
    TKT = f"memref<{steps + 1}x{tokens}xi32>"
    NFT = f"memref<{dim}xf32>"

    stages = 9
    # Around the layers: embed at the front, then the final norm, the lm head
    # and Fleet's two-stage argmax (argmax_partial_layer + argmax_reduce_layer,
    # builder.py:799-811). Their slots sit past the layers' in the same
    # per-step block.
    extras = 5
    per_step = layers * stages + extras
    events = per_step * steps
    maxdies = 16
    qslots = (stages + extras) * maxdies
    QUT = f"memref<{steps}x{layers}x{qslots}xi32>"
    slots = steps * per_step * maxdies
    flushword = 2 * slots
    locwords = flushword + 1
    strided_stages = 7  # every layer stage but 0 and 5
    # embed, lm head and the partial argmax are split by piece too; the final
    # norm and the argmax reduce are single-task.
    naive = workers * (strided_stages * layers + 3) * steps

    s_qkv = _scale(MOD_QKV, dim)
    s_o = _scale(MOD_O, dim)
    s_g = _scale(MOD_G, dim)
    s_u = _scale(MOD_U, dim)
    s_d = _scale(MOD_D, inter)
    s_k = _scale(MOD_K, cache, _math.sqrt(cache))
    s_v = _scale(MOD_V, cache, _math.sqrt(cache))

    o = []
    w = o.append

    w(f"""// Generated by test/gpu/megakernel_gen/gen.py -- do not edit.
//   layers={layers} dim={dim} inter={inter} heads={heads} kv_heads={kv_heads}
//   head_dim={hd} tasks/stage={tasks} workers={workers} tokens={tokens}
//   kv prefix={cache} attention length={total}
//
// A decode chain as a megakernel: one launch, {layers} layers, every stage
// boundary an event rather than a return to the host.
module {{
  func.func @main() {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cdim = arith.constant {dim} : index
    %cinter = arith.constant {inter} : index
    %c2inter = arith.constant {2 * inter} : index
    %cqkvo = arith.constant {qkvo} : index
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

    %X = memref.alloc() : {AT}
    %X0 = memref.alloc() : {AT}
    %Rv = memref.alloc() : {AT}
    %QKV = memref.alloc() : {QT}
    %Sc = memref.alloc() : {SCT}
    %Av = memref.alloc() : {AT}
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
        memref.store %fzero, %Av[%m, %i] : {AT}
        memref.store %fzero, %Aov[%m, %i] : {AT}
        memref.store %fzero, %Xa[%m, %i] : {AT}
      }}
      scf.for %i = %c0 to %cqkvo step %c1 {{
        memref.store %fzero, %QKV[%m, %i] : {QT}
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
    scf.for %l = %c0 to %clayers step %c1 {{
      %ll = arith.index_cast %l : index to i32
      %fl = arith.sitofp %ll : i32 to f32
      scf.for %i = %c0 to %cdim step %c1 {{
        %ii = arith.index_cast %i : index to i32
        %fi = arith.sitofp %ii : i32 to f32
        %a0 = arith.addf %fi, %fl : f32
        %r0 = arith.remf %a0, %c3f : f32
        %d0 = arith.subf %r0, %fone : f32
        %s0 = arith.mulf %d0, %quarter : f32
        %n1v = arith.addf %fone, %s0 : f32
        memref.store %n1v, %N1[%l, %i] : {NT}
        %a1 = arith.addf %a0, %fone : f32
        %r1 = arith.remf %a1, %c5f : f32
        %d1 = arith.subf %r1, %c2f : f32
        %s1 = arith.mulf %d1, %eighth : f32
        %n2v = arith.addf %fone, %s1 : f32
        memref.store %n2v, %N2[%l, %i] : {NT}
      }}
      scf.for %d = %c0 to %chd step %c1 {{
        %dd = arith.index_cast %d : index to i32
        %fd = arith.sitofp %dd : i32 to f32
        %b0 = arith.addf %fd, %fl : f32
        %q0 = arith.remf %b0, %c3f : f32
        %q1 = arith.subf %q0, %fone : f32
        %q2 = arith.mulf %q1, %quarter : f32
        %qw = arith.addf %fone, %q2 : f32
        memref.store %qw, %QKN[%l, %d] : {QKNT}
        %b1 = arith.addf %b0, %fone : f32
        %k0 = arith.remf %b1, %c3f : f32
        %k1 = arith.subf %k0, %fone : f32
        %k2 = arith.mulf %k1, %quarter : f32
        %kw = arith.addf %fone, %k2 : f32
        %dk = arith.addi %d, %chd : index
        memref.store %kw, %QKN[%l, %dk] : {QKNT}
      }}
    }}

    // RoPE tables, cos in [0, hd) and sin in [hd, 2hd). Both halves of a head
    // use the same frequency, which is what makes rotate-half a rotation.
    %lnbase = arith.constant {LN10000:.10e} : f32
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
    // The prompt, and room for what each step produces.
    %csteps1 = arith.constant {steps + 1} : index
    %csliceVh = arith.constant {slice_v} : index
    scf.for %sp = %c0 to %csteps1 step %c1 {{
      scf.for %m = %c0 to %ctok step %c1 {{
        memref.store %zero, %Tok[%sp, %m] : {TKT}
      }}
    }}
    scf.for %m = %c0 to %ctok step %c1 {{
      %mm = arith.index_cast %m : index to i32
      %t0 = arith.muli %mm, %c3i : i32
      %t1 = arith.addi %t0, %one : i32
      %t2 = arith.remsi %t1, %cvocabi : i32
      memref.store %t2, %Tok[%c0, %m] : {TKT}
    }}
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
        scf.for %j = %c0 to %cdim step %c1 {{
          %jj = arith.index_cast %j : index to i32
          %fj = arith.sitofp %jj : i32 to f32
          %d = arith.subf %fi, %fj : f32
          %d2 = arith.addf %d, %fl : f32
          %d3 = arith.addf %d2, %fone : f32
          %ro = arith.remf %d3, %c{MOD_O}f : f32
          memref.store %ro, %Wo[%l, %i, %j] : {WT}
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
      // The prefix of the KV cache. The window entries are written by the
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

    for nm, val in (("sqkv", s_qkv), ("so", s_o), ("sg", s_g), ("su", s_u),
                    ("sd", s_d), ("sk", s_k), ("sv", s_v)):
        w(f"\n    %{nm}c = arith.constant {val:.10e} : f32")
    w(centre("%Wqkv", WQT, "%cdim", "%fred_d", "%cqkvo", "%sqkvc", "wq"))
    w(centre("%Wo", WT, "%cdim", "%fred_d", "%cdim", "%soc", "wo"))
    # Wgu holds the gate half and the up half side by side and they come from
    # different moduli, so the scale is selected per column.
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
    # The KV prefix is 4-D and only its prefix rows exist, so it gets its own
    # loop rather than the helper's.
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

    # ---- host reference ----
    w(f"""

    %invsqrthd = arith.constant {invsqrthd:.8e} : f32
    %negbig = arith.constant -1.000000e30 : f32
    %rq = memref.alloc() : memref<{qkvo}xf32>
    %rsc = memref.alloc() : memref<{total}xf32>
    %ra = memref.alloc() : memref<{dim}xf32>
    %rao = memref.alloc() : memref<{dim}xf32>
    %rxa = memref.alloc() : memref<{dim}xf32>
    %rgu = memref.alloc() : memref<{2 * inter}xf32>
    %ract = memref.alloc() : memref<{inter}xf32>
    scf.for %sp = %c0 to %csteps step %c1 {{
     // Step s appends its window at cache + s*tokens and attends everything up
     // to and including its own entry, so both the slot and the length move
     // with the step. Fleet does the same thing by advancing
     // config.step[request_id] on the device (persistent_kernel.cuh:392).
     %spt = arith.muli %sp, %ctok : index
     %wbase = arith.addi %cpre, %spt : index
     %curlen = arith.addi %wbase, %ctok : index
     // embed: this step's input is the token the last step produced
     scf.for %m = %c0 to %ctok step %c1 {{
       %tk = memref.load %Tok[%sp, %m] : {TKT}
       %tki = arith.index_cast %tk : i32 to index
       scf.for %i = %c0 to %cdim step %c1 {{
         %ev = memref.load %Emb[%tki, %i] : {EMT}
         memref.store %ev, %ref[%m, %i] : {AT}
       }}
     }}
     scf.for %l = %c0 to %clayers step %c1 {{
      // 0, 1, 2 for every token before any attention, because token m's scores
      // read what tokens before it appended.
      scf.for %m = %c0 to %ctok step %c1 {{
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
           %wv = memref.load %Wqkv[%l, %i, %n] : {WQT}
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
      scf.for %m = %c0 to %ctok step %c1 {{
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
           memref.store %acc, %ra[%oi] : memref<{dim}xf32>
         }}
       }}
       // 4: ao = a @ Wo
       scf.for %j = %c0 to %cdim step %c1 {{
         %acc = scf.for %i = %c0 to %cdim step %c1
             iter_args(%s = %fzero) -> (f32) {{
           %av = memref.load %ra[%i] : memref<{dim}xf32>
           %wv = memref.load %Wo[%l, %i, %j] : {WT}
           %mu = arith.mulf %av, %wv : f32
           %s2 = arith.addf %s, %mu : f32
           scf.yield %s2 : f32
         }}
         memref.store %acc, %rao[%j] : memref<{dim}xf32>
       }}
       // 5: xa = rmsnorm(x + ao) * n2
       %ssa = scf.for %j = %c0 to %cdim step %c1
           iter_args(%s = %fzero) -> (f32) {{
         %xv = memref.load %ref[%m, %j] : {AT}
         %av = memref.load %rao[%j] : memref<{dim}xf32>
         %xa = arith.addf %xv, %av : f32
         memref.store %xa, %rxa[%j] : memref<{dim}xf32>
         %sqa = arith.mulf %xa, %xa : f32
         %s2 = arith.addf %s, %sqa : f32
         scf.yield %s2 : f32
       }}
       %meana = arith.divf %ssa, %fdim : f32
       %mea = arith.addf %meana, %eps : f32
       %rmsa = math.sqrt %mea : f32
       scf.for %j = %c0 to %cdim step %c1 {{
         %xv = memref.load %rxa[%j] : memref<{dim}xf32>
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
           %wv = memref.load %Wgu[%l, %i, %p] : {WGT}
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
           %wv = memref.load %Wd[%l, %p, %j] : {WDT}
           %mu = arith.mulf %av, %wv : f32
           %s2 = arith.addf %s, %mu : f32
           scf.yield %s2 : f32
         }}
         %xv = memref.load %rxa[%j] : memref<{dim}xf32>
         %nx = arith.addf %xv, %acc : f32
         memref.store %nx, %ref[%m, %j] : {AT}
       }}
      }}
     }}
     // final norm, lm head, then argmax the way Fleet does it: a partial per
     // piece, then a reduce over the pieces
     scf.for %m = %c0 to %ctok step %c1 {{
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
           %wv = memref.load %Wlm[%i, %v] : {LMT}
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
           %lv = memref.load %Lg[%m, %v] : {LGT}
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
       %spn = arith.addi %sp, %c1 : index
       memref.store %rd#1, %Tok[%spn, %m] : {TKT}
     }}
    }}
    // The reference used these as scratch; hand the device zeroed copies.
    scf.for %m = %c0 to %ctok step %c1 {{
      scf.for %i = %c0 to %cdim step %c1 {{
        memref.store %fzero, %Rv[%m, %i] : {AT}
      }}
      scf.for %i = %c0 to %cqkvo step %c1 {{
        memref.store %fzero, %QKV[%m, %i] : {QT}
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
    %dAv = gpu.alloc () : {AT}
    %dAov = gpu.alloc () : {AT}
    %dXa = gpu.alloc () : {AT}
    %dGU = gpu.alloc () : {GT}
    %dActv = gpu.alloc () : {IT}
    %dWqkv = gpu.alloc () : {WQT}
    %dWo = gpu.alloc () : {WT}
    %dWgu = gpu.alloc () : {WGT}
    %dWd = gpu.alloc () : {WDT}
    %dKc = gpu.alloc () : {KVT}
    %dVc = gpu.alloc () : {KVT}
    %dN1 = gpu.alloc () : {NT}
    %dN2 = gpu.alloc () : {NT}
    %dQKN = gpu.alloc () : {QKNT}
    %dRO = gpu.alloc () : {ROT}
    %dEmb = gpu.alloc () : {EMT}
    %dWlm = gpu.alloc () : {LMT}
    %dNf = gpu.alloc () : {NFT}
    %dTok = gpu.alloc () : {TKT}
    %dLg = gpu.alloc () : {LGT}
    %dPV = gpu.alloc () : {PVT}
    %dPI = gpu.alloc () : {PIT}
    %dQ = gpu.alloc () : {QUT}
    %dLoc = gpu.alloc () : memref<{locwords}xi32>
    %dE = gpu.alloc () : memref<{events}xi32>
    gpu.memcpy %dX, %X : {AT}, {AT}
    gpu.memcpy %dRv, %Rv : {AT}, {AT}
    gpu.memcpy %dQKV, %QKV : {QT}, {QT}
    gpu.memcpy %dSc, %Sc : {SCT}, {SCT}
    gpu.memcpy %dAv, %Av : {AT}, {AT}
    gpu.memcpy %dAov, %Aov : {AT}, {AT}
    gpu.memcpy %dXa, %Xa : {AT}, {AT}
    gpu.memcpy %dGU, %GU : {GT}, {GT}
    gpu.memcpy %dActv, %Actv : {IT}, {IT}
    gpu.memcpy %dWqkv, %Wqkv : {WQT}, {WQT}
    gpu.memcpy %dWo, %Wo : {WT}, {WT}
    gpu.memcpy %dWgu, %Wgu : {WGT}, {WGT}
    gpu.memcpy %dWd, %Wd : {WDT}, {WDT}
    gpu.memcpy %dKc, %Kc : {KVT}, {KVT}
    gpu.memcpy %dVc, %Vc : {KVT}, {KVT}
    gpu.memcpy %dN1, %N1 : {NT}, {NT}
    gpu.memcpy %dN2, %N2 : {NT}, {NT}
    gpu.memcpy %dQKN, %QKN : {QKNT}, {QKNT}
    gpu.memcpy %dRO, %RO : {ROT}, {ROT}
    gpu.memcpy %dEmb, %Emb : {EMT}, {EMT}
    gpu.memcpy %dWlm, %Wlm : {LMT}, {LMT}
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
      scf.for %sp2 = %c0 to %csteps1 step %c1 {{
        scf.for %m = %c0 to %ctok step %c1 {{
          %keep = arith.cmpi eq, %sp2, %c0 : index
          %cur = memref.load %Tok[%sp2, %m] : {TKT}
          %nv = arith.select %keep, %cur, %zero : i32
          memref.store %nv, %Tok0[%sp2, %m] : {TKT}
        }}
      }}
      gpu.memcpy %dTok, %Tok0 : {TKT}, {TKT}
      func.call @chain(%dQ, %dE, %dX, %dRv, %dQKV, %dSc, %dAv, %dAov, %dXa,
                       %dGU, %dActv, %dWqkv, %dWo, %dWgu, %dWd, %dKc, %dVc,
                       %dN1, %dN2, %dQKN, %dRO, %dEmb, %dWlm, %dNf, %dTok,
                       %dLg, %dPV, %dPI, %dLoc)
        : ({QUT}, memref<{events}xi32>, {AT}, {AT},
           {QT}, {SCT}, {AT}, {AT}, {AT}, {GT}, {IT},
           {WQT}, {WT}, {WGT}, {WDT}, {KVT}, {KVT}, {NT}, {NT}, {QKNT}, {ROT},
           {EMT}, {LMT}, {NFT}, {TKT}, {LGT}, {PVT}, {PIT},
           memref<{locwords}xi32>) -> ()
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
    %cflush = arith.constant {flushword} : index
    %flushes = memref.load %Loc[%cflush] : memref<{locwords}xi32>
    vector.print str "device-scope event flushes = "
    vector.print %flushes : i32
    %cnaive = arith.constant {naive} : i32
    vector.print str "what signalling per worker would have been = "
    vector.print %cnaive : i32
    vector.print str "output elements differing from the reference = "
    vector.print %bad : i32
    // The token check is exact, not a tolerance: argmax turns the whole chain
    // into a discrete answer, so a token either matches or it does not.
    gpu.memcpy %Tok0, %dTok : {TKT}, {TKT}
    %tbad = scf.for %sp = %c1 to %csteps1 step %c1
        iter_args(%bo = %zero) -> (i32) {{
      %bi = scf.for %m = %c0 to %ctok step %c1
          iter_args(%b = %bo) -> (i32) {{
        %g = memref.load %Tok0[%sp, %m] : {TKT}
        %wt = memref.load %Tok[%sp, %m] : {TKT}
        %eq = arith.cmpi eq, %g, %wt : i32
        %inc = arith.select %eq, %zero, %one : i32
        %b2 = arith.addi %b, %inc : i32
        scf.yield %b2 : i32
      }}
      scf.yield %bi : i32
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
    scf.for %sp = %c1 to %csteps1 step %c1 {{
      scf.for %m = %c0 to %ctok step %c1 {{
        %tv = memref.load %Tok[%sp, %m] : {TKT}
        vector.print %tv : i32
      }}
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

    layer_consts = "\n".join(
        f"        %L{i} = arith.constant {i} : index" for i in range(layers))
    w(f"""
  func.func @chain(%Q: {QUT}, %E: memref<{events}xi32>,
                   %X: {AT}, %Rv: {AT}, %QKV: {QT}, %Sc: {SCT},
                   %Av: {AT}, %Aov: {AT}, %Xa: {AT}, %GU: {GT}, %Actv: {IT},
                   %Wqkv: {WQT}, %Wo: {WT}, %Wgu: {WGT}, %Wd: {WDT},
                   %Kc: {KVT}, %Vc: {KVT}, %N1: {NT}, %N2: {NT},
                   %QKN: {QKNT}, %RO: {ROT}, %Emb: {EMT}, %Wlm: {LMT},
                   %Nf: {NFT}, %Tok: {TKT}, %Lg: {LGT}, %PV: {PVT},
                   %PI: {PIT}, %Loc: memref<{locwords}xi32>) {{
    %c1 = arith.constant 1 : index
    %cw = arith.constant {workers} : index
    air.launch (%bx, %by) in (%nbx=%cw, %nby=%c1)
        args(%q=%Q, %eb=%E, %x=%X, %r=%Rv, %qkv=%QKV, %scb=%Sc, %av=%Av,
             %aov=%Aov, %xab=%Xa, %gub=%GU, %actb=%Actv, %wqkv=%Wqkv,
             %wo=%Wo, %wgu=%Wgu, %wd=%Wd, %kc=%Kc, %vc=%Vc, %n1=%N1,
             %n2=%N2, %qkn=%QKN, %ro=%RO, %emb=%Emb, %wlm=%Wlm, %nf=%Nf,
             %tok=%Tok, %lg=%Lg, %pvb=%PV, %pib=%PI, %loc=%Loc)
        : {QUT}, memref<{events}xi32>, {AT}, {AT},
          {QT}, {SCT}, {AT}, {AT}, {AT}, {GT}, {IT},
          {WQT}, {WT}, {WGT}, {WDT}, {KVT}, {KVT}, {NT}, {NT}, {QKNT}, {ROT},
          {EMT}, {LMT}, {NFT}, {TKT}, {LGT}, {PVT}, {PIT},
          memref<{locwords}xi32> {{
      air.segment @worker args(%sq=%q, %se=%eb, %sx=%x, %sr=%r, %sqkv=%qkv,
                               %ssc=%scb, %sav=%av, %saov=%aov, %sxa=%xab,
                               %sgu=%gub, %sact=%actb, %swqkv=%wqkv,
                               %swo=%wo, %swgu=%wgu, %swd=%wd, %skc=%kc,
                               %svc=%vc, %sn1=%n1, %sn2=%n2, %sqkn=%qkn,
                               %sro=%ro, %semb=%emb, %swlm=%wlm, %snf=%nf,
                               %stok=%tok, %slg=%lg, %spv=%pvb, %spi=%pib,
                               %sloc=%loc)
          : {QUT}, memref<{events}xi32>, {AT}, {AT},
            {QT}, {SCT}, {AT}, {AT}, {AT}, {GT}, {IT},
            {WQT}, {WT}, {WGT}, {WDT}, {KVT}, {KVT}, {NT}, {NT}, {QKNT}, {ROT},
            {EMT}, {LMT}, {NFT}, {TKT}, {LGT}, {PVT}, {PIT},
            memref<{locwords}xi32> {{
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %ctasks = arith.constant {tasks} : index
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
        %ckbase = arith.constant {heads * hd} : index
        %cvbase = arith.constant {(heads + kv_heads) * hd} : index
        %one_s = arith.constant 1 : i32
        %zero_s = arith.constant 0 : i32
        %ctok_s = arith.constant {tokens} : index
        // A stage's pieces are the (token, n) grid, so the event it signals
        // counts tokens * n.
        %ntasks_t = arith.constant {tokens * tasks} : i32
        %nheads_t = arith.constant {tokens * heads} : i32
        %n1_s = arith.constant 1 : i32
        %fzero_s = arith.constant 0.0 : f32
        %fone_s = arith.constant 1.0 : f32
        %eps_s = arith.constant 1.0e-6 : f32
        %fdim_s = arith.constant {float(dim):.6e} : f32
        %fhd_s = arith.constant {float(hd):.6e} : f32
        %invsqrthd_s = arith.constant {invsqrthd:.8e} : f32
        %negbig_s = arith.constant -1.000000e30 : f32
        %true = arith.constant true

        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index
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
        %cflushW = arith.constant {4 * flushword} : i64
        %flushP = llvm.getelementptr %locp[%cflushW] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %myrank2 = air.chiplet_block_id
        %mycnt = air.chiplet_dim_blocks
        %mycnt_i = arith.index_cast %mycnt : index to i32
        %evbase = memref.extract_aligned_pointer_as_index %se : memref<{events}xi32> -> index
        %evi = arith.index_cast %evbase : index to i64
        %evptr = llvm.inttoptr %evi : i64 to !llvm.ptr

        scf.if %isLead {{
          // One launch runs the whole decode. The task graph is the same every
          // step; what changes is the iteration it belongs to. Fleet versions
          // task identity the same way -- TaskId is
          // (iteration_num << 32) | position_index
          // (persistent_kernel.cuh:263) -- so a step gets fresh event and queue
          // slots without anything having to be reset between steps.
          scf.for %step = %c0_s to %csteps_s step %c1_s {{
            %stepi = arith.index_cast %step : index to i64
            %stepev = arith.muli %stepi, %cevstep : i64
            %steploc = arith.muli %stepi, %clocstep : i64
            %spt = arith.muli %step, %ctok_s : index
            %wbase = arith.addi %cpre_s, %spt : index
            %curlen = arith.addi %wbase, %ctok_s : index""")

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
    def strided_stage(l, stage, ev, count_expr, total_const, body,
                      slot=None, lc=None):
        slot = (l * stages + stage) if slot is None else slot
        lc = f"%L{l}" if lc is None else lc
        return f"""
          // stage {stage}
          %ec{l}_{stage} = arith.constant {4 * ev} : i64
          %eo{l}_{stage} = arith.addi %stepev, %ec{l}_{stage} : i64
          %p{l}_{stage} = llvm.getelementptr %evptr[%eo{l}_{stage}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %hb{l}_{stage} = arith.constant {stage * maxdies} : index
          %t{l}_{stage} = scf.for %pp = %c0_s to %cmaxdies step %c1_s
              iter_args(%outer = %zero_s) -> (i32) {{
            // Own die first, then the others in order.
            %draw = arith.addi %mydie, %pp : index
            %d = arith.remui %draw, %cmaxdies : index
            %hidx = arith.addi %hb{l}_{stage}, %d : index
            %inner:2 = scf.while (%go = %true, %acc = %outer) : (i1, i32) -> (i1, i32) {{
              scf.condition(%go) %go, %acc : i1, i32
            }} do {{
            ^bb0(%g: i1, %acc: i32):
              %cl = memref.atomic_rmw addi %one_s, %sq[%step, {lc}, %hidx] : (i32, {QUT}) -> i32
              %k = arith.index_cast %cl : i32 to index
              // M fast, N slow: the token moves every claim, the weight block
              // only every `tokens` claims.
              %m = arith.remui %k, %ctok_s : index
              %kn = arith.divui %k, %ctok_s : index
              %kstride = arith.muli %kn, %cmaxdies : index
              %ix = arith.addi %d, %kstride : index
              %has = arith.cmpi ult, %ix, {count_expr} : index
              %acc2 = scf.if %has -> i32 {{
{body}
                %n = arith.addi %acc, %one_s : i32
                scf.yield %n : i32
              }} else {{
                scf.yield %acc : i32
              }}
              scf.yield %has, %acc2 : i1, i32
            }}
            scf.yield %inner#1 : i32
          }}
          // Two-level: add into a counter only this die touches, then let the
          // last worker on the die flush the die's whole share once. The
          // instructions are the same as signalling per worker; what changes is
          // how many times the device-scope one runs.
          %lw{l}_{stage} = arith.constant {4 * slot * maxdies} : i64
          %myd{l}_{stage} = arith.index_cast %mydie : index to i64
          %myd4{l}_{stage} = arith.muli %myd{l}_{stage}, %fourL : i64
          %lb{l}_{stage} = arith.addi %lw{l}_{stage}, %myd4{l}_{stage} : i64
          %lo{l}_{stage} = arith.addi %lb{l}_{stage}, %steploc : i64
          %locP{l}_{stage} = llvm.getelementptr %locp[%lo{l}_{stage}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %aw{l}_{stage} = arith.constant {4 * slots} : i64
          %ao{l}_{stage} = arith.addi %aw{l}_{stage}, %lo{l}_{stage} : i64
          %arrP{l}_{stage} = llvm.getelementptr %locp[%ao{l}_{stage}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %la{l}_{stage} = llvm.atomicrmw add %locP{l}_{stage}, %t{l}_{stage} syncscope("agent") monotonic : !llvm.ptr, i32
          // Release on the arrival so the accumulate above is visible to
          // whoever turns out to be last.
          %ar{l}_{stage} = llvm.atomicrmw add %arrP{l}_{stage}, %one_s syncscope("agent") release : !llvm.ptr, i32
          %last{l}_{stage} = arith.subi %mycnt_i, %one_s : i32
          %amLast{l}_{stage} = arith.cmpi eq, %ar{l}_{stage}, %last{l}_{stage} : i32
          scf.if %amLast{l}_{stage} {{
            %tot{l}_{stage} = llvm.load %locP{l}_{stage} atomic syncscope("agent") acquire {{alignment = 4 : i64}} : !llvm.ptr -> i32
            %sig{l}_{stage} = llvm.atomicrmw add %p{l}_{stage}, %tot{l}_{stage} syncscope("") release : !llvm.ptr, i32
            %cnt{l}_{stage} = llvm.atomicrmw add %flushP, %one_s syncscope("") monotonic : !llvm.ptr, i32
          }}
          scf.while : () -> () {{
            %seen = llvm.load %p{l}_{stage} atomic syncscope("") acquire {{alignment = 4 : i64}} : !llvm.ptr -> i32
            %notYet = arith.cmpi ult, %seen, {total_const} : i32
            scf.condition(%notYet)
          }} do {{
            scf.yield
          }}"""

    def single_stage(l, stage, ev, body, slot=None, lc=None):
        slot = (l * stages + stage) if slot is None else slot
        lc = f"%L{l}" if lc is None else lc
        return f"""
          // stage {stage} -- one task: a reduction over the whole row, so it
          // cannot be split by output slice the way the matmuls can.
          %hsingle{stage}_{l} = arith.constant {stage * maxdies} : index
          %ec{l}_{stage} = arith.constant {4 * ev} : i64
          %eo{l}_{stage} = arith.addi %stepev, %ec{l}_{stage} : i64
          %p{l}_{stage} = llvm.getelementptr %evptr[%eo{l}_{stage}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %cl{l}_{stage} = memref.atomic_rmw addi %one_s, %sq[%step, {lc}, %hsingle{stage}_{l}] : (i32, {QUT}) -> i32
          %mine{l}_{stage} = arith.cmpi eq, %cl{l}_{stage}, %zero_s : i32
          scf.if %mine{l}_{stage} {{
            scf.for %m = %c0_s to %ctok_s step %c1_s {{
{body}
            }}
            %sig{l}_{stage} = llvm.atomicrmw add %p{l}_{stage}, %n1_s syncscope("") release : !llvm.ptr, i32
          }}
          scf.while : () -> () {{
            %seen = llvm.load %p{l}_{stage} atomic syncscope("") acquire {{alignment = 4 : i64}} : !llvm.ptr -> i32
            %notYet = arith.cmpi ult, %seen, %n1_s : i32
            scf.condition(%notYet)
          }} do {{
            scf.yield
          }}"""

    # The weights are read once per layer and never again inside a decode step,
    # and they are far larger than anything else in flight, so they are exactly
    # what would evict the activations. Fleet loads them non-temporally too
    # (gang_ksplit_linear_mi300.cuh:88 uses amd_buffer_coherence_enum(18),
    # which is nt|sc1).
    def matmul_stage(l, stage, ev, out, outty, lhs, lhsty, wmat, wty,
                     slice_c, red_c, residual=None):
        store = (f"""
                  %rv = memref.load {residual}[%m, %j] : {outty}
                  %a2 = arith.addf %rv, %a : f32
                  memref.store %a2, {out}[%m, %j] : {outty}"""
                 if residual else f"""
                  memref.store %a, {out}[%m, %j] : {outty}""")
        return strided_stage(l, stage, ev, "%ctasks", "%ntasks_t", f"""                %j0 = arith.muli %ix, {slice_c} : index
                scf.for %jj = %c0_s to {slice_c} step %c1_s {{
                  %j = arith.addi %j0, %jj : index
                  %a = scf.for %i = %c0_s to {red_c} step %c1_s
                      iter_args(%sacc = %fzero_s) -> (f32) {{
                    %lv = memref.load {lhs}[%m, %i] : {lhsty}
                    %wv = memref.load {wmat}[%L{l}, %i, %j] {{nontemporal = true}} : {wty}
                    %mp = arith.mulf %lv, %wv : f32
                    %s2 = arith.addf %sacc, %mp : f32
                    scf.yield %s2 : f32
                  }}{store}
                }}""")

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
        return f"""                  %hs_{tag} = scf.for %hdi = %c0_s to %chd_s step %c1_s
                      iter_args(%sa_{tag} = %fzero_s) -> (f32) {{
                    %hi_{tag} = arith.addi {base}, %hdi : index
                    %hv_{tag} = memref.load %sqkv[%m, %hi_{tag}] : {QT}
                    %hq_{tag} = arith.mulf %hv_{tag}, %hv_{tag} : f32
                    %hn_{tag} = arith.addf %sa_{tag}, %hq_{tag} : f32
                    scf.yield %hn_{tag} : f32
                  }}
                  %hm_{tag} = arith.divf %hs_{tag}, %fhd_s : f32
                  %hme_{tag} = arith.addf %hm_{tag}, %eps_s : f32
                  %hr_{tag} = math.sqrt %hme_{tag} : f32
                  scf.for %hdi = %c0_s to %chd_s step %c1_s {{
                    %wi_{tag} = arith.addi %hdi, {wofs} : index
                    %hj_{tag} = arith.addi {base}, %hdi : index
                    %hw_{tag} = memref.load %sqkv[%m, %hj_{tag}] : {QT}
                    %nv_{tag} = arith.divf %hw_{tag}, %hr_{tag} : f32
                    %nw_{tag} = memref.load %sqkn[%L{l}, %wi_{tag}] : {QKNT}
                    %ov_{tag} = arith.mulf %nv_{tag}, %nw_{tag} : f32
                    memref.store %ov_{tag}, %sqkv[%m, %hj_{tag}] : {QT}
                  }}
                  scf.for %hdi = %c0_s to %ch2_s step %c1_s {{
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
    w(strided_stage("x", stages + 0, base_x + 0, "%ctasks", "%ntasks_t",
                    f"""                %tk = memref.load %stok[%step, %m] : {TKT}
                %tki = arith.index_cast %tk : i32 to index
                %j0 = arith.muli %ix, %csliceD : index
                scf.for %jj = %c0_s to %csliceD step %c1_s {{
                  %j = arith.addi %j0, %jj : index
                  %ev = memref.load %semb[%tki, %j] {{nontemporal = true}} : {EMT}
                  memref.store %ev, %sx[%m, %j] : {AT}
                }}""", slot=base_x + 0, lc="%L0"))

    for l in range(layers):
        base = stages * l
        w(f"\n          // ================= layer {l} =================")

        # 0: r = rmsnorm(x) * n1
        w(single_stage(l, 0, base + 0, f"""              %ss{l} = scf.for %i = %c0_s to %cdim_s step %c1_s
                  iter_args(%s = %fzero_s) -> (f32) {{
                %v = memref.load %sx[%m, %i] : {AT}
                %sq2 = arith.mulf %v, %v : f32
                %s2 = arith.addf %s, %sq2 : f32
                scf.yield %s2 : f32
              }}
              %mean{l} = arith.divf %ss{l}, %fdim_s : f32
              %me{l} = arith.addf %mean{l}, %eps_s : f32
              %rms{l} = math.sqrt %me{l} : f32
              scf.for %i = %c0_s to %cdim_s step %c1_s {{
                %v = memref.load %sx[%m, %i] : {AT}
                %nv = arith.divf %v, %rms{l} : f32
                %nw = memref.load %sn1[%L{l}, %i] : {NT}
                %rv = arith.mulf %nv, %nw : f32
                memref.store %rv, %sr[%m, %i] : {AT}
              }}"""))

        # 1: qkv = r @ Wqkv
        w(matmul_stage(l, 1, base + 1, "%sqkv", QT, "%sr", AT, "%swqkv", WQT,
                       "%csliceQ", "%cdim_s"))

        # 2: per-head norm, rope, and the cache append. One piece per
        # (token, query head); the kv heads ride along on the first `kv_heads`
        # of them, which is Fleet's shape too -- the update is per kv head.
        w(strided_stage(l, 2, base + 2, "%cheads_s", "%nheads_t",
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
                  scf.for %hdi = %c0_s to %chd_s step %c1_s {{
                    %ki = arith.addi %kb, %hdi : index
                    %kv = memref.load %sqkv[%m, %ki] : {QT}
                    memref.store %kv, %skc[%L{l}, %pos, %ix, %hdi] : {KVT}
                    %vi = arith.addi %vb, %hdi : index
                    %vv = memref.load %sqkv[%m, %vi] : {QT}
                    memref.store %vv, %svc[%L{l}, %pos, %ix, %hdi] : {KVT}
                  }}
                }}"""))

        # 3: attention for one (token, query head). Scores, softmax and the
        # weighted sum of V in one task, which is how Fleet packages it
        # (paged_attention_layer is one task per request and kv head).
        w(strided_stage(l, 3, base + 3, "%cheads_s", "%nheads_t",
                        f"""                %pos = arith.addi %wbase, %m : index
                %qhb = arith.muli %ix, %chd_s : index
                %hb = arith.muli %ix, %chd_s : index
                %hk = arith.divui %ix, %cgroup_s : index
                %mxs = scf.for %t = %c0_s to %curlen step %c1_s
                    iter_args(%mv = %negbig_s) -> (f32) {{
                  %dot = scf.for %hdi = %c0_s to %chd_s step %c1_s
                      iter_args(%s = %fzero_s) -> (f32) {{
                    %hi = arith.addi %qhb, %hdi : index
                    %qv = memref.load %sqkv[%m, %hi] : {QT}
                    %kv = memref.load %skc[%L{l}, %t, %hk, %hdi] {{nontemporal = true}} : {KVT}
                    %mp = arith.mulf %qv, %kv : f32
                    %s2 = arith.addf %s, %mp : f32
                    scf.yield %s2 : f32
                  }}
                  %scv = arith.mulf %dot, %invsqrthd_s : f32
                  // causal over the window: token m sees the prefix and the
                  // window entries up to and including its own
                  %okm = arith.cmpi ule, %t, %pos : index
                  %scm = arith.select %okm, %scv, %negbig_s : f32
                  memref.store %scm, %ssc[%m, %ix, %t] : {SCT}
                  %m2 = arith.maxnumf %mv, %scm : f32
                  scf.yield %m2 : f32
                }}
                %sum = scf.for %t = %c0_s to %curlen step %c1_s
                    iter_args(%sm = %fzero_s) -> (f32) {{
                  %v = memref.load %ssc[%m, %ix, %t] : {SCT}
                  %dd = arith.subf %v, %mxs : f32
                  %e = math.exp %dd : f32
                  memref.store %e, %ssc[%m, %ix, %t] : {SCT}
                  %s2 = arith.addf %sm, %e : f32
                  scf.yield %s2 : f32
                }}
                scf.for %hdi = %c0_s to %chd_s step %c1_s {{
                  %a = scf.for %t = %c0_s to %curlen step %c1_s
                      iter_args(%sacc = %fzero_s) -> (f32) {{
                    %e = memref.load %ssc[%m, %ix, %t] : {SCT}
                    %pv = arith.divf %e, %sum : f32
                    %vv = memref.load %svc[%L{l}, %t, %hk, %hdi] {{nontemporal = true}} : {KVT}
                    %mp = arith.mulf %pv, %vv : f32
                    %s2 = arith.addf %sacc, %mp : f32
                    scf.yield %s2 : f32
                  }}
                  %oi = arith.addi %hb, %hdi : index
                  memref.store %a, %sav[%m, %oi] : {AT}
                }}"""))

        # 4: ao = a @ Wo
        w(matmul_stage(l, 4, base + 4, "%saov", AT, "%sav", AT, "%swo", WT,
                       "%csliceD", "%cdim_s"))

        # 5: xa = rmsnorm(x + ao) * n2. The norm in front of the MLP is not
        # decoration: without it nothing renormalises the residual stream --
        # stage 0's rmsnorm feeds only the projection -- so the stream grows or
        # decays geometrically with depth, and since the host comparison is
        # relative, a stream in the thousands hides every error smaller than
        # itself.
        w(single_stage(l, 5, base + 5, f"""              %ssa{l} = scf.for %i = %c0_s to %cdim_s step %c1_s
                  iter_args(%s = %fzero_s) -> (f32) {{
                %xv = memref.load %sx[%m, %i] : {AT}
                %avv = memref.load %saov[%m, %i] : {AT}
                %xa = arith.addf %xv, %avv : f32
                memref.store %xa, %sxa[%m, %i] : {AT}
                %sqa = arith.mulf %xa, %xa : f32
                %s2 = arith.addf %s, %sqa : f32
                scf.yield %s2 : f32
              }}
              %meana{l} = arith.divf %ssa{l}, %fdim_s : f32
              %mea{l} = arith.addf %meana{l}, %eps_s : f32
              %rmsa{l} = math.sqrt %mea{l} : f32
              scf.for %i = %c0_s to %cdim_s step %c1_s {{
                %xv3 = memref.load %sxa[%m, %i] : {AT}
                %nv = arith.divf %xv3, %rmsa{l} : f32
                %nw = memref.load %sn2[%L{l}, %i] : {NT}
                %xn = arith.mulf %nv, %nw : f32
                memref.store %xn, %sxa[%m, %i] : {AT}
              }}"""))

        # 6: gu = xa @ Wgu, gate and up in one matmul as Fleet fuses them
        w(matmul_stage(l, 6, base + 6, "%sgu", GT, "%sxa", AT, "%swgu", WGT,
                       "%cslice2I", "%cdim_s"))

        # 7: SwiGLU. Elementwise, so a piece is a slice of the intermediate
        # width rather than a reduction.
        w(strided_stage(l, 7, base + 7, "%ctasks", "%ntasks_t",
                        f"""                %j0 = arith.muli %ix, %csliceI : index
                scf.for %jj = %c0_s to %csliceI step %c1_s {{
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
                }}"""))

        # 8: x = xa + act @ Wd, the residual folded into the matmul as Fleet
        # folds it (linear_with_residual_layer).
        w(matmul_stage(l, 8, base + 8, "%sx", AT, "%sact", IT, "%swd", WDT,
                       "%csliceD", "%cinter_s", residual="%sxa"))

    # final norm, lm head, and Fleet's two-stage argmax
    w(single_stage("x", stages + 1, base_x + 1, f"""              %fs = scf.for %i = %c0_s to %cdim_s step %c1_s
                  iter_args(%a = %fzero_s) -> (f32) {{
                %v = memref.load %sx[%m, %i] : {AT}
                %fq = arith.mulf %v, %v : f32
                %a2 = arith.addf %a, %fq : f32
                scf.yield %a2 : f32
              }}
              %fm = arith.divf %fs, %fdim_s : f32
              %fme = arith.addf %fm, %eps_s : f32
              %fr = math.sqrt %fme : f32
              scf.for %i = %c0_s to %cdim_s step %c1_s {{
                %v = memref.load %sx[%m, %i] : {AT}
                %nv = arith.divf %v, %fr : f32
                %nw = memref.load %snf[%i] : {NFT}
                %o = arith.mulf %nv, %nw : f32
                memref.store %o, %sr[%m, %i] : {AT}
              }}""", slot=base_x + 1, lc="%L0"))
    w(strided_stage("x", stages + 2, base_x + 2, "%ctasks", "%ntasks_t",
                    f"""                %v0 = arith.muli %ix, %csliceV : index
                scf.for %jj = %c0_s to %csliceV step %c1_s {{
                  %v = arith.addi %v0, %jj : index
                  %a = scf.for %i = %c0_s to %cdim_s step %c1_s
                      iter_args(%sacc = %fzero_s) -> (f32) {{
                    %xv = memref.load %sr[%m, %i] : {AT}
                    %wv = memref.load %swlm[%i, %v] {{nontemporal = true}} : {LMT}
                    %mp = arith.mulf %xv, %wv : f32
                    %s2 = arith.addf %sacc, %mp : f32
                    scf.yield %s2 : f32
                  }}
                  memref.store %a, %slg[%m, %v] : {LGT}
                }}""", slot=base_x + 2, lc="%L0"))
    # argmax_partial_layer: the best in this piece of the vocabulary
    w(strided_stage("x", stages + 3, base_x + 3, "%ctasks", "%ntasks_t",
                    f"""                %v0 = arith.muli %ix, %csliceV : index
                %bi:2 = scf.for %jj = %c0_s to %csliceV step %c1_s
                    iter_args(%bv = %negbig_s, %bidx = %zero_s) -> (f32, i32) {{
                  %v = arith.addi %v0, %jj : index
                  %lv = memref.load %slg[%m, %v] : {LGT}
                  %gt = arith.cmpf ogt, %lv, %bv : f32
                  %nv2 = arith.select %gt, %lv, %bv : f32
                  %vi = arith.index_cast %v : index to i32
                  %nx = arith.select %gt, %vi, %bidx : i32
                  scf.yield %nv2, %nx : f32, i32
                }}
                memref.store %bi#0, %spv[%m, %ix] : {PVT}
                memref.store %bi#1, %spi[%m, %ix] : {PIT}""",
                    slot=base_x + 3, lc="%L0"))
    # argmax_reduce_layer: the best across pieces, ties to the lower index
    w(single_stage("x", stages + 4, base_x + 4, f"""              %rd:2 = scf.for %k = %c0_s to %ctasks step %c1_s
                  iter_args(%bv = %negbig_s, %bidx = %zero_s) -> (f32, i32) {{
                %apv = memref.load %spv[%m, %k] : {PVT}
                %api = memref.load %spi[%m, %k] : {PIT}
                %gt = arith.cmpf ogt, %apv, %bv : f32
                %nv2 = arith.select %gt, %apv, %bv : f32
                %nx = arith.select %gt, %api, %bidx : i32
                scf.yield %nv2, %nx : f32, i32
              }}
              %spn = arith.addi %step, %c1_s : index
              memref.store %rd#1, %stok[%spn, %m] : {TKT}""",
                   slot=base_x + 4, lc="%L0"))

    w(f"""
          }}
        }}

        air.herd @herd tile (%htx, %hty) in (%ntx=%c1_s, %nty=%c1_s) {{
        }}
      }}
    }}
    return
  }}
}}
""")
    return "".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--inter", type=int, default=0,
                    help="MLP intermediate width; defaults to 2*dim")
    ap.add_argument("--heads", type=int, default=4, help="query heads")
    ap.add_argument("--kv-heads", type=int, default=2,
                    help="key/value heads; heads must be a multiple of this")
    ap.add_argument("--tasks", type=int, default=8,
                    help="tasks per matmul stage; dim, inter and the qkv width "
                         "must divide by this")
    ap.add_argument("--workers", type=int, default=32,
                    help="resident workgroups; must fit the device at once")
    ap.add_argument("--cache", type=int, default=32,
                    help="KV cache prefix the window attends")
    ap.add_argument("--tokens", type=int, default=1,
                    help="tokens in flight (M). 1 is a decode step; >1 is a "
                         "prefill or speculative window, and is what makes the "
                         "M-major traversal do anything")
    ap.add_argument("--vocab", type=int, default=256,
                    help="vocabulary size; must divide by tasks")
    ap.add_argument("--steps", type=int, default=1,
                    help="decode steps in one launch. Each step appends its "
                         "own window to the KV cache and attends everything up "
                         "to it, so the attention length is a runtime value")
    ap.add_argument("--repeat", type=int, default=1,
                    help="how many times to launch the chain, for timing")
    a = ap.parse_args()
    sys.stdout.write(emit(a.layers, a.dim, a.tasks, a.workers, a.repeat,
                          a.cache, a.tokens, a.inter, a.heads, a.kv_heads,
                          a.steps, a.vocab))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

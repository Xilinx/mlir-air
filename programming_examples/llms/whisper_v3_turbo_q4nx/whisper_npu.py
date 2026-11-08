# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Whisper large-v3-turbo with the bundle's affine Q4 weights: every GEMM on
the shared fused prefill device (whisper_spec.py), the encoder's attention on
its own FlashAttention ELF (whisper_attn.py).

The host computes the log-mel input. The conv front end (conv1 -> gelu ->
conv2 stride 2 -> gelu, + positions) runs both convs as GEMMs on the device,
a row of A being 3 frames of the conv's input (front). Each of the
32 encoder layers is then, over the 1500 frames (12 chunks of 128 rows, the
last 36 rows padding):

  h = ln1(x);  q | k | v = h Wqkv                     (qkv GEMM, chunk pairs)
  q += bq;  a = attention(q, k, v) over keys [0, 1500)  (one ELF call)
  x += a Wo + (bo + Wo bv)                            (o GEMM)
  x += gelu(ln2(x) W1 + b1) W2 + b2                    (fc1, fc2 GEMMs)

with the residual x in float32 on the host and the LayerNorms, biases and the
erf GELU (hostops) feeding each GEMM's A. V's bias is folded into o's: the
attention's weights in a row sum to 1. The qkv GEMMs write into the buffer
the attention reads. The final LayerNorm's output feeds one GEMM for all four
decoder layers' cross-attention K and V, which become each layer's cross KV
records.

The decoder runs a token at a time (step), the prompt's tokens too, each a
one-row chunk on the same GEMMs. Its self-attention is the device's noncausal
attention op reading keys [0, pos + 1), which for a single query is the causal
mask; its cross-attention reads the 1500 cross keys. The final LayerNorm's
output feeds the bundle's Q4 lm_head (a copy of the tied embedding) in pieces.
"""

from collections import deque
from pathlib import Path

import numpy as np
import pyxrt as xrt
from ml_dtypes import bfloat16

from air_examples.llms.shared.fused_prefill import device as D
from air_examples.llms.shared.fused_prefill import hostops as H
from air_examples.llms.shared.fused_prefill import packing as P
from air_examples.llms.shared.fused_prefill.engine import FROM, TO, Engine, view

from . import whisper_spec as S
from .whisper_attn import EncoderAttention
from .whisper_weights import D as DM, DEC_LAYERS, ENC_LAYERS, FF, H as HEADS, DH
from .whisper_weights import MAX_TOKENS, N_FRAMES

EPS = 1e-5
N_CTX = S.FRAMES
ROWS = S.max_len
XKV = 2 * DEC_LAYERS * DM
LKP = S.CFG.attn["a0"].lkp


def _f32(a):
    return np.ascontiguousarray(a, np.float32)


class WhisperNpu(Engine):
    def __init__(self, build_dir, weights):
        super().__init__(build_dir, S.CFG, ROWS)

        def q4(site, name, rows, k):
            self.gemm_site(site, k, rows, self.pack_q4(*weights.q4(name, rows, k)), 1)

        def q4cat(site, names, rows, k):
            parts = [weights.q4(n, rows, k) for n in names]
            packed = self.pack_q4(*map(np.concatenate, zip(*parts)))
            self.gemm_site(site, k, rows * len(names), packed, 1)

        def bf(n):
            return _f32(weights.bf16(n))

        def ln(n):
            return bf(n + ".weight"), bf(n + ".bias")

        zero = np.zeros(DM, np.float32)
        self.enc = []
        for L in range(ENC_LAYERS):
            p = f"model.encoder.layers.{L}."
            a = p + "self_attn."
            q4cat(f"{L}.qkv", [a + f"{n}_proj.weight" for n in "qkv"], DM, DM)
            q4(f"{L}.o", a + "out_proj.weight", DM, DM)
            q4(f"{L}.fc1", p + "fc1.weight", FF, DM)
            q4(f"{L}.fc2", p + "fc2.weight", DM, FF)
            # The attention's rows of weights sum to 1, so V's bias passes
            # through it unchanged: it goes into o's bias as Wo bv.
            wo = weights.dequant(a + "out_proj.weight", DM, DM)
            self.enc.append(
                dict(
                    ln1=ln(p + "self_attn_layer_norm"),
                    ln2=ln(p + "final_layer_norm"),
                    q=bf(a + "q_proj.bias"),
                    o=bf(a + "out_proj.bias") + wo @ bf(a + "v_proj.bias"),
                    fc1=bf(p + "fc1.bias"),
                    fc2=bf(p + "fc2.bias"),
                )
            )
        self.ln_post = ln("model.encoder.layer_norm")
        # The conv front end as GEMMs (front): column j * C_in + c of weight
        # row o is tap j of input channel c.
        e = "model.encoder."
        for name, (k, n) in S.CONV_SHAPES.items():
            w = _f32(weights.bf16(e + name + ".weight"))  # [out, in, 3]
            self.gemm_site(
                name, k, n, self.pack_bf16(w.transpose(0, 2, 1).reshape(n, k)), 0
            )
        self.conv_bias = {n: bf(e + n + ".bias") for n in S.CONV_SHAPES}
        self.positions = weights.bf16(e + "embed_positions.weight")[:N_CTX]
        # each conv's input, time-major, frame f at row f + 1, zero rows around
        self.conv_in = (
            np.zeros((N_FRAMES + 2, S.N_MELS), np.float32),
            np.zeros((N_FRAMES + 2, DM), bfloat16),
        )
        # cross-attention K | V of each decoder layer, K without bias
        names, xb = [], []
        for L in range(DEC_LAYERS):
            a = f"model.decoder.layers.{L}.encoder_attn."
            names += [a + "k_proj.weight", a + "v_proj.weight"]
            xb += [zero, bf(a + "v_proj.bias")]
        q4cat("xkv", names, DM, DM)
        self.xkv_bias = np.concatenate(xb)

        self.dec = []
        for L in range(DEC_LAYERS):
            p = f"model.decoder.layers.{L}."
            a, x = p + "self_attn.", p + "encoder_attn."
            q4cat(f"d{L}.qkv", [a + f"{n}_proj.weight" for n in "qkv"], DM, DM)
            q4(f"d{L}.o", a + "out_proj.weight", DM, DM)
            q4(f"d{L}.xq", x + "q_proj.weight", DM, DM)
            q4(f"d{L}.xo", x + "out_proj.weight", DM, DM)
            q4(f"d{L}.fc1", p + "fc1.weight", FF, DM)
            q4(f"d{L}.fc2", p + "fc2.weight", DM, FF)
            self.dec.append(
                dict(
                    ln1=ln(p + "self_attn_layer_norm"),
                    ln2=ln(p + "encoder_attn_layer_norm"),
                    ln3=ln(p + "final_layer_norm"),
                    qkv=np.concatenate(
                        [bf(a + "q_proj.bias"), zero, bf(a + "v_proj.bias")]
                    ),
                    o=bf(a + "out_proj.bias"),
                    xq=bf(x + "q_proj.bias"),
                    xo=bf(x + "out_proj.bias"),
                    fc1=bf(p + "fc1.bias"),
                    fc2=bf(p + "fc2.bias"),
                )
            )
        self.ln_dec = ln("model.decoder.layer_norm")
        self.embed = weights.bf16("model.decoder.embed_tokens.weight")
        self.pos = weights.bf16("model.decoder.embed_positions.weight")
        q, s, m = weights.q4("lm_head.weight", S.VOCAB_PAD, DM)
        for i, (n0, n1) in enumerate(S.HEAD_PIECES):
            self.gemm_site(
                f"head.{i}", DM, n1 - n0, self.pack_q4(q[n0:n1], s[n0:n1], m[n0:n1]), 1
            )

        self.attn_setup()
        # The encoder's q | k | v, bf16 [ROWS, 3 DM]: each chunk pair's qkv
        # GEMM writes its rows here and the attention reads them in place.
        self.qkv_bo = self.fp.bo(ROWS * 3 * DM * 2, 2)
        self.qkv = view(self.qkv_bo, bfloat16, (ROWS, 3 * DM))
        npd = 3 * DM
        assert P.n_pad(npd) == npd
        pair = 2 * D.M * npd * 2
        for g in range(ROWS // (2 * D.M)):
            sub = xrt.bo(self.qkv_bo, pair, g * pair)
            self.chunk_bufs[("c", "qkv", npd, g, 1)] = (
                sub,
                view(sub, bfloat16, (2, D.M, npd)),
            )
        self.enc_attn = EncoderAttention(build_dir, self.dev)
        self.attn_out = view(self.enc_attn.out, bfloat16, (ROWS, DM))
        self.xkv_f = np.zeros((ROWS, XKV), np.float32)
        self.xkvb = [self.kv_bo("a") for _ in range(DEC_LAYERS)]
        self.skvb = [self.kv_bo("a") for _ in range(DEC_LAYERS)]
        # host copies of the decoder's self K | V, whose last KV block the
        # device's records are rewritten from at every token
        self.skv = np.zeros((DEC_LAYERS, MAX_TOKENS, 2, HEADS, DH), bfloat16)
        self.warm("0.fc1")

    # ---- shared pieces ----

    def _ln(self, rows, ln, k, c):
        a = self.a_chunk(k, c)
        H.ln_tile(rows, *ln, EPS, a[1], D.TM, D.TK)
        a[0].sync(TO)

    def _q(self, src, t, col0, bias, c):
        """Query heads at col0 of src (+ bias) into chunk c's attention."""
        ch = self.attn_chunk("a", c)
        h0 = 0
        for p in ch["parts"]:
            qm = self.q_stage(p)
            H.head_post(
                src, t, col0 + h0 * DH, p["hg"], DH, bias, None, EPS, None,
                1.0, qm, D.M * DH, DH,
            )  # fmt: skip
            if t < D.M:
                qm[:, t:] = 0
            self.q_write(p, qm)
            h0 += p["hg"]

    def _o(self, pend, t, c):
        """The attention's output into chunk c's A of depth DM."""
        a = self.a_chunk(DM, c)
        h0 = 0
        for p in self.attn_wait(pend):
            self.o_read(p, t, h0, DM, a[1])
            h0 += p["hg"]
        a[0].sync(TO)

    def _add(self, rows, pend, t, bias):
        """rows += the GEMM's output + bias."""
        H.add_rms_tile(
            rows, self.gemm_wait(pend, t), t, None, EPS, None, None, D.TM, D.TK
        )
        rows += bias

    def _gelu(self, pend, t, bias, c):
        a = self.a_chunk(FF, c)
        H.bias_gelu_erf_tile(self.gemm_wait(pend, t), bias, t, FF, a[1], D.TM, D.TK)
        a[0].sync(TO)

    # ---- conv front end ----

    def _conv_gemm(self, site, x, rows, stride):
        """site's GEMM over rows of A, row r the 3 frames x[stride r : stride r
        + 3] side by side: overlapping rows of x, no copy. bf16 [rows, n]."""
        k, n = S.CONV_SHAPES[site]
        a_rows = np.lib.stride_tricks.as_strided(
            x, (rows, k), (stride * x.strides[0], x.strides[1])
        )
        chunks = self.chunk_rows(rows)
        pend = []
        for c, (r0, t) in enumerate(chunks):
            a = self.a_chunk(k, c)
            H.tile_a(a_rows[r0 : r0 + t], a[1], D.TM, D.TK)
            a[0].sync(TO)
            pend.append(self.gemm_start(site, c))
        out = np.empty((rows, n), bfloat16)
        for c, (r0, t) in enumerate(chunks):
            out[r0 : r0 + t] = self.gemm_wait(pend[c], t)[:t, :n]
        return out

    def front(self, mel):
        """log-mel float32 [128, 3000] -> the first encoder layer's input
        float32 [1500, 1280]: conv1 -> gelu -> conv2 (stride 2) -> gelu, +
        positions, each conv a GEMM on the NPU and each step rounded to
        bf16."""
        x1, x2 = self.conv_in
        x1[1 : N_FRAMES + 1] = mel.T
        y = self._conv_gemm("conv1", x1, N_FRAMES, 1)
        H.bias_gelu_erf_rows(y, self.conv_bias["conv1"], N_FRAMES, DM, x2[1:])
        y = self._conv_gemm("conv2", x2, N_CTX, 2)
        g = np.empty((N_CTX, DM), bfloat16)
        H.bias_gelu_erf_rows(y, self.conv_bias["conv2"], N_CTX, DM, g)
        x = _f32(g) + _f32(self.positions)
        return _f32(x.astype(bfloat16))

    # ---- encoder ----

    def _kv_records(self, kvb, r0, k, v):
        """K and V bf16 [t, HEADS * DH] of keys [r0, r0 + t) (r0 a whole
        number of blocks) as kvb's records."""
        t = k.shape[0]
        for i, p in enumerate(kvb["parts"]):
            g = self.cfg.attn[p["a"]]
            b0, nb = r0 // g.lkp, -(-t // g.lkp)
            rec, nblk = self.kv_rec(g) * 2, p["m"].shape[1]
            for h in range(g.kv_heads):
                self.kv_write(k, v, (i * g.kv_heads + h) * DH, t, g, p["m"][h, b0])
                p["bo"].sync(TO, nb * rec, (h * nblk + b0) * rec)

    def _pair_start(self, site, u):
        """site's GEMM on the chunk pair u (chunks 2u, 2u + 1), one dispatch."""
        return [self.gemm_start(site, c) for c in (2 * u, 2 * u + 1)]

    def _enc_layer(self, L, x, last):
        """Layer L from its q | k | v in self.qkv, then the next layer's qkv
        GEMM (or, after the last, the cross K/V GEMM) on each chunk pair as
        soon as its rows are done.

        The q bias is added on the host and the attention runs once over all
        rows. The device runs the GEMMs in the order they are queued, so the
        host waits on the oldest, does the work it enables and queues what
        follows: one pair's host work overlaps the other pairs' GEMMs."""
        ly, chunks = self.enc[L], self.chunks
        qkv = self.qkv
        self.qkv_bo.sync(FROM)
        H.head_post(
            qkv, ROWS, 0, HEADS, DH, ly["q"], None, EPS, None, 1.0, qkv, DH, 3 * DM
        )
        self.qkv_bo.sync(TO)
        self.enc_attn.start(self.qkv_bo)
        self.enc_attn.wait()
        fifo = deque()
        for u in range(len(chunks) // 2):
            for c in (2 * u, 2 * u + 1):
                r0, t = chunks[c]
                a = self.a_chunk(DM, c)
                H.tile_a(H.bf16_to_f32(self.attn_out[r0:], t, DM), a[1], D.TM, D.TK)
            a[0].sync(TO)
            fifo.append(("o", u, self._pair_start(f"{L}.o", u)))
        while fifo:
            op, u, pend = fifo.popleft()
            for i, p in enumerate(pend):
                c = 2 * u + i
                r0, t = chunks[c]
                rows = x[r0 : r0 + t]
                if op == "o":
                    self._add(rows, p, t, ly["o"])
                    self._ln(rows, ly["ln2"], DM, c)
                elif op == "fc1":
                    self._gelu(p, t, ly["fc1"], c)
                elif op == "fc2":
                    self._add(rows, p, t, ly["fc2"])
                    ln = self.ln_post if last else self.enc[L + 1]["ln1"]
                    self._ln(rows, ln, DM, c)
                elif op == "qkv":
                    self.gemm_wait(p, t)
                else:
                    out = self.gemm_wait(p, t)
                    self.xkv_f[r0 : r0 + t] = H.bf16_to_f32(out, t, XKV)
            follow = {
                "o": ("fc1", f"{L}.fc1"),
                "fc1": ("fc2", f"{L}.fc2"),
                "fc2": ("xkv", "xkv") if last else ("qkv", f"{L + 1}.qkv"),
            }.get(op)
            if follow:
                fifo.append((follow[0], u, self._pair_start(follow[1], u)))

    def encode(self, x0):
        """x0 float32 [1500, 1280] (front) through the encoder, into the
        decoder's cross KV records."""
        self.chunks = self.chunk_rows(ROWS)
        x = np.zeros((ROWS, DM), np.float32)
        x[:N_CTX] = x0
        pend = []
        for c, (r0, t) in enumerate(self.chunks):
            self._ln(x[r0 : r0 + t], self.enc[0]["ln1"], DM, c)
            pend.append(self.gemm_start("0.qkv", c))  # pairs start with their second
        for c, (r0, t) in enumerate(self.chunks):
            self.gemm_wait(pend[c], t)
        for L in range(ENC_LAYERS):
            self._enc_layer(L, x, L == ENC_LAYERS - 1)
        xkv = (self.xkv_f[:N_CTX] + self.xkv_bias).astype(bfloat16)
        self.xkv = xkv.reshape(N_CTX, DEC_LAYERS, 2, DM)
        for L in range(DEC_LAYERS):
            k, v = (np.ascontiguousarray(self.xkv[:, L, i]) for i in (0, 1))
            self._kv_records(self.xkvb[L], 0, k, v)

    # ---- decoder ----

    def step(self, token, pos):
        """The logits float32 [51866] after token at position pos, the
        tokens before it having been stepped at positions [0, pos)."""
        if pos >= MAX_TOKENS:
            raise ValueError(f"position {pos} past the decoder's {MAX_TOKENS}")
        self.n_chunks = 1
        x = _f32(self.embed[token]) + _f32(self.pos[pos])
        x = x[None].copy()
        b0 = pos // LKP * LKP
        for L, ly in enumerate(self.dec):
            self._ln(x, ly["ln1"], DM, 0)
            qkv = self.gemm_wait(self.gemm_start(f"d{L}.qkv", 0), 1)
            kv = self.skv[L, pos].reshape(1, 2 * DM)
            H.head_post(
                qkv, 1, DM, 2 * HEADS, DH, ly["qkv"], None, EPS, None, 1.0,
                kv, DH, 2 * DM,
            )  # fmt: skip
            blk = self.skv[L, b0 : pos + 1]
            self.kv_append(self.skvb[L], b0, blk[:, 0], blk[:, 1])
            self._q(qkv, 1, 0, ly["qkv"], 0)
            self._o(self.attn_start(self.skvb[L], 0, 0, 1, nk=pos + 1), 1, 0)
            self._add(x, self.gemm_start(f"d{L}.o", 0), 1, ly["o"])

            self._ln(x, ly["ln2"], DM, 0)
            xq = self.gemm_wait(self.gemm_start(f"d{L}.xq", 0), 1)
            self._q(xq, 1, 0, ly["xq"], 0)
            self._o(self.attn_start(self.xkvb[L], 0, 0, 1, nk=N_CTX), 1, 0)
            self._add(x, self.gemm_start(f"d{L}.xo", 0), 1, ly["xo"])

            self._ln(x, ly["ln3"], DM, 0)
            self._gelu(self.gemm_start(f"d{L}.fc1", 0), 1, ly["fc1"], 0)
            self._add(x, self.gemm_start(f"d{L}.fc2", 0), 1, ly["fc2"])
        self._ln(x, self.ln_dec, DM, 0)
        pend = [self.gemm_start(f"head.{i}", 0) for i in range(len(S.HEAD_PIECES))]
        z = [
            H.bf16_to_f32(self.gemm_wait(p, 1), 1, n1 - n0)
            for p, (n0, n1) in zip(pend, S.HEAD_PIECES)
        ]
        return np.concatenate(z, 1)[0, : S.VOCAB]


# ---- decoding ----

# the original checkpoint's generation_config.json: suppressed tokens, task
# and language ids
GEN_REPO = "openai/whisper-large-v3-turbo"
GEN_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
MAX_NEW = 128


class Decoding:
    """Greedy decoding as transformers' WhisperForConditionalGeneration.generate
    does it without timestamps: the prompt <|startoftranscript|> <|lang|>
    <|task|> <|notimestamps|>, begin_suppress_tokens masked at the first
    generated token and suppress_tokens at every one, until EOS."""

    def __init__(self, language="en", task="transcribe"):
        import json

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            GEN_REPO, "generation_config.json", revision=GEN_REVISION
        )
        g = json.loads(Path(path).read_text())
        lang = g["lang_to_id"].get(f"<|{language}|>")
        if lang is None:
            raise ValueError(f"unknown language {language!r}")
        self.prompt = [
            g["decoder_start_token_id"],
            lang,
            g["task_to_id"][task],
            g["no_timestamps_token_id"],
        ]
        self.eos = g["eos_token_id"]
        self.suppress = np.asarray(g["suppress_tokens"])
        self.begin = np.asarray(g["begin_suppress_tokens"])

    def mask(self, logits, first):
        """The logits the next token is chosen from (a copy)."""
        z = np.array(logits, np.float32)
        z[self.suppress] = -np.inf
        if first:
            z[self.begin] = -np.inf
        return z


def log_mel(audio):
    """float32 16 kHz audio of at most 30 s -> log-mel float32 [128, 3000]."""
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(feature_size=128)
    return fe(audio, sampling_rate=16000, return_tensors="np").input_features[0]


def transcribe(npu, audio, dec, max_new=MAX_NEW, on_step=None, times=None):
    """Greedy transcription of audio (float32, 16 kHz, at most 30 s); the
    generated tokens, EOS included if reached. on_step(i, logits) sees each
    step's masked logits before its token is chosen. Raises if a step's
    logits are not all finite. times, a dict, receives the seconds of the
    front end (log-mel, convs) and the encoder, and those from the start to
    the first token and to the last."""
    import time

    t0 = time.perf_counter()
    x0 = npu.front(log_mel(audio))
    t1 = time.perf_counter()
    npu.encode(x0)
    t2 = time.perf_counter()
    for pos, t in enumerate(dec.prompt[:-1]):
        npu.step(t, pos)
    tok, pos, out = dec.prompt[-1], len(dec.prompt) - 1, []
    for i in range(max_new):
        z = npu.step(tok, pos)
        if not np.isfinite(z).all():
            raise RuntimeError(f"non-finite logits at step {i}")
        z = dec.mask(z, i == 0)
        if on_step:
            on_step(i, z)
        tok, pos = int(np.argmax(z)), pos + 1
        out.append(tok)
        if i == 0:
            t3 = time.perf_counter()
        if tok == dec.eos:
            break
    if times is not None:
        t4 = time.perf_counter()
        times.update(front=t1 - t0, encoder=t2 - t1, first=t3 - t0, total=t4 - t0)
    return out


def tokenizer(weights):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(weights.path.parent / "tokenizer.json"))


def main():
    import argparse
    import time

    import soundfile as sf

    from .whisper_weights import Q4nx

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("audio", nargs="+", help="audio files, at most 30 s each")
    ap.add_argument("--build", required=True, help="the device's build directory")
    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="transcribe", choices=("transcribe", "translate"))
    a = ap.parse_args()
    w = Q4nx()
    npu = WhisperNpu(a.build, w)
    tok, dec = tokenizer(w), Decoding(a.language, a.task)
    for f in a.audio:
        audio, sr = sf.read(f, dtype="float32", always_2d=True)
        audio = audio.mean(1)
        if sr != 16000:
            raise ValueError(f"{f}: {sr} Hz; resample to 16 kHz first")
        t0 = time.perf_counter()
        out = transcribe(npu, audio, dec)
        dt = time.perf_counter() - t0
        print(f"{f}: {tok.decode([t for t in out if t < dec.eos])}")
        print(f"  {len(audio) / sr:.1f} s audio, {len(out)} tokens, {dt * 1e3:.0f} ms")

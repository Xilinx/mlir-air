# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Fused superkernel decode for Llama-3.2-1B on NPU2 (MLIR-AIR).
#
# ONE dispatch = 16 decoder layers + LM head, reading and appending a shared per-layer
# KV cache. A single xclbin built at ATTN_MAXL=2048 serves every context length L in
# [1, 2048]: the attention loop skips fully-masked blocks and the per-token instruction
# stream is patched on the host (decode_insts_gen.py) -- recompile-free per-token decode.
#
# Quick start:
#   make compile-decode   Build the decode kernels + two templates (decode_L2048 + L2047)
#
# This example provides the decode path consumed by the llama32_1b_q4nx LLM end-to-end
# (programming_examples/llms/llama32_1b_q4nx). It needs Q4NX weights for the GEMM shapes.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

DECODE_DRIVER  := $(srcdir)/fused_decode.py
AIEOPT_DIR ?= $(if $(MLIR_AIE_INSTALL_DIR),$(subst \,/,$(MLIR_AIE_INSTALL_DIR)),$(shell realpath $(dir $(shell which aie-opt))/..))
# Per-kernel -O is LOAD-BEARING (do not use one -O for all):
#   proj_qmm/rms_residual/glu/rope MUST be -O2; attn_qk/attn_kv MUST be -O1.
#   rope.cc at -O1 MISCOMPILES (Peano loop-metadata) -> the decode HANGS on device
#   (6143 ms ERT_CMD_STATE_TIMEOUT). attn_qk/attn_kv at -O2 hit a separate do-while
#   deadlock, so they must stay -O1.
NONATTN_KERNELS := proj_qmm rms_residual glu rope   # object .o, -O2
ATTN_LL_KERNELS := attn_qk attn_kv                  # inline .ll only, -O1 (merge-linked)
# Extra per-group kernel defines, for A/B diagnostics ONLY (empty in every normal
# build, so the shipped path is byte-identical). They exist because the kernels'
# cost is not separable from the outside: end-to-end latency cannot say how much
# of a phase is compute vs waiting on its feed, and hardware tracing is not
# available. Building a variant that keeps the dataflow but drops the math
# answers it directly. Example -- attn_kv.cc/attn_qk.cc honour SKIP_ATTN_FV and
# SKIP_CALC_L, which keep every lock handshake and DMA but remove the attention
# math, and measured that attn_fv compute is 0.746 of the 1.137 ms/1k context
# slope (the rest is feed):
#     make compile-decode ATTN_EXTRA="-DSKIP_ATTN_FV -DSKIP_CALC_L"
# Such builds produce WRONG results -- pair them with bench-decode (latency
# only), never with make verify.
ATTN_EXTRA     ?=
NONATTN_EXTRA  ?=
PEANO_KBASE    := -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -DNDEBUG -DMODEL_TYPE=LLAMA_3_2_1B -D__AIE_API_AIE_ADF_HPP__ \
  -I $(AIEOPT_DIR)/include -I $(srcdir)/kernels -I $(srcdir)/models \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
# DECODE_GOLDEN=1 enables the post-attention-RMS decode path (a boolean flag; no
# weight files are read at build time -- the compile is weight-free).
# W_DUAL_CHAN=1 feeds each proj column's weights on BOTH of its shim MM2S channels
# (@inW0c{cx} takes the low half of the column's cascade rows, @inW1c{cx} the high
# half) instead of ch0 only. It also changes the DDR cascade order, so any
# consumer must pack its weights with the SAME flag -- the
# llms/*_q4nx drivers read it from the environment and key their requant cache on
# it. Inherited from the environment so a consumer can set it once for both the
# build and the run.
W_DUAL_CHAN    ?= 1
# VOCAB_CHUNK_I2 / UNI_LM: overridable so the lm-head wave count can be varied at
# CONSTANT work (UNI_LM*VOCAB_CHUNK_I2 must stay = VOCAB_FULL_ROWBLKS/32). Defaults
# reproduce the shipped config exactly. fused_decode.py asserts the two agree.
VOCAB_CHUNK_I2 ?= 18
# PROJ_RC_CACHE=1 (default) computes the proj +min-term reduction of the
# activation ONCE per col-block per projection and reuses it across that
# projection's row-blocks, instead of recomputing it on every (row, col) block.
# It is the reference proj_main's `if (i == 0)` guard -- see
# kernels/proj_qmm.cc::proj_qmm_acc256_c. Numerically EXACT (same values, fewer
# times): measured bit-identical 64-token greedy output at ctx 263 and 1933.
# Worth -0.58 ms/token, all of it on the context-INDEPENDENT part (intercept
# 15.87 -> 15.30 ms; slope unchanged at 1.14 ms/1k). Only engages on the
# PAIR_ROWS=2 proj path (llama); the PAIR_ROWS=1 gemma path is untouched.
# PROJ_RC_CACHE=0 restores the recompute-every-block kernel for A/B.
PROJ_RC_CACHE  ?= 1
DECODE_ENV     := VOCAB_CHUNK_I2=$(VOCAB_CHUNK_I2) LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=1 \
                  W_DUAL_CHAN=$(W_DUAL_CHAN) PROJ_RC_CACHE=$(PROJ_RC_CACHE) \
                  $(if $(UNI_LM),UNI_LM=$(UNI_LM),)

.PHONY: help compile-decode _compile_decode_build compile-decode-dynseq preflight-peano preflight-llvm-link clean

help:
	@echo "============================================================"
	@echo " Fused superkernel decode (Llama-3.2-1B) on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "  make compile-decode   Build decode kernels + templates (decode_L2048 + L2047, ~15 min)"
	@echo "  make bench-decode     C++ per-token latency in FLM-comparable scope; LATENCY ONLY,"
	@echo "                        never a correctness gate.  L=1933 ITERS=64 WARMUP=8"
	@echo "  make clean            Remove kernels + templates"
	@echo ""
	@echo "Consumed by the llama32_1b_q4nx LLM e2e (make chat / make gen there)."
	@echo "Weight-free build; needs an llvm-link on PATH from LLVM <23"
	@echo "-- see README.md Reproducibility."

## Build the fused decode: Peano kernels + two templates (decode_L2048 + decode_L2047
## slope ref). ~15 min. Weight-free (runtime BOs), like the prefill compile.
##
## Idempotent: if both templates are already present AND were built with the
## current W_DUAL_CHAN setting, skip (the ~15 min build only runs once). `make clean`
## forces a rebuild. Consumers that call this repeatedly (e.g. the llama32_1b_q4nx
## verify + profile lits) thus pay the cost at most once. The .decode_flags stamp
## exists because W_DUAL_CHAN changes the DDR weight layout: a warm template built
## with the other setting would silently be fed the wrong blocks.
##
## REPRODUCIBILITY (toolchain): the inline-attn merge path shells out to an EXTERNAL
## `llvm-link` (resolved from PATH), which must be PRE-LIFETIME-CHANGE, i.e. LLVM < 23.
## A >=23 llvm-link upgrades the llvm.lifetime intrinsic to the no-size form and Peano
## opt then rejects the merged IR ("Broken module found"). The preflight aborts on it.
## This is the same rule the lit harness gates on (lit.cfg.py, feature llvm_link_pre23);
## the two MUST agree -- see the preflight comment below.
LBUILD ?= 2048
LREF   := $(shell echo $$(($(LBUILD) - 1)))
DECODE_FLAGS_STAMP := $(srcdir)/.decode_flags
DECODE_FLAGS       := W_DUAL_CHAN=$(W_DUAL_CHAN)

compile-decode:
	@if [ -f $(srcdir)/decode_L2048.xclbin ] && [ -f $(srcdir)/decode_L2048.insts.bin ] \
	    && [ -f $(srcdir)/decode_L2047.xclbin ] && [ -f $(srcdir)/decode_L2047.insts.bin ] \
	    && [ "`cat $(DECODE_FLAGS_STAMP) 2>/dev/null`" = "$(DECODE_FLAGS)" ]; then \
	  echo "Decode templates already built (decode_L2048 + decode_L2047, $(DECODE_FLAGS)); skipping. Run 'make clean' to force a rebuild."; \
	else \
	  $(MAKE) -f $(srcdir)/Makefile _compile_decode_build \
	    && echo "$(DECODE_FLAGS)" > $(DECODE_FLAGS_STAMP); \
	fi

## Shared preflight for every inline-attn decode build. Any model driving this
## engine (llms/*_q4nx) must run it rather than copy it, so the rule lives in
## exactly one place.
##
## The rule is LLVM < 23, and it must stay identical to the `llvm_link_pre23`
## feature in programming_examples/lit.cfg.py. Every decode e2e lit test
## (llama32_1b_q4nx, llama32_3b_q4nx, gemma3_4b_q4nx) is REQUIRES-gated on that
## feature, and lit will shim ANY <23 llvm-link it finds on PATH to the front --
## with no regard for Peano's major. If this preflight is stricter than lit, lit
## marks such a test supported, runs it, and the test hard-fails right here,
## which is the exact outcome the feature exists to avoid.
# The repo pins llvm-aie (utils/build-mlir-air-using-wheels.sh). That script only
# installs at setup time, so a sandbox created before a pin bump keeps building
# with a stale Peano and nothing says so. #1808 retired the NOINLINE attn
# workaround on the strength of a Peano fix; built against the older Peano the
# decode emits a coherent first token and then garbage -- and the Paris gate
# still passes, because that token comes from prefill.
preflight-peano:
	@PIN=$$(grep -oE 'llvm-aie==[0-9][^"]*' $(srcdir)/../../utils/build-mlir-air-using-wheels.sh | head -1 | cut -d= -f3); \
	  if [ -z "$$PIN" ]; then echo "    preflight: no llvm-aie pin found, skipping"; exit 0; fi; \
	  if [ -z "$(PEANO_INSTALL_DIR)" ] || [ ! -x "$(PEANO_INSTALL_DIR)/bin/clang" ]; then \
	    echo "ERROR: PEANO_INSTALL_DIR does not point at a Peano install"; \
	    echo "       (no $(PEANO_INSTALL_DIR)/bin/clang); this repo pins llvm-aie==$$PIN"; \
	    exit 1; fi; \
	  HAVE=$$(cat $(PEANO_INSTALL_DIR)/../llvm_aie-*.dist-info/METADATA 2>/dev/null \
	          | grep -m1 '^Version:' | awk '{print $$2}'); \
	  if [ -z "$$HAVE" ]; then \
	    echo "ERROR: cannot read an llvm-aie version next to $(PEANO_INSTALL_DIR)"; \
	    echo "       (no llvm_aie-*.dist-info/METADATA), so it cannot be checked"; \
	    echo "       against the pinned llvm-aie==$$PIN. A stale Peano miscompiles"; \
	    echo "       the inlined attn: correct first token, garbage after."; \
	    exit 1; fi; \
	  if [ "$$HAVE" != "$$PIN" ]; then \
	    echo "ERROR: Peano is $$HAVE but this repo pins $$PIN"; \
	    echo "       A stale Peano miscompiles the inlined attn: correct first token, garbage after."; \
	    echo "       Fix: pip install --upgrade --force-reinstall 'llvm-aie==$$PIN' \\"; \
	    echo "              -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"; \
	    exit 1; fi; \
	  echo "    Peano = $$HAVE (matches pin)"

preflight-llvm-link: preflight-peano
	@echo ">>> preflight: llvm-link must be LLVM <23 (pre-lifetime-change) for the inline-attn merge"
	@# A >=23 llvm-link rewrites llvm.lifetime.* to the no-size form, which Peano
	@# opt then rejects outright ("Broken module found") -- loud, at build time.
	@# Peano's own major is reported below for context only; it is NOT the gate.
	@LLVM_LINK=$$(command -v llvm-link || true); \
	  PEANO_CLANG="$(PEANO_INSTALL_DIR)/bin/clang"; \
	  if [ -z "$(PEANO_INSTALL_DIR)" ] || [ ! -x "$$PEANO_CLANG" ]; then \
	    echo "ERROR: PEANO_INSTALL_DIR does not point at a Peano install (no $$PEANO_CLANG)"; exit 1; fi; \
	  PEANO_VER=$$($$PEANO_CLANG --version | grep -oE 'clang version [0-9]+' | grep -oE '[0-9]+' | head -1); \
	  if [ -z "$$LLVM_LINK" ]; then echo "ERROR: no llvm-link on PATH (need LLVM <23)"; exit 1; fi; \
	  VER=$$($$LLVM_LINK --version | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' | head -1); \
	  if [ -z "$$VER" ]; then \
	    echo "ERROR: could not parse an LLVM major from '$$LLVM_LINK --version' (need LLVM <23)"; exit 1; fi; \
	  echo "    llvm-link = $$LLVM_LINK (LLVM $$VER); Peano = LLVM $$PEANO_VER"; \
	  if [ "$$VER" -ge 23 ]; then \
	    echo "ERROR: llvm-link is LLVM $$VER; the inline-attn merge needs LLVM <23"; \
	    echo "       (see README Reproducibility for how to fetch one)"; exit 1; fi

## Dynamic-runtime-sequence template: ONE build, whatever the context length.
## The context length becomes a runtime scalar, so a single xclbin plus the
## compiler-emitted TXN builder (decode_dyn_L<M>.txn.h) serves every L -- the
## readback moving this token's context rather than the padded ATTN_MAXL. Run
## with DECODE_DYNSEQ=1. No template pair: there is no slope to calibrate.
compile-decode-dynseq: preflight-llvm-link
	@echo ">>> non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) $(NONATTN_EXTRA) -O2 -c $(srcdir)/kernels/$$k.cc -o $(srcdir)/$$k.o || exit 1; done
	@echo ">>> attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) $(ATTN_EXTRA) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(srcdir)/kernels/$$k.cc -o $(srcdir)/$$k.ll || exit 1; done
	@echo ">>> dynseq template decode_dyn_L$(LBUILD)"
	@cd $(srcdir) && $(DECODE_ENV) DECODE_DYNSEQ=1 DECODE_GOLDEN_L=$(LBUILD) $(PYTHON) $(DECODE_DRIVER) >/tmp/decode_dyn_L$(LBUILD).log 2>&1; \
	  if [ -f decode.xclbin ] && [ -f air_project/npu.air.txn.h ]; then \
	    mv -f decode.xclbin decode_dyn_L$(LBUILD).xclbin; \
	    cp -f air_project/npu.air.txn.h decode_dyn_L$(LBUILD).txn.h; \
	    rm -f npu_seq_*.mlir; \
	  else echo "decode_dyn_L$(LBUILD) FAILED (see /tmp/decode_dyn_L$(LBUILD).log)"; exit 1; fi
	@echo "Dynseq build complete: decode_dyn_L$(LBUILD).{xclbin,txn.h}. Run with DECODE_DYNSEQ=1."

## Staircase templates: one pair per ATTN_MAXL window. Each token dispatches on the
## smallest window covering the current context, so a short context streams
## proportionally less KV. Writes $(DECODE_WINDOWS_STAMP) so the driver can reject a
## directory whose templates do not match what was built.
WINDOWS ?= 64 128 256 512 1024 2048
DECODE_WINDOWS_STAMP := $(srcdir)/.decode_windows

compile-decode-windows:
	@for W in $(WINDOWS); do \
	  echo ">>> staircase window ATTN_MAXL=$$W"; \
	  $(MAKE) --no-print-directory -f $(srcdir)/Makefile _compile_decode_build LBUILD=$$W \
	    || exit 1; \
	done
	@echo "$(WINDOWS)" > $(DECODE_WINDOWS_STAMP)
	@echo "$(DECODE_FLAGS)" > $(DECODE_FLAGS_STAMP)
	@echo "Staircase templates built (ATTN_MAXL: $(WINDOWS)). Run with DECODE_STAIRCASE=1."

_compile_decode_build: preflight-llvm-link
	@echo ">>> non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) $(NONATTN_EXTRA) -O2 -c $(srcdir)/kernels/$$k.cc -o $(srcdir)/$$k.o || exit 1; done
	@echo ">>> attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@# attn_qk/attn_kv are built ONLY as inline .ll (merge-linked into the core; mlir-aie #3399).
	@# They MUST be -O1 (a -O2 attn do-while deadlock); non-attn MUST be -O2 (rope -O1 miscompiles).
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) $(ATTN_EXTRA) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(srcdir)/kernels/$$k.cc -o $(srcdir)/$$k.ll || exit 1; done
	@echo ">>> template decode_L2048 (+ L2047 slope ref) [region-major KV + fire-and-free readback + inline attn]"
	@# The two same-ATTN_MAXL builds (L2048 + L2047) calibrate the L-slope so DecodeInstsGen can
	@# patch any context length in [1, 2048] into one template at runtime (recompile-free decode).
	@# The block loop is single-buffered: air-label-scf-for-to-ping-pong declines it, because the
	@# running max and accumulator are live across blocks. Turbo for ~50 tok/s.
	@cd $(srcdir) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LBUILD) $(PYTHON) $(DECODE_DRIVER) >/tmp/decode_L$(LBUILD).log 2>&1; \
	  if [ -f decode.xclbin ]; then mv -f decode.xclbin decode_L$(LBUILD).xclbin; mv -f decode.insts.bin decode_L$(LBUILD).insts.bin; \
	  else echo "decode_L$(LBUILD) FAILED (see /tmp/decode_L$(LBUILD).log)"; exit 1; fi
	@cd $(srcdir) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LREF) $(PYTHON) $(DECODE_DRIVER) >/tmp/decode_L$(LREF).log 2>&1; \
	  if [ -f decode.insts.bin ]; then mv -f decode.insts.bin decode_L$(LREF).insts.bin; mv -f decode.xclbin decode_L$(LREF).xclbin; \
	  else echo "decode_L$(LREF) FAILED (see /tmp/decode_L$(LREF).log)"; exit 1; fi
	@echo "Decode build complete: decode_L$(LBUILD) + decode_L$(LREF) templates."

## ---- C++ decode-latency benchmark (FLM-comparable scope) --------------------
## `make bench-decode` times the same per-token sequence the Python driver's
## dispatch() runs, but in C++ and ending where FastFlowLM's DECODING_TIME
## accumulator stops: logits synced back, NO sampling and NO detokenization
## (FLM keeps those in separate accumulators, so they are not in its published
## tok/s). That is the only scope in which the two per-token numbers are
## directly comparable. The Python driver's tok/s additionally carries a
## 128k-element argmax (~0.14 ms) and ~0.10 ms of Python host work, so it reads
## ~0.25 ms/token slower than this for the same build.
##
## LATENCY ONLY -- it fills the weight/KV/rms BOs with a fixed pattern instead
## of loading the real requant, so it CANNOT check numerics. (Decode time here
## is data-independent: every trip count is fixed except the attention block
## loop, which is bounded by the patched RTP-L word. Verified -- synthetic vs
## real weights agree to 0.05%.) Correctness stays with the llama32_1b_q4nx
## driver's greedy id-sequence gate; never promote this to a gate.
##
## Needs both templates (compile-decode) since the per-L instruction stream is
## derived from their per-word slope, exactly as DecodeInstsGen does it.
## Geometry defaults are llama-3.2-1B; another model must pass --k/--w-elems/
## --rms-size/--ny/--kv-elems/--decode-y/--voc-n/--rms-lut-off explicitly or the
## numbers are silently wrong.
L        ?= 1933
ITERS    ?= 64
WARMUP   ?= 8
BENCH_EXE := $(srcdir)/bench_decode.exe

$(BENCH_EXE): $(srcdir)/bench_decode.cpp
	@GPP=$$( \
		for bin in /usr/bin/g++-*; do \
			ver=$$(echo $$bin | grep -oE '[0-9]+$$'); \
			if [ "$$ver" -ge 13 ] 2>/dev/null; then echo "$$ver $$bin"; fi; \
		done | sort -nr | head -n1 | awk '{print $$2}' \
	); \
	if [ -z "$$GPP" ]; then echo "Error: No g++ >= 13 in /usr/bin."; exit 1; fi; \
	if [ -z "$$XILINX_XRT" ]; then echo "Error: source /opt/xilinx/xrt/setup.sh."; exit 1; fi; \
	echo "Using compiler: $$GPP"; \
	$$GPP $(srcdir)/bench_decode.cpp -o $(BENCH_EXE) \
		-std=c++23 -O2 -Wall \
		-I$$XILINX_XRT/include -L$$XILINX_XRT/lib \
		-luuid -lxrt_coreutil -lrt -lstdc++

.PHONY: bench-decode-exe bench-decode
bench-decode-exe: $(BENCH_EXE)

bench-decode: $(BENCH_EXE)
	@if [ ! -f $(srcdir)/decode_L2048.insts.bin ] || [ ! -f $(srcdir)/decode_L2047.insts.bin ]; then \
	  echo "Need both templates; run 'make compile-decode' first."; exit 1; fi
	$(BENCH_EXE) --dir $(srcdir) --l $(L) --iters $(ITERS) --warmup $(WARMUP)

## Remove compiled kernels + decode templates.
clean:
	rm -f $(srcdir)/*.o $(srcdir)/*.ll $(srcdir)/decode.xclbin $(srcdir)/decode.insts.bin \
	      $(srcdir)/decode_L*.xclbin $(srcdir)/decode_L*.insts.bin \
	      $(BENCH_EXE) $(DECODE_FLAGS_STAMP) 2>/dev/null || true
	@echo "Kernels and decode templates removed. Rebuild with 'make compile-decode'."

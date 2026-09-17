#!/usr/bin/env bash
#===- run_qwen.sh ----------------------------------*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#===------------------------------------------------------------------===//
#
# Run a real Qwen3 checkpoint through the generated megakernel and check the
# tokens against an independent numpy implementation of the same model.
#
# Not in run_all.sh: it needs a checkpoint, which is gigabytes and not in the
# repository. Point it at one:
#
#   QWEN_DIR=/shared/erweiw/qwen3-0.6b test/gpu/megakernel_gen/run_qwen.sh
#
# The directory needs config.json and model.safetensors, as downloaded from
# Hugging Face. The first run converts the checkpoint into the flat float32
# blob the chain reads and leaves it in <QWEN_DIR>/air.
#
# The two sides being compared are: the chain (device, plus the host reference
# in the same program) and qwen3_ref.py, which reads the Hugging Face files
# directly. That second one is the point -- the chain agreeing with the host
# reference next to it only proves the two match.
#
#===------------------------------------------------------------------===//

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_DIR="${QWEN_DIR:?set QWEN_DIR to a directory holding config.json and model.safetensors}"
PY="${PY:-python3}"
TMPDIR="${TMPDIR:-/tmp/air_qwen}"
LAYERS="${LAYERS:-0}"        # 0 means every layer in the checkpoint
# A stage has `tasks` pieces at one token per step, so `tasks` is the ceiling
# on how many workgroups can be doing anything -- and each workgroup is one
# wavefront, so it is also the ceiling on how much memory latency the device
# has anything to hide behind. Measured on Qwen3-0.6B, 28 layers, six tokens,
# slope over eight extra launches:
#
#   workers/tasks    ms per token
#     32 /  32          227.0
#     64 /  64          179.1
#    128 / 128          153.0
#    256 / 128          261.1
#
# The last row is the shape of the cost: a workgroup with no piece left to
# claim does not go away, it spins on every event for the rest of the launch,
# so over-provisioning workers is worse than not provisioning them. Keep the
# two equal.
#
# 128 is the ceiling for this checkpoint, not a tuned optimum: gen.py requires
# tasks to divide the width of every stage it splits, and the vocabulary is
# 151936 = 128 * 1187. It asserts if they do not divide, so a checkpoint that
# cannot take 128 says so rather than computing a fraction of the work.
TASKS="${TASKS:-128}"
# Wavefronts per workgroup; see gen.py --waves.
WAVES="${WAVES:-8}"
WORKERS="${WORKERS:-128}"
# Decode steps in the single launch. Step 0 prefills the whole prompt; each
# later step carries one token, which is what makes this a decode rather than
# a prefill -- the window has to shrink after the prompt is used up.
STEPS="${STEPS:-1}"
# Launches per run. The wall time is dominated by reading three gigabytes of
# weights off NFS and by the single-threaded host reference, both of which
# happen once; the slope over REPEAT is what isolates the device.
REPEAT="${REPEAT:-1}"
# "The capital of France is"
PROMPT="${PROMPT:-785,6722,315,9625,374}"
mkdir -p "$TMPDIR"

[ -f "$QWEN_DIR/config.json" ] || { echo "no $QWEN_DIR/config.json" >&2; exit 1; }
[ -f "$QWEN_DIR/model.safetensors" ] || { echo "no $QWEN_DIR/model.safetensors" >&2; exit 1; }

if [ ! -f "$QWEN_DIR/air/manifest.json" ]; then
  echo "converting the checkpoint (once)..."
  "$PY" "$SCRIPT_DIR/weights.py" "$QWEN_DIR" "$QWEN_DIR/air"
fi
if [ "$LAYERS" = "0" ]; then
  LAYERS=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['config']['layers'])" "$QWEN_DIR/air/manifest.json")
fi
NTOK=$(awk -F, '{print NF}' <<< "$PROMPT")
# Tokens per step. Smaller than the prompt means the prompt is prefilled in
# chunks over several steps before any decoding starts, which is what Fleet's
# per-request cap does.
WIN="${WIN:-$NTOK}"
# Steps spent on the prompt, then one generated token per step from the last
# of them onwards.
PRE=$(( (NTOK + WIN - 1) / WIN ))
[ "$STEPS" -ge "$PRE" ] || { echo "STEPS must be at least $PRE to finish the prompt" >&2; exit 1; }
NGEN=$(( STEPS - PRE + 1 ))

if [ -z "${GFX_TARGET:-}" ]; then
  AMDGPU_ARCH_BIN=$(command -v amdgpu-arch || echo /opt/rocm/llvm/bin/amdgpu-arch)
  GFX_TARGET=$("$AMDGPU_ARCH_BIN" 2>/dev/null | head -1 | cut -d: -f1 || true)
fi
[ -n "$GFX_TARGET" ] || { echo "ERROR: set GFX_TARGET, e.g. GFX_TARGET=gfx942 $0" >&2; exit 1; }
echo "GFX_TARGET=$GFX_TARGET LAYERS=$LAYERS STEPS=$STEPS WIN=$WIN NGEN=$NGEN PROMPT=$PROMPT ($NTOK tokens)"

clang -O2 -shared -fPIC -o "$TMPDIR/libairweights.so" "$SCRIPT_DIR/weights_loader.c"

"$PY" "$SCRIPT_DIR/gen.py" --weights "$QWEN_DIR/air" --layers "$LAYERS" \
  --tasks "$TASKS" --workers "$WORKERS" --tokens "$WIN" --cache 0 --waves "$WAVES" \
  --steps "$STEPS" --repeat "$REPEAT" --prompt "$PROMPT" > "$TMPDIR/chain.mlir"
air-opt "$TMPDIR/chain.mlir" -air-to-rocdl -o "$TMPDIR/s1.mlir"
air-opt "$TMPDIR/s1.mlir" -air-gpu-outlining -o "$TMPDIR/s2.mlir"
mlir-opt "--pass-pipeline=builtin.module(func.func(lower-affine, convert-linalg-to-loops, convert-scf-to-cf), gpu-kernel-outlining)" \
    "$TMPDIR/s2.mlir" -o "$TMPDIR/s3.mlir"
mlir-opt "--pass-pipeline=builtin.module(rocdl-attach-target{chip=$GFX_TARGET O=3},gpu.module(convert-gpu-to-rocdl{chipset=$GFX_TARGET runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary, func.func(gpu-async-region),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" \
    "$TMPDIR/s3.mlir" -o "$TMPDIR/s4.mlir"

LLVM_LIB_DIR="${LLVM_INSTALL_DIR:+$LLVM_INSTALL_DIR/lib}"
LLVM_LIB_DIR="${LLVM_LIB_DIR:-$(dirname "$(which mlir-opt)")/../lib}"
MLIR_AIR_LIB_DIR="${MLIR_AIR_INSTALL_DIR:+$MLIR_AIR_INSTALL_DIR/lib}"
MLIR_AIR_LIB_DIR="${MLIR_AIR_LIB_DIR:-$(dirname "$(which air-opt)")/../lib}"

AIR_WEIGHTS="$QWEN_DIR/air/weights.f32" mlir-runner --entry-point-result=void \
    --shared-libs="$LLVM_LIB_DIR/libmlir_rocm_runtime.so" \
    --shared-libs="$LLVM_LIB_DIR/libmlir_runner_utils.so" \
    --shared-libs="$LLVM_LIB_DIR/libmlir_c_runner_utils.so" \
    --shared-libs="$MLIR_AIR_LIB_DIR/libairgpu.so" \
    --shared-libs="$TMPDIR/libairweights.so" \
    "$TMPDIR/s4.mlir" | tee "$TMPDIR/out.txt"

# The chain prints the reference token ids one per line after the marker.
# It prints a slot for every position the run could reach, and the ones past
# the last step are still zero, so only the first STEPS of them are generated.
GOT=$(sed -n '/^reference tokens:/,$p' "$TMPDIR/out.txt" \
      | sed 's/^reference tokens://' | grep -E '^[0-9]+$' \
      | head -n "$NGEN" | paste -sd, -)
WANT=$("$PY" "$SCRIPT_DIR/qwen3_ref.py" "$QWEN_DIR" "$PROMPT" "$LAYERS" "$NGEN" \
       | sed -n 's/^generated: //p' | tr -d ' ')
echo
echo "chain : $GOT"
echo "numpy : $WANT"
[ -n "$GOT" ] || { echo "FAIL: the chain printed no tokens"; exit 1; }
[ -n "$WANT" ] || { echo "FAIL: the numpy reference printed no tokens"; exit 1; }
if [ "$GOT" != "$WANT" ]; then echo "FAIL: tokens differ"; exit 1; fi
if ! tail -1 "$TMPDIR/out.txt" | grep -q '= 0$'; then
  echo "FAIL: $(tail -1 "$TMPDIR/out.txt")"; exit 1
fi
echo "PASS: device, host reference and an independent numpy Qwen3 all agree"

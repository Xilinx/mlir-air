# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# `make accuracy`: teacher-forced PPL + KL divergence of the model's NPU decode
# path against its reference on wikitext-2 (verify/accuracy.py). Informational;
# `make verify` is the correctness gate. The reference log-probs are computed
# once and cached under LLM_ACCURACY_CACHE (default ~/.cache/mlir-air/accuracy).
# The decode must be built for at least the chunk length (--ctx).
#
# Include at the END of a model Makefile, after PYTHON, srcdir and
# RUNNER_ADAPTER are defined.
#
#   ACC_CHUNKS  wikitext-2 chunks scored (accuracy.py --ctx sets their length)
#   ACC_REF     bf16 (original checkpoint) or dequant (the same quantized
#               weights, dequantized, from build_dequant_hf_model)
#   ACC_PATH    decode (prefill, then teacher-forced decode steps) or prefill
#               (every scored row from one prefill)
#   ACC_JSON    also write the result record to this path
#   ACC_MODEL   adapter MODEL_CHOICES key (default: $(MODEL), if set)
#   ACC_DIR     directory to run in (default: $(BUILD_DIR), as `make verify`)
#   ACC_ENV     environment assignments to run with
#   ACC_EXTRA   further accuracy.py arguments (e.g. --ctx 1024)

PYTHON     ?= python3
ACC_CHUNKS ?= 4
ACC_REF    ?= bf16
ACC_PATH   ?= decode
ACC_JSON   ?=
ACC_MODEL  ?= $(MODEL)
ACC_DIR    ?= $(BUILD_DIR)
ACC_ENV    ?=
ACC_EXTRA  ?=

.PHONY: accuracy
accuracy:
	@mkdir -p $(ACC_DIR)
	cd $(ACC_DIR) && $(ACC_ENV) $(PYTHON) $(srcdir)/../verify/accuracy.py \
		--runner=$(RUNNER_ADAPTER) $(if $(ACC_MODEL),--model $(ACC_MODEL)) \
		--chunks $(ACC_CHUNKS) --ref $(ACC_REF) --path $(ACC_PATH) $(ACC_EXTRA) \
		$(if $(ACC_JSON),--json-out $(abspath $(ACC_JSON)))


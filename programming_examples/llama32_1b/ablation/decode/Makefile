# Llama-3.2-1B Plan 2 (full decode) ablation harness
#
# make compile       — compile all 4 cells' ELFs + LM head (~5-10 min, cached)
# make regen-golden  — regenerate committed golden fixtures (rare; only after Cell D changes)
# make run           — run all 4 cells, 5 trials each, emit JSON
# make report        — generate markdown report from latest results JSON
# make test          — NPU-free unit tests (kv_cache + validation gate)
# make all           — compile + run + report
# make clean         — wipe build/

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
BUILD := build

.PHONY: help compile regen-golden run report test all clean

help:
	@echo "make compile | regen-golden | run | report | test | all | clean"

compile:
	@mkdir -p $(BUILD)
	cd $(BUILD) && PYTHONPATH=$(srcdir):$(srcdir)/..:$(srcdir)/../..:$(srcdir)/../..:$(srcdir)/../prefill:$(srcdir)/../../..:$$PYTHONPATH flock -x -w 1800 /tmp/mlir-air-npu.lock python3 -c "from cells.cell_d_merged import compile_cell_d; from cells.lm_head_const import compile_lm_head; from kernel_builder.cache import KernelCache; from golden.regen_golden import CONFIG; c = KernelCache(cache_dir='.', verbose=True); c.load_manifest(); compile_cell_d(c, CONFIG); compile_lm_head(c, CONFIG)"

regen-golden: compile
	cd $(BUILD) && PYTHONPATH=$(srcdir):$(srcdir)/..:$(srcdir)/../..:$(srcdir)/../prefill:$(srcdir)/../../..:$$PYTHONPATH flock -x -w 1800 /tmp/mlir-air-npu.lock python3 $(srcdir)/golden/regen_golden.py

run: compile
	cd $(BUILD) && PYTHONPATH=$(srcdir):$(srcdir)/..:$(srcdir)/../..:$(srcdir)/../prefill:$(srcdir)/../../..:$$PYTHONPATH flock -x -w 1800 /tmp/mlir-air-npu.lock python3 $(srcdir)/run_ablation.py --out results_latest.json

report:
	cd $(BUILD) && python3 $(srcdir)/analyze.py results_latest.json > report_latest.md && cat report_latest.md

test:
	cd $(srcdir) && python3 -m pytest tests/ -v

all: compile run report

clean:
	rm -rf $(BUILD)

# Llama-3.2-1B prefill ablation harness
#
# make compile       — compile all standalone ELFs + Cell D's 2 merged ELFs + FA (~10-15 min, cached)
# make regen-golden  — regenerate committed golden fixtures (rare; only after Cell D changes)
# make run           — run all 4 cells × 2 kernel-groups × both scopes, emit JSON
# make report        — generate markdown report from latest results JSON
# make all           — compile + run + report
# make clean         — wipe build/

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
BUILD := build

.PHONY: help compile regen-golden run report all clean

help:
	@echo "make compile | regen-golden | run | report | all | clean"

compile:
	@mkdir -p $(BUILD)
	cd $(BUILD) && PYTHONPATH=$(srcdir):$(srcdir)/..:$(srcdir)/../..:$(srcdir)/../../..:$$PYTHONPATH flock -x -w 1800 /tmp/mlir-air-npu.lock python3 -m cells.common

regen-golden: compile
	cd $(BUILD) && PYTHONPATH=$(srcdir):$(srcdir)/..:$(srcdir)/../..:$(srcdir)/../../..:$$PYTHONPATH flock -x -w 1800 /tmp/mlir-air-npu.lock python3 $(srcdir)/golden/regen_golden.py

run: compile
	cd $(BUILD) && PYTHONPATH=$(srcdir):$(srcdir)/..:$(srcdir)/../..:$(srcdir)/../../..:$$PYTHONPATH flock -x -w 1800 /tmp/mlir-air-npu.lock python3 $(srcdir)/run_ablation.py --out results_latest.json

report:
	cd $(BUILD) && python3 $(srcdir)/analyze.py results_latest.json --out report_latest.md && cat report_latest.md

all: compile run report

clean:
	rm -rf $(BUILD)

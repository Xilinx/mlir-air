# ./test/lit.cfg.py -*- Python -*-
#
# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -*- Python -*-

import os
import platform
import re
import subprocess
import shutil
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIR_PROGRAMMING_EXAMPLES"

# The internal shell, unconditionally. `not llvm_config.use_lit_shell` was the
# old LLVM idiom for "external shell except on Windows", and lit 23 removed the
# external shell: ShTest(execute_external=True) now raises at config-parse time,
# so the suite fails before a single test runs.
#
# Migrating rather than setting force_execute_external=True, which is an
# explicit stay of execution that LLVM-24 removes anyway. The one incompatibility
# in this repo was a bare `VAR=value` command prefix, which bash treats as an
# environment assignment and lit's internal shell treats as a command name; see
# the %ld_lib_path substitution in mlir/test/lit.cfg.py.
config.test_format = lit.formats.ShTest()
config.environment["PYTHONPATH"] = os.pathsep.join(
    [
        os.path.join(config.air_obj_root, "python"),
        os.path.join(config.aie_obj_root, "python"),
        os.path.join(config.xrt_dir, "python"),
    ]
)

# os.environ['PYTHONPATH']
print("Running with PYTHONPATH", config.environment["PYTHONPATH"])

# The example Makefiles resolve the AIE headers through MLIR_AIE_INSTALL_DIR
# rather than by locating aie-opt on PATH, so point it at the mlir-aie this
# build was configured against. An explicit setting in the environment wins,
# matching how llms/shared/infra/external_kernels.py treats the same variable.
config.environment["MLIR_AIE_INSTALL_DIR"] = os.environ.get(
    "MLIR_AIE_INSTALL_DIR", config.aie_obj_root
)

# Makefiles and tests that shell out to Python must use the interpreter this
# build was configured with, not whichever one PATH happens to resolve first.
config.environment["PYTHON"] = config.python_executable

# Examples that build a host program need the XRT this build was configured
# against. On Linux that arrives via run_on_npu.sh sourcing setup.sh, which
# exports XILINX_XRT; there is no such script on Windows, so pass the
# configured path down directly. XILINX_XRT still wins where it is set.
config.environment["XRT_ROOT"] = os.environ.get("XRT_ROOT", config.xrt_dir)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".lit"]

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = []

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, "programming_examples")
air_runtime_lib = os.path.join(
    config.air_obj_root, "runtime_lib", config.runtime_test_target
)

config.substitutions.append(("%PYTHON", config.python_executable))
config.substitutions.append(("%CLANG", "clang++ -fuse-ld=lld -DLIBXAIENGINEV2"))
config.substitutions.append(("%LIBXAIE_DIR%", config.libxaie_dir))
config.substitutions.append(
    (
        "%AIE_RUNTIME_DIR%",
        os.path.join(config.aie_obj_root, "runtime_lib", config.runtime_test_target),
    )
)
config.substitutions.append(("%aietools", config.vitis_aietools_dir))

# for xchesscc_wrapper
llvm_config.with_environment("AIETOOLS", config.vitis_aietools_dir)

run_on_npu1 = "echo"
run_on_npu2 = "echo"
xrt_flags = ""

# XRT
if config.xrt_lib_dir and config.enable_run_xrt_tests:
    print("xrt found at", os.path.dirname(config.xrt_lib_dir))
    xrt_flags = "-I{} -L{} -luuid -lxrt_coreutil".format(
        config.xrt_include_dir, config.xrt_lib_dir
    )
    config.available_features.add("xrt")

    try:
        xrtsmi = os.path.join(config.xrt_bin_dir, "xrt-smi")
        # Windows ships xrt-smi.exe with the NPU driver (System32\AMD, on PATH)
        # rather than under the XRT SDK, and it needs the .exe suffix. which()
        # covers both; on Linux the path above already resolves and is used.
        xrtsmi = shutil.which(xrtsmi) or shutil.which("xrt-smi") or xrtsmi
        result = subprocess.run(
            [xrtsmi, "examine"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out = result.stdout.decode("utf-8")
        out_lc = out.lower()
        # Case-insensitive substring match against known NPU model strings,
        # aligned with mlir-aie's hostruntime.py and the AIR XRT backend
        # (NPU_MODELS). Robust to xrt-smi table-format changes and covers newer
        # parts (Strix Halo, Krackan) that the old strict regex missed.
        npu2_models = ["npu4", "strix", "npu5", "strix halo", "npu6", "krackan"]
        npu1_models = ["npu1", "phoenix"]
        # Serializing through flock and run_on_npu.sh is POSIX-only: flock is
        # util-linux, and the script is bash that sources
        # /opt/xilinx/xrt/setup.sh. Now that xrt-smi is found on Windows too,
        # reusing it there would let the suite configure and then fail to launch
        # every test carrying this substitution. Run the command directly
        # instead -- the environment is already set up by whoever invoked lit.
        if os.name == "nt":
            run_on_npu = ""
        else:
            run_on_npu = (
                f"flock /tmp/npu.lock {config.air_src_root}/utils/run_on_npu.sh"
            )
        if any(k in out_lc for k in npu2_models):
            config.available_features.add("ryzen_ai")
            config.available_features.add("ryzen_ai_npu2")
            # Resolved once here so the ~230 test processes in a suite run do not
            # each re-run xrt-smi examine. That probe has a 10 s timeout and
            # intermittently exceeds it under load; on timeout the backend falls
            # back to npu1 and silently builds a 4-column design for an 8-column part.
            config.environment["AIR_TARGET_DEVICE"] = "npu2"
            run_on_npu2 = run_on_npu
            print("Running tests on NPU2 with command line: ", run_on_npu2 or "(none)")
        elif any(k in out_lc for k in npu1_models):
            config.available_features.add("ryzen_ai")
            config.available_features.add("ryzen_ai_npu1")
            config.environment["AIR_TARGET_DEVICE"] = "npu1"
            run_on_npu1 = run_on_npu
            print("Running tests on NPU1 with command line: ", run_on_npu1 or "(none)")
        else:
            # No recognized model: dump xrt-smi output so the cause (format
            # change, or a driver error such as an mmap/memlock failure) is
            # visible instead of silently skipping every NPU test.
            print("WARNING: xrt-smi did not report a recognized NPU model.")
            print("xrt-smi returncode:", result.returncode)
            print("xrt-smi examine stdout:\n" + out)
            print("xrt-smi examine stderr:\n" + result.stderr.decode("utf-8"))
    except Exception as e:
        print(f"Failed to run xrt-smi: {e}")
else:
    print("xrt not found or xrt tests disabled")
    config.excludes.append("xrt")

config.substitutions.append(("%run_on_npu1%", run_on_npu1))
config.substitutions.append(("%run_on_npu2%", run_on_npu2))
config.substitutions.append(("%xrt_flags", xrt_flags))
config.substitutions.append(("%XRT_DIR", config.xrt_dir))

# Tests that download Hugging Face Hub gated models (e.g. meta-llama/*) need
# HF_TOKEN to be set. Mark `hf_token` as available only when the env var is
# present so REQUIRES: hf_token tests skip cleanly on machines without it.
if os.environ.get("HF_TOKEN"):
    config.available_features.add("hf_token")
    llvm_config.with_environment("HF_TOKEN", os.environ["HF_TOKEN"])
    print("HF_TOKEN found in environment; hf_token feature enabled.")
else:
    print("HF_TOKEN not set; hf_token feature disabled.")

# Forward HF Hub download tuning if the host set it (e.g. the perf runner sets
# HF_HUB_DISABLE_XET=1 because the hf_xet backend stalls there). Propagated only
# when present, so it's a no-op on hosts that don't set it.
if os.environ.get("HF_HUB_DISABLE_XET"):
    llvm_config.with_environment("HF_HUB_DISABLE_XET", os.environ["HF_HUB_DISABLE_XET"])

# Forward HF_HUB_OFFLINE when the host sets it (the perf runner sets it to 1 so
# from_pretrained serves the pre-seeded weight cache without a network HEAD --
# the runner's HF egress is flaky). lit sanitizes the environment, so tests only
# see vars explicitly forwarded here. No-op when unset. Weights are seeded by the
# manual downloadLLMWeights.yml workflow.
if os.environ.get("HF_HUB_OFFLINE"):
    llvm_config.with_environment("HF_HUB_OFFLINE", os.environ["HF_HUB_OFFLINE"])


# Gate each LLM example on the weights it loads actually being present in the
# local HF cache, so the suite runs whatever is seeded and auto-includes a model
# the moment its weights land -- no hardcoded exclusion to maintain. Each model's
# verify/profile .lit declares `REQUIRES: hfweights_<normalized repo id>`; here we
# add that feature for every repo present in the cache. The scan is a local disk
# read (offline-safe); a no-op if huggingface_hub is unimportable or the cache is
# empty, in which case those tests report UNSUPPORTED instead of failing on a
# missing checkpoint. The perf runner's HF egress is flaky, so weights are seeded
# out of band; this keeps CI green on a partially-seeded cache.
def _hf_weight_feature(repo_id):
    return "hfweights_" + re.sub(r"[^a-z0-9]+", "_", repo_id.lower()).strip("_")


try:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.utils import CacheNotFound

    try:
        _hf_cache = scan_cache_dir()
    except CacheNotFound:
        _hf_cache = None
    if _hf_cache is not None:
        _seeded = sorted(
            _hf_weight_feature(r.repo_id)
            for r in _hf_cache.repos
            if any(rev.size_on_disk > 0 for rev in r.revisions)
        )
        for _feat in _seeded:
            config.available_features.add(_feat)
        print("HF weights present -> features:", ", ".join(_seeded) or "(none)")
except ImportError:
    print("huggingface_hub not importable; hfweights_* features disabled.")

# The SmolVLA example verifies against the upstream `lerobot` policy running on
# CPU, which is a heavier dependency than the other LLM examples need (torch +
# lerobot[dataset] + num2words; see llms/smolvla/requirements.txt). CI installs
# only llama32_1b's requirements, so mark `lerobot` available when it actually
# imports and let REQUIRES: lerobot tests skip cleanly instead of failing on an
# ImportError. Its compile test needs none of this and carries no such REQUIRES.
try:
    import lerobot  # noqa: F401

    config.available_features.add("lerobot")
    print("lerobot importable; lerobot feature enabled.")
except ImportError:
    print("lerobot not installed; lerobot feature disabled.")

# OS is forwarded because lit sanitizes the environment: without it, make's
# $(OS) is empty under lit and any Windows_NT test in a Makefile silently takes
# the POSIX branch. Unset on Linux, so forwarding it is a no-op there.
llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP", "OS"])

llvm_config.use_default_substitutions()

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, "test")
config.aie_tools_dir = os.path.join(config.aie_obj_root, "bin")
config.air_tools_dir = os.path.join(config.air_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.peano_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.aie_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.air_tools_dir, append_path=True)

config.substitutions.append(("%LLVM_TOOLS_DIR", config.llvm_tools_dir))

tool_dirs = [config.aie_tools_dir, config.llvm_tools_dir]

# Test if Peano is available
try:
    result = subprocess.run(
        [os.path.join(config.peano_tools_dir, "llc"), "-mtriple=aie", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if re.search("Xilinx AI Engine", result.stdout.decode("utf-8")) is not None:
        config.available_features.add("peano")
        config.substitutions.append(
            ("%PEANO_INSTALL_DIR", os.path.dirname(config.peano_tools_dir))
        )
        print("Peano found: " + os.path.join(config.peano_tools_dir, "llc"))
        peano_flags = "-O2 -std=c++20 -DNDEBUG -I{}".format(
            os.path.join(config.aie_obj_root, "include")
        )
        config.substitutions.append(("%peano_flags", peano_flags))
    else:
        print("Peano not detected at expected path:", config.peano_tools_dir)
except Exception:
    print("Peano check failed.")

# The fused-decode inline-attn merge (programming_examples/fused_decode) shells out
# to an external `llvm-link` that MUST be a pre-lifetime-change LLVM (< 23); a newer
# llvm-link rewrites the llvm.lifetime intrinsic and Peano opt then rejects the
# merged IR. Expose a feature so decode-e2e tests (e.g. llama32_1b_q4nx
# verify/profile) report UNSUPPORTED rather than hard-fail on a runner where no
# compatible llvm-link is on PATH.
#
# Resolve it against the PATH the TESTS get (config.environment), not this
# process's os.environ: the block above prepends config.llvm_tools_dir -- the
# mlir distro, currently LLVM 24 -- ahead of the ambient PATH, so a usable
# llvm-link further down the ambient PATH is shadowed inside the tests. Checking
# os.environ instead reported "enabled" while the tests ran the distro's LLVM 23
# and died in the fused_decode preflight.
#
# When the usable binary is NOT the one the tests would pick, expose it through a
# shim dir holding only a symlink to it, prepended to the test PATH. Prepending
# its real directory instead would be unsafe -- for a dev with /usr/bin/llvm-link
# that would put /usr/bin ahead of the AIE toolchain and shadow clang/opt/llc.
try:
    _test_path = config.environment.get("PATH", "")
    _llvm_link = shutil.which("llvm-link", path=_test_path)

    def _llvm_link_major(exe):
        out = subprocess.run(
            [exe, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout.decode("utf-8")
        m = re.search(r"version (\d+)", out)
        return (int(m.group(1)) if m else None), out

    _ll_ver, _ll_out = _llvm_link_major(_llvm_link) if _llvm_link else (None, "")

    # Shadowed-but-present case: the winner is unusable, so look for a usable one
    # further along the same PATH and shim it to the front.
    if _llvm_link and (_ll_ver is None or _ll_ver >= 23):
        for _d in _test_path.split(os.pathsep):
            _cand = os.path.join(_d, "llvm-link")
            if not os.access(_cand, os.X_OK) or os.path.isdir(_cand):
                continue
            _cand_ver, _cand_out = _llvm_link_major(_cand)
            if _cand_ver is not None and _cand_ver < 23:
                _shim = os.path.join(config.air_obj_root, "llvm-link-shim")
                os.makedirs(_shim, exist_ok=True)
                _link = os.path.join(_shim, "llvm-link")
                if os.path.lexists(_link):
                    os.remove(_link)
                os.symlink(_cand, _link)
                llvm_config.with_environment("PATH", _shim, append_path=True)
                # _ll_ver is None when --version did not parse, which is a
                # separate reason for rejecting it than being >=23; do not
                # report an unknown version as though it were a known one.
                _why = (
                    f"is LLVM {_ll_ver}"
                    if _ll_ver is not None
                    else "has an unparseable version"
                )
                print(
                    f"llvm-link on the test PATH ({_llvm_link}) {_why}; shimming "
                    f"{_cand} (LLVM {_cand_ver}) ahead of it."
                )
                _llvm_link, _ll_ver, _ll_out = _cand, _cand_ver, _cand_out
                break

    if _llvm_link and _ll_ver is not None and _ll_ver < 23:
        config.available_features.add("llvm_link_pre23")
        print(f"llvm-link {_ll_ver} (<23) found: fused-decode merge enabled.")
    elif _llvm_link:
        print(
            "llvm-link on the test PATH is not <23 "
            f"({_ll_out.strip().splitlines()[0] if _ll_out.strip() else '?'}); "
            "fused-decode decode tests will be UNSUPPORTED."
        )
    else:
        print(
            "llvm-link not on the test PATH; "
            "fused-decode decode tests will be UNSUPPORTED."
        )
except Exception as e:
    print(f"llvm-link check failed: {e}")

# Test if Chess is available
if not config.enable_chess_tests:
    print("Chess tests disabled.")
else:
    print("Looking for Chess...")

    chess_path = shutil.which("xchesscc")
    if chess_path:
        print("Chess found: " + chess_path)
        config.available_features.add("chess")
        lm_license_file = os.getenv("LM_LICENSE_FILE")
        xilinxd_license_file = os.getenv("XILINXD_LICENSE_FILE")

        if lm_license_file:
            llvm_config.with_environment("LM_LICENSE_FILE", lm_license_file)
        if xilinxd_license_file:
            llvm_config.with_environment("XILINXD_LICENSE_FILE", xilinxd_license_file)

        # Optionally validate license
        validate_chess = False
        if validate_chess:
            result = subprocess.run(
                ["xchesscc", "+v"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if len(result.stderr.decode("utf-8")) == 0:
                config.available_features.add("valid_xchess_license")
        else:
            if lm_license_file or xilinxd_license_file:
                config.available_features.add("valid_xchess_license")
            else:
                print("WARNING: Chess license environment variables not found.")

    elif os.getenv("XILINXD_LICENSE_FILE") is not None:
        print("Chess license found")
        llvm_config.with_environment(
            "XILINXD_LICENSE_FILE", os.getenv("XILINXD_LICENSE_FILE")
        )
    else:
        print("Chess not found")

tool_dirs = [
    config.peano_tools_dir,
    config.aie_tools_dir,
    config.air_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    "aie-opt",
    "aie-translate",
    "aiecc.py",
    "aircc",
    "air-opt",
    "ld.lld",
    "llc",
    "llvm-objdump",
    "mlir-translate",
    "opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

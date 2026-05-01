//===- aircc.cpp ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This is the main entry point for the AIR compiler driver (aircc).
// It orchestrates the compilation flow for AIR MLIR programs targeting
// AIE/NPU devices.
//
// This is the native C++ aircc compiler driver with the following
// architecture:
//
// 1. Command-line argument parsing using LLVM CommandLine library
// 2. MLIR module loading and parsing
// 3. AIR transformation pipeline execution (in-process via PassManager)
// 4. AIR-to-AIE conversion
// 5. Invocation of aiecc (mlir-aie C++ compiler driver) for backend
// 6. Host code compilation (optional, for non-NPU targets)
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/InitAll.h"

#if AIR_ENABLE_AIE
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#endif

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

static cl::OptionCategory airCompilerOptions("AIR Compiler Options");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input AIR MLIR file>"),
                                          cl::Required,
                                          cl::cat(airCompilerOptions));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::init(""),
                                           cl::cat(airCompilerOptions));

static cl::opt<std::string>
    instsFilename("i", cl::desc("Output insts file name (NPU only)"),
                  cl::init(""), cl::cat(airCompilerOptions));

static cl::opt<std::string> tmpDir("tmpdir",
                                   cl::desc("Directory for temporary files"),
                                   cl::init("air_project"),
                                   cl::cat(airCompilerOptions));

static cl::opt<bool> verbose("verbose", cl::desc("Enable verbose output"),
                             cl::init(false), cl::cat(airCompilerOptions));

static cl::alias verboseShort("v", cl::desc("Alias for --verbose"),
                              cl::aliasopt(verbose),
                              cl::cat(airCompilerOptions));

static cl::opt<int>
    rowOffset("row-offset",
              cl::desc("Default row offset for generated segments"),
              cl::init(-1), cl::cat(airCompilerOptions));

static cl::opt<int>
    colOffset("col-offset",
              cl::desc("Default column offset for generated segments"),
              cl::init(-1), cl::cat(airCompilerOptions));

static cl::opt<int>
    numRows("num-rows",
            cl::desc("Default number of rows for generated segments"),
            cl::init(-1), cl::cat(airCompilerOptions));

static cl::opt<int>
    numCols("num-cols",
            cl::desc("Default number of columns for generated segments"),
            cl::init(-1), cl::cat(airCompilerOptions));

static cl::opt<unsigned>
    traceSize("trace-size",
              cl::desc("Create packet routed traces for cores and memtiles"),
              cl::init(0), cl::cat(airCompilerOptions));

static cl::opt<unsigned>
    traceOffset("trace-offset",
                cl::desc("Trace buffer offset appended to output"), cl::init(0),
                cl::cat(airCompilerOptions));

static cl::opt<std::string> cc("cc", cl::desc("Compiler to use for host code"),
                               cl::init("clang"), cl::cat(airCompilerOptions));

static cl::opt<std::string> sysroot("sysroot",
                                    cl::desc("Sysroot for cross-compilation"),
                                    cl::init(""), cl::cat(airCompilerOptions));

static cl::opt<std::string>
    hostTarget("host-target",
               cl::desc("Target architecture of the host program"),
               cl::init(""), cl::cat(airCompilerOptions));

static cl::opt<bool>
    shared("shared", cl::desc("Generate a shared library instead of static"),
           cl::init(false), cl::cat(airCompilerOptions));

static cl::opt<bool> xbridge("xbridge", cl::desc("Link using xbridge"),
                             cl::init(false), cl::cat(airCompilerOptions));

static cl::opt<bool> noXbridge("no-xbridge", cl::desc("Link using peano"),
                               cl::init(false), cl::cat(airCompilerOptions));

static cl::opt<bool> xchesscc("xchesscc", cl::desc("Compile using xchesscc"),
                              cl::init(false), cl::cat(airCompilerOptions));

static cl::opt<bool> noXchesscc("no-xchesscc", cl::desc("Compile using peano"),
                                cl::init(false), cl::cat(airCompilerOptions));

static cl::opt<std::string> peanoInstallDir(
    "peano",
    cl::desc("Root directory where peano compiler is installed. "
             "Falls back to PEANO_INSTALL_DIR env var if not specified."),
    cl::init(""), cl::cat(airCompilerOptions));

static cl::opt<std::string> deviceName("device", cl::desc("Target AIE device"),
                                       cl::init("xcvc1902"),
                                       cl::cat(airCompilerOptions));

static cl::opt<std::string> target("target",
                                   cl::desc("Target backend: 'aie' or 'gpu'"),
                                   cl::init("aie"),
                                   cl::cat(airCompilerOptions));

static cl::opt<std::string>
    gpuArch("gpu-arch",
            cl::desc("GPU architecture for ROCDL target (e.g., gfx942)"),
            cl::init("gfx942"), cl::cat(airCompilerOptions));

static cl::opt<std::string>
    gpuRuntime("gpu-runtime",
               cl::desc("GPU runtime for ROCDL target (HIP or OpenCL)"),
               cl::init("HIP"), cl::cat(airCompilerOptions));

static cl::opt<bool>
    omitWhileTrueLoop("omit-while-true-loop",
                      cl::desc("Do not add while(true) loop around per-core "
                               "logic"),
                      cl::init(false), cl::cat(airCompilerOptions));

// Use ValueOptional so --omit-ping-pong-transform (no value) defaults to "all"
// matching Python argparse nargs="?" const="all" behavior.
static cl::opt<std::string> omitPingpong(
    "omit-ping-pong-transform",
    cl::desc("Omit ping-pong buffering (values: '', L1, L2, all). "
             "Using the flag without a value is equivalent to 'all'."),
    cl::init(""), cl::ValueOptional, cl::cat(airCompilerOptions));

static cl::opt<std::string>
    lowerLinalgToFunc("lower-linalg-to-func",
                      cl::desc("Lower linalg.generic to function calls with "
                               "given object file"),
                      cl::init(""), cl::cat(airCompilerOptions));

static cl::opt<bool>
    airLoopFusion("air-loop-fusion",
                  cl::desc("Add air-loop-fusion pass to the pipeline"),
                  cl::init(false), cl::cat(airCompilerOptions));

static cl::list<unsigned> runtimeLoopTilingSizes(
    "air-runtime-loop-tiling-sizes",
    cl::desc("Tiling factors for runtime host affine loop nest. "
             "Omit to disable tiling; provide one or more values to enable."),
    cl::cat(airCompilerOptions));

// Track whether the flag was present on the command line at all
static bool runtimeLoopTilingSizesPresent = false;

static cl::opt<bool> omitAutoBroadcast(
    "omit-auto-broadcast",
    cl::desc("Omit air-broadcast-detection / specialize-dma-broadcast"),
    cl::init(false), cl::cat(airCompilerOptions));

static cl::list<std::string> channelMultiplexing(
    "air-channel-multiplexing",
    cl::desc("Memory spaces for air channel time-multiplexing"),
    cl::cat(airCompilerOptions));

static cl::opt<bool> useLockRaceConditionFix(
    "use-lock-race-condition-fix",
    cl::desc("Enable fix for lock race condition (inserts extra dummy BDs)"),
    cl::init(false), cl::cat(airCompilerOptions));

enum OutputFormatKind { OF_xclbin, OF_txn, OF_elf, OF_none };

static cl::opt<OutputFormatKind> outputFormat(
    "output-format", cl::desc("Output format for the generated binary"),
    cl::values(clEnumValN(OF_xclbin, "xclbin", "Generate xclbin"),
               clEnumValN(OF_txn, "txn", "Generate transaction binary"),
               clEnumValN(OF_elf, "elf", "Generate ELF"),
               clEnumValN(OF_none, "none", "Compile-only, no binary output")),
    cl::init(OF_xclbin), cl::cat(airCompilerOptions));

static cl::opt<std::string> kernelName("xclbin-kernel-name",
                                       cl::desc("Kernel name in xclbin file"),
                                       cl::init(""),
                                       cl::cat(airCompilerOptions));

static cl::opt<std::string>
    instanceName("xclbin-instance-name",
                 cl::desc("Instance name in xclbin metadata"), cl::init(""),
                 cl::cat(airCompilerOptions));

static cl::opt<std::string> kernelId("xclbin-kernel-id",
                                     cl::desc("Kernel id in xclbin file"),
                                     cl::init(""), cl::cat(airCompilerOptions));

static cl::opt<std::string>
    xclbinInput("xclbin-input",
                cl::desc("Generate kernel into existing xclbin file"),
                cl::init(""), cl::cat(airCompilerOptions));

static cl::opt<std::string>
    elfName("elf-name",
            cl::desc("Output filename for full ELF (--output-format=elf)"),
            cl::init("aie.elf"), cl::cat(airCompilerOptions));

static cl::opt<bool>
    debugIr("debug-ir",
            cl::desc("Emit IR after each pass to <tmpdir>/debug_ir/"),
            cl::init(false), cl::cat(airCompilerOptions));

static cl::opt<bool>
    bf16Emulation("bf16-emulation",
                  cl::desc("Emulate f32 vector arithmetic using bf16"),
                  cl::init(false), cl::cat(airCompilerOptions));

static cl::opt<unsigned> stackSize(
    "stack-size",
    cl::desc("Stack size in bytes per AIE core (default: 1024). Increase when "
             "kernels have deep call chains (e.g., scalar fdiv)."),
    cl::init(1024u), cl::cat(airCompilerOptions));

//===----------------------------------------------------------------------===//
// Debug IR Support
//===----------------------------------------------------------------------===//

static unsigned passCounter = 0;
static std::string debugIrDir;

/// Represents one entry in the pass log.
struct PassLogEntry {
  unsigned index;
  std::string passName;
  std::string outputFile;
  bool isCheckpoint = false;
  std::string equivalentFile; // For checkpoints
};

static std::vector<PassLogEntry> passLog;

/// Extract a short name from a pass string for filename use.
/// Examples:
///   "canonicalize" -> "canonicalize"
///   "air-to-aie{emit-while-loop=true...}" -> "air-to-aie"
///   "func.func(air-split-l2-memref)" -> "func.func_air-split-l2-memref"
static std::string getPassShortName(StringRef passStr) {
  // Remove arguments in braces
  std::string name = passStr.split('{').first.str();
  // Replace parentheses with underscores for nested passes
  for (char &c : name) {
    if (c == '(')
      c = '_';
    else if (c == ')')
      c = '\0'; // will be trimmed
  }
  // Trim trailing nulls/underscores
  while (!name.empty() && (name.back() == '\0' || name.back() == '_'))
    name.pop_back();
  return name;
}

/// Split a pass pipeline string into individual passes.
/// Handles nested parentheses and braces:
///   "pass1,pass2{opt=val},func.func(nested-pass)"
///   -> ["pass1", "pass2{opt=val}", "func.func(nested-pass)"]
static std::vector<std::string> splitPipelineToPasses(StringRef pipeline) {
  // Remove builtin.module() wrapper if present
  StringRef inner = pipeline;
  if (inner.starts_with("builtin.module(") && inner.ends_with(")"))
    inner = inner.drop_front(15).drop_back(1);

  std::vector<std::string> passes;
  std::string current;
  int depth = 0;

  for (char c : inner) {
    if (c == '(' || c == '{') {
      ++depth;
      current += c;
    } else if (c == ')' || c == '}') {
      --depth;
      current += c;
    } else if (c == ',' && depth == 0) {
      // Top-level comma — end of current pass
      StringRef trimmed = StringRef(current).trim();
      if (!trimmed.empty())
        passes.push_back(trimmed.str());
      current.clear();
    } else {
      current += c;
    }
  }

  // Don't forget the last pass
  StringRef trimmed = StringRef(current).trim();
  if (!trimmed.empty())
    passes.push_back(trimmed.str());

  return passes;
}

/// Save IR to file after a pass (in debug mode).
static void saveDebugIr(ModuleOp moduleOp, StringRef passName) {
  if (debugIrDir.empty())
    return;

  ++passCounter;
  std::string shortName = getPassShortName(passName);
  SmallString<256> irFile(debugIrDir);
  char counterBuf[8];
  snprintf(counterBuf, sizeof(counterBuf), "%03u", passCounter);
  std::string filename =
      "pass_" + std::string(counterBuf) + "_after_" + shortName + ".mlir";
  // Sanitize any remaining special characters
  for (char &c : filename) {
    if (c == '{' || c == '}' || c == '(' || c == ')' || c == ' ' || c == '=' ||
        c == ',')
      c = '_';
  }
  sys::path::append(irFile, filename);

  std::error_code ec;
  raw_fd_ostream os(irFile, ec);
  if (!ec) {
    moduleOp.print(os);
  }

  if (verbose) {
    llvm::outs() << "[PASS " << format("%03u", passCounter) << "] " << passName
                 << "\n";
  }

  // Record in pass log
  PassLogEntry entry;
  entry.index = passCounter;
  entry.passName = passName.str();
  entry.outputFile = irFile.str().str();
  passLog.push_back(entry);
}

/// Add a checkpoint marker to the pass log.
static void addCheckpoint(StringRef name, StringRef equivalentFile) {
  PassLogEntry entry;
  entry.index = 0;
  entry.passName = name.str();
  entry.isCheckpoint = true;
  entry.equivalentFile = equivalentFile.str();
  if (!passLog.empty())
    entry.outputFile = passLog.back().outputFile;
  passLog.push_back(entry);
}

/// Write the pass log to a file.
static void dumpPassLog() {
  if (debugIrDir.empty())
    return;

  SmallString<256> logFile(debugIrDir);
  sys::path::append(logFile, "pass.log");

  std::error_code ec;
  raw_fd_ostream os(logFile, ec);
  if (ec)
    return;

  os << "# MLIR-AIR Compilation Pass Log\n";
  os << "# Generated by aircc (C++)\n";
  os << "#\n";
  os << "# Each pass_XXX_after_*.mlir file shows the IR AFTER that pass.\n";
  os << "# CHECKPOINT markers indicate standard output file equivalents.\n";
  os << "#\n\n";

  for (const auto &entry : passLog) {
    if (entry.isCheckpoint) {
      os << "\n";
      for (int i = 0; i < 70; ++i)
        os << "=";
      os << "\n";
      os << "CHECKPOINT: " << entry.passName << "\n";
      os << "  This is equivalent to: " << entry.equivalentFile << "\n";
      os << "  Debug IR file: " << entry.outputFile << "\n";
      for (int i = 0; i < 70; ++i)
        os << "=";
      os << "\n\n";
    } else {
      os << "[PASS " << format("%03u", entry.index) << "] " << entry.passName
         << "\n";
      if (!entry.outputFile.empty())
        os << "  -> Output: " << entry.outputFile << "\n";
    }
  }
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Forward declarations
static LogicalResult saveModule(ModuleOp moduleOp, StringRef path);

/// Execute a command and return success/failure.
static LogicalResult runCommand(ArrayRef<std::string> command) {
  if (verbose) {
    for (const auto &arg : command) {
      llvm::outs() << arg << " ";
    }
    llvm::outs() << "\n";
  }

  // Build argument list as StringRef vector
  std::vector<StringRef> args;
  for (const auto &arg : command) {
    args.push_back(arg);
  }

  std::string errMsg;
  auto program = sys::findProgramByName(command[0]);
  if (!program) {
    llvm::errs() << "Error: could not find program: " << command[0] << "\n";
    return failure();
  }

  int result = sys::ExecuteAndWait(*program, args,
                                   /*Env=*/std::nullopt,
                                   /*Redirects=*/{},
                                   /*secondsToWait=*/0,
                                   /*memoryLimit=*/0, &errMsg);
  if (result != 0) {
    llvm::errs() << "Error running command: " << command[0];
    if (!errMsg.empty())
      llvm::errs() << " (" << errMsg << ")";
    llvm::errs() << "\n";
    return failure();
  }
  return success();
}

/// Execute a command and capture stdout.
static LogicalResult runCommandCaptureOutput(ArrayRef<std::string> command,
                                             std::string &output) {
  // Create temp file for stdout
  SmallString<128> tempFile;
  std::error_code ec = sys::fs::createTemporaryFile("aircc", "out", tempFile);
  if (ec) {
    llvm::errs() << "Error creating temp file: " << ec.message() << "\n";
    return failure();
  }

  // Build args
  std::vector<StringRef> args;
  for (const auto &arg : command) {
    args.push_back(arg);
  }

  auto program = sys::findProgramByName(command[0]);
  if (!program) {
    llvm::errs() << "Error: could not find program: " << command[0] << "\n";
    return failure();
  }

  // Also capture stderr to a temp file for diagnostics on failure
  SmallString<128> stderrFile;
  ec = sys::fs::createTemporaryFile("aircc", "err", stderrFile);
  if (ec) {
    llvm::errs() << "Error creating stderr temp file: " << ec.message() << "\n";
    return failure();
  }

  std::optional<StringRef> redirects[] = {/*stdin=*/std::nullopt,
                                          /*stdout=*/StringRef(tempFile),
                                          /*stderr=*/StringRef(stderrFile)};

  std::string errMsg;
  int result = sys::ExecuteAndWait(*program, args,
                                   /*Env=*/std::nullopt, redirects,
                                   /*secondsToWait=*/0,
                                   /*memoryLimit=*/0, &errMsg);

  // Read the captured stdout
  auto bufOrErr = MemoryBuffer::getFile(tempFile);
  if (bufOrErr) {
    output = (*bufOrErr)->getBuffer().str();
  }

  if (result != 0) {
    llvm::errs() << "Error running command: " << command[0] << "\n";
    // Print captured stderr on failure for diagnostics
    auto stderrBuf = MemoryBuffer::getFile(stderrFile);
    if (stderrBuf && !(*stderrBuf)->getBuffer().empty()) {
      llvm::errs() << (*stderrBuf)->getBuffer();
    }
  }

  // Clean up temp files
  sys::fs::remove(tempFile);
  sys::fs::remove(stderrFile);

  if (result != 0) {
    return failure();
  }
  return success();
}

/// Write a string to a file.
static LogicalResult writeFile(StringRef path, StringRef content) {
  std::error_code ec;
  raw_fd_ostream os(path, ec);
  if (ec) {
    llvm::errs() << "Error writing file " << path << ": " << ec.message()
                 << "\n";
    return failure();
  }
  os << content;
  return success();
}

/// Copy a file (cross-platform replacement for "cp").
static LogicalResult copyFile(StringRef src, StringRef dst) {
  std::error_code ec = sys::fs::copy_file(src, dst);
  if (ec) {
    llvm::errs() << "Error copying " << src << " to " << dst << ": "
                 << ec.message() << "\n";
    return failure();
  }
  return success();
}

/// Run a single pass (wrapped in builtin.module) and save debug IR.
static LogicalResult runSinglePass(StringRef passStr, ModuleOp moduleOp) {
  MLIRContext *ctx = moduleOp.getContext();
  std::string fullPipeline = "builtin.module(" + passStr.str() + ")";

  auto parsedPm = parsePassPipeline(fullPipeline);
  if (failed(parsedPm)) {
    llvm::errs() << "Error: failed to parse pass: " << passStr << "\n";
    return failure();
  }

  PassManager pm(ctx);
  pm.enableVerifier(true);
  static_cast<OpPassManager &>(pm) = std::move(*parsedPm);

  if (failed(pm.run(moduleOp))) {
    llvm::errs() << "Error: pass failed: " << passStr << "\n";
    return failure();
  }

  saveDebugIr(moduleOp, passStr);
  return success();
}

/// Run an MLIR pass pipeline string on a module.
/// The pipeline should be in the form "builtin.module(pass1,pass2,...)".
///
/// In debug mode (--debug-ir), splits the pipeline into individual passes
/// and saves IR after each one.
/// This produces files like pass_001_after_air-dependency.mlir,
/// pass_002_after_air-dma-to-channel.mlir, etc.
static LogicalResult runPassPipeline(StringRef pipeline, ModuleOp moduleOp) {
  MLIRContext *ctx = moduleOp.getContext();

  if (debugIr) {
    // Split the pipeline and run each pass individually, saving IR after each.
    auto passes = splitPipelineToPasses(pipeline);

    if (verbose) {
      llvm::outs() << "[DEBUG] Splitting pipeline into " << passes.size()
                   << " individual passes\n";
    }

    // Save initial IR before first pipeline (only once)
    if (passCounter == 0 && !debugIrDir.empty()) {
      SmallString<256> initFile(debugIrDir);
      sys::path::append(initFile, "pass_000_initial_input.mlir");
      if (failed(saveModule(moduleOp, initFile)))
        return failure();

      PassLogEntry entry;
      entry.index = 0;
      entry.passName = "[Initial IR before passes]";
      entry.outputFile = initFile.str().str();
      passLog.push_back(entry);

      if (verbose)
        llvm::outs() << "[PASS 000] Saved initial IR\n";
    }

    for (const auto &singlePass : passes) {
      if (failed(runSinglePass(singlePass, moduleOp)))
        return failure();
    }

    // Write the pass log after each pipeline
    dumpPassLog();
    return success();
  }

  // Non-debug mode: run the whole pipeline at once
  auto parsedPm = parsePassPipeline(pipeline);
  if (failed(parsedPm)) {
    llvm::errs() << "Error: failed to parse pass pipeline: " << pipeline
                 << "\n";
    return failure();
  }

  PassManager pm(ctx);
  pm.enableVerifier(true);
  static_cast<OpPassManager &>(pm) = std::move(*parsedPm);

  if (verbose) {
    std::string pipelineStr;
    raw_string_ostream pipelineOS(pipelineStr);
    pm.printAsTextualPipeline(pipelineOS);
    llvm::outs() << "Running pipeline: " << pipelineStr << "\n";
  }

  if (failed(pm.run(moduleOp))) {
    llvm::errs() << "Error: pass pipeline failed: " << pipeline << "\n";
    return failure();
  }

  return success();
}

/// Save a module's IR to a file.
static LogicalResult saveModule(ModuleOp moduleOp, StringRef path) {
  std::error_code ec;
  raw_fd_ostream os(path, ec);
  if (ec) {
    llvm::errs() << "Error writing module to " << path << ": " << ec.message()
                 << "\n";
    return failure();
  }
  moduleOp.print(os);
  return success();
}

/// Clone a module (by serializing and re-parsing).
static OwningOpRef<ModuleOp> cloneModule(ModuleOp moduleOp) {
  std::string irStr;
  raw_string_ostream os(irStr);
  moduleOp.print(os);

  MLIRContext *ctx = moduleOp.getContext();
  return parseSourceString<ModuleOp>(irStr, ctx);
}

//===----------------------------------------------------------------------===//
// GPU Compilation Pipeline
//===----------------------------------------------------------------------===//

static LogicalResult runGpuCompilation() {
  SmallString<256> baseName(sys::path::stem(inputFilename));

  // Find tools
  auto airOpt = sys::findProgramByName("air-opt");
  auto mlirOpt = sys::findProgramByName("mlir-opt");

  if (!airOpt) {
    llvm::errs() << "Error: could not find air-opt in PATH\n";
    return failure();
  }
  if (!mlirOpt) {
    llvm::errs() << "Error: could not find mlir-opt in PATH\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "GPU compilation for " << inputFilename << "\n";
    llvm::outs() << "  Architecture: " << gpuArch << "\n";
    llvm::outs() << "  Runtime: " << gpuRuntime << "\n";
    llvm::outs() << "  Tmpdir: " << tmpDir << "\n";
  }

  // Step 1: AIR to ROCDL
  SmallString<256> step1(tmpDir);
  sys::path::append(step1, baseName + "_step1_rocdl.mlir");
  if (failed(runCommand(
          {*airOpt, inputFilename, "-air-to-rocdl", "-o", step1.str().str()})))
    return failure();

  // Step 2: GPU Kernel Outlining (AIR-specific)
  SmallString<256> step2(tmpDir);
  sys::path::append(step2, baseName + "_step2_outlined.mlir");
  if (failed(runCommand({*airOpt, step1.str().str(), "-air-gpu-outlining", "-o",
                         step2.str().str()})))
    return failure();

  // Step 3: LLVM Lowering + GPU kernel outlining
  SmallString<256> step3(tmpDir);
  sys::path::append(step3, baseName + "_step3_gpu.mlir");
  std::string step3Pipeline =
      "--pass-pipeline=builtin.module(func.func(lower-affine,"
      "convert-linalg-to-loops,convert-scf-to-cf),gpu-kernel-outlining)";
  if (failed(runCommand({*mlirOpt, step2.str().str(), step3Pipeline, "-o",
                         step3.str().str()})))
    return failure();

  // Step 4: ROCDL Binary Generation
  std::string gpu4Pipeline =
      std::string("--pass-pipeline=builtin.module(") +
      "rocdl-attach-target{chip=" + gpuArch.getValue() + " O=3}," +
      "gpu.module(convert-gpu-to-rocdl{chipset=" + gpuArch.getValue() +
      " runtime=" + gpuRuntime.getValue() + "},reconcile-unrealized-casts)," +
      "gpu-module-to-binary," + "func.func(gpu-async-region)," +
      "gpu-to-llvm," + "convert-to-llvm," + "reconcile-unrealized-casts)";

  std::string finalOutput;
  if (!outputFilename.empty()) {
    finalOutput = outputFilename;
  } else {
    SmallString<256> tmp(tmpDir);
    sys::path::append(tmp, baseName + "_final.mlir");
    finalOutput = tmp.str().str();
  }

  if (failed(runCommand(
          {*mlirOpt, step3.str().str(), gpu4Pipeline, "-o", finalOutput})))
    return failure();

  if (verbose)
    llvm::outs() << "GPU compilation complete! Output: " << finalOutput << "\n";

  // Print to stdout if no output file specified (matches Python behavior)
  if (outputFilename.empty()) {
    auto bufOrErr = MemoryBuffer::getFile(finalOutput);
    if (bufOrErr)
      llvm::outs() << (*bufOrErr)->getBuffer();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AIE Compilation Pipeline (Core)
//===----------------------------------------------------------------------===//

/// Build the AIR optimization pass pipeline string.
static std::string buildOptimizationPipeline() {
  std::string pipeline;
  raw_string_ostream os(pipeline);

  os << "air-dependency,air-hoist-dma-in-accum-pattern";

  if (!omitAutoBroadcast) {
    os << ",air-broadcast-detection,air-specialize-dma-broadcast";
  }

  os << ",air-dma-to-channel,canonicalize,cse";
  os << ",air-dependency-canonicalize,canonicalize,cse";
  os << ",air-isolate-async-dma-loop-nests{scope=launch},canonicalize,cse";

  // Channel multiplexing
  if (!channelMultiplexing.empty()) {
    os << ",air-fuse-channels{aggressive-mode=";
    for (size_t i = 0; i < channelMultiplexing.size(); ++i) {
      if (i > 0)
        os << ",";
      os << channelMultiplexing[i];
    }
    os << "}";
  } else {
    os << ",air-fuse-channels";
  }

  os << ",canonicalize,cse";

  // L2 splitting (skip for npu_1col)
  if (deviceName.getValue().find("npu_1col") == std::string::npos) {
    os << ",func.func(air-split-l2-memref),canonicalize,cse";
    os << ",air-isolate-async-dma-loop-nests{scope=launch},canonicalize,cse";
  }

  // Loop fusion or alloc/dealloc optimization
  if (airLoopFusion) {
    os << ",func.func(air-loop-fusion)";
  } else {
    os << ",func.func(air-fuse-alloc-dealloc)";
    os << ",func.func(air-shrink-memref-sizes-by-access)";
  }

  // Ping-pong transform
  if (omitPingpong.getValue().empty() || omitPingpong.getValue() == "L1" ||
      omitPingpong.getValue() == "L2") {
    std::string labelPass = "air-label-scf-for-to-ping-pong";
    std::string ppPass = "air-ping-pong-transform";
    if (omitPingpong.getValue() == "L1" || omitPingpong.getValue() == "L2") {
      labelPass = "air-label-scf-for-to-ping-pong{omit-memory-space=" +
                  omitPingpong.getValue() + "}";
      ppPass = "air-ping-pong-transform{omit-memory-space=" +
               omitPingpong.getValue() + "}";
    }
    os << "," << labelPass << "," << ppPass << ",canonicalize,cse";
  }

  // Linalg lowering
  if (!lowerLinalgToFunc.empty()) {
    os << ",air-linalg-to-func{link-with=" << lowerLinalgToFunc.getValue()
       << "}";
  } else {
    os << ",func.func(convert-linalg-to-loops)";
  }

  os << ",func.func(air-opt-memtile-dma-bds{device=" << deviceName.getValue()
     << "}),canonicalize,cse";

  return pipeline;
}

/// Run the full AIE compilation pipeline.
static LogicalResult runAieCompilation() {
  SmallString<256> airMlirFilename(sys::path::filename(inputFilename));

  // Resolve default device parameters (-1 means unset / use default)
  int resolvedNumCols = numCols;
  if (resolvedNumCols < 0) {
    if (deviceName.getValue().find("npu1") != std::string::npos)
      resolvedNumCols = 4;
    else if (deviceName.getValue() == "npu2_4col")
      resolvedNumCols = 4;
    else if (deviceName.getValue().find("npu2") != std::string::npos)
      resolvedNumCols = 8;
    else
      resolvedNumCols = 10;
  }

  int resolvedColOffset = colOffset;
  if (resolvedColOffset < 0) {
    resolvedColOffset =
        (deviceName.getValue().find("npu") != std::string::npos) ? 0 : 7;
  }

  int resolvedNumRows = numRows >= 0 ? (int)numRows : 6;
  int resolvedRowOffset = rowOffset >= 0 ? (int)rowOffset : 2;

  if (verbose) {
    llvm::outs() << "compiling " << inputFilename << " for "
                 << deviceName.getValue() << "\n";
  }

  // Validate output format
  if (outputFormat == OF_elf &&
      deviceName.getValue().find("npu1") != std::string::npos) {
    llvm::errs() << "Error: output_format='elf' is not supported for "
                 << deviceName.getValue() << " target.\n";
    return failure();
  }

  // Setup debug IR directory
  if (debugIr) {
    SmallString<256> dir(tmpDir);
    sys::path::append(dir, "debug_ir");
    std::error_code ec = sys::fs::create_directories(dir);
    if (ec) {
      llvm::errs() << "Error creating debug_ir directory: " << ec.message()
                   << "\n";
      return failure();
    }
    debugIrDir = dir.str().str();
    if (verbose)
      llvm::outs() << "Debug mode enabled: saving IRs to " << debugIrDir
                   << "\n";
  }

  // Find aiecc
  auto aiecc = sys::findProgramByName("aiecc");
  if (!aiecc) {
    // Fallback for older mlir-aie
    aiecc = sys::findProgramByName("aiecc.py");
  }
  if (!aiecc) {
    llvm::errs() << "Error: could not find aiecc in PATH\n";
    return failure();
  }

  if (verbose)
    llvm::outs() << "Using aiecc from: " << *aiecc << "\n";

  // --- Set up MLIR context and parse input ---
  mlir::registerAllPasses();
  xilinx::air::registerAllPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  xilinx::air::registerAllDialects(registry);
#if AIR_ENABLE_AIE
  registry.insert<xilinx::AIE::AIEDialect>();
  registry.insert<xilinx::AIEX::AIEXDialect>();
#endif
  registerAllExtensions(registry);

  MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // Parse input file
  OwningOpRef<ModuleOp> inputModule;
  {
    auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (auto err = fileOrErr.getError()) {
      llvm::errs() << "Error reading file " << inputFilename << ": "
                   << err.message() << "\n";
      return failure();
    }

    SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
    inputModule = parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!inputModule) {
      llvm::errs() << "Error parsing MLIR file: " << inputFilename << "\n";
      return failure();
    }
  }

  ModuleOp moduleOp = inputModule.get();

  // Note: initial IR is saved by runPassPipeline() on first invocation
  // in debug mode.

  // --- Build and run the placement pipeline ---
  int traceColOffset = traceSize > 0 ? 1 : 0;

  std::string placementPipeline;
  {
    raw_string_ostream os(placementPipeline);
    os << "builtin.module(";
    os << "air-rank-to-launch";
    os << ",air-insert-launch-around-herd{insert-segment=true}";
    os << ",func.func(air-lower-herd-parallel)";
    os << ",scf-forall-to-parallel";

    // Add optimization passes for NPU targets
    if (deviceName.getValue().find("npu") != std::string::npos) {
      os << "," << buildOptimizationPipeline();
    }

    // Collapse herd and place
    os << ",func.func(air-collapse-herd{max-col-size=4})";
    os << ",canonicalize,cse";
    os << ",air-place-herds{";
    os << "num-rows=" << resolvedNumRows;
    os << " num-cols=" << resolvedNumCols;
    os << " row-anchor=" << resolvedRowOffset;
    os << " col-anchor=" << (resolvedColOffset + traceColOffset);
    os << "}";
    os << ",canonicalize,cse";
    os << ",func.func(air-renumber-dma)";
    os << ")";
  }

  // Clone module for placed version
  auto placedModule = cloneModule(moduleOp);
  if (!placedModule) {
    llvm::errs() << "Error: failed to clone module for placement\n";
    return failure();
  }

  SmallString<256> placedFile(tmpDir);
  sys::path::append(placedFile, "placed." + airMlirFilename);
  if (failed(runPassPipeline(placementPipeline, placedModule.get())))
    return failure();
  if (failed(saveModule(placedModule.get(), placedFile)))
    return failure();

  if (debugIr)
    addCheckpoint("AIR Placement Complete", "placed.air.mlir");

  // Split launch for non-tile-aligned DMA padding. No-op if no launch
  // has the air.actual_sizes attribute (set by air-wrap-func-with-parallel).
  if (failed(runPassPipeline("builtin.module(air-split-launch-for-padding{"
                             "pad-location=memtile})",
                             placedModule.get())))
    return failure();

  // --- AIR to AIE conversion ---
  std::string airToAiePipeline;
  {
    raw_string_ostream os(airToAiePipeline);
    os << "builtin.module(";
    os << "air-to-aie{";
    os << "emit-while-loop=" << (omitWhileTrueLoop ? "false" : "true");
    os << " row-offset=" << resolvedRowOffset;
    os << " col-offset=" << resolvedColOffset;
    os << " device=" << deviceName.getValue();
    if (traceSize > 0)
      os << " insert-trace-packet-flow=true";
    os << " use-lock-race-condition-fix="
       << (useLockRaceConditionFix ? "true" : "false");
    if (stackSize.getNumOccurrences() > 0)
      os << " stack-size=" << stackSize.getValue();
    os << "}";
    os << ",air-merge-unrolled-devices";
    os << ")";
  }

  auto aieModule = cloneModule(placedModule.get());
  if (!aieModule) {
    llvm::errs() << "Error: failed to clone module for AIR-to-AIE\n";
    return failure();
  }

  SmallString<256> aieFile(tmpDir);
  sys::path::append(aieFile, "aie." + airMlirFilename);
  if (failed(runPassPipeline(airToAiePipeline, aieModule.get())))
    return failure();
  if (failed(saveModule(aieModule.get(), aieFile)))
    return failure();

  if (debugIr)
    addCheckpoint("AIR to AIE Conversion Complete", "aie.air.mlir");

  // --- NPU path ---
  if (deviceName.getValue().find("npu") != std::string::npos) {
    // Build shim DMA BD optimization pass
    std::string shimBdPass;
    {
      raw_string_ostream os(shimBdPass);
      os << "func.func(air-opt-shim-dma-bds{device=" << deviceName.getValue();
      // Default tiling sizes [4, 4] unless overridden.
      // If flag was explicitly passed with no values, disable tiling.
      std::vector<unsigned> tilingSizes;
      if (!runtimeLoopTilingSizesPresent) {
        tilingSizes = {}; // Default: no tiling when flag not used
      } else {
        tilingSizes.assign(runtimeLoopTilingSizes.begin(),
                           runtimeLoopTilingSizes.end());
        // Flag present but empty = disable tiling (no sizes)
      }
      if (!tilingSizes.empty()) {
        os << " shim-dma-tile-sizes=";
        for (size_t i = 0; i < tilingSizes.size(); ++i) {
          if (i > 0)
            os << ",";
          os << tilingSizes[i];
        }
      }
      os << "})";
    }

    // Build airrt-to-npu pass
    std::string airrtToNpuPass;
    {
      raw_string_ostream os(airrtToNpuPass);
      os << "airrt-to-npu{";
      os << " trace-size=" << traceSize;
      os << " trace-offset=" << traceOffset;
      bool outputElf = (outputFormat == OF_elf);
      os << " output-elf=" << (outputElf ? "true" : "false");
      os << "}";
    }

    std::string npuPipeline;
    {
      raw_string_ostream os(npuPipeline);
      os << "builtin.module(";
      os << shimBdPass;
      os << ",canonicalize,cse";
      os << ",air-to-std";
      os << ",symbol-dce";
      os << ",affine-expand-index-ops";
      os << ",canonicalize,cse";
      os << "," << airrtToNpuPass;
      // symbol-dce: drop module-level globals orphaned by mmio lowering
      // before aiecc promotes them to llvm.mlir.global at module scope.
      os << ",symbol-dce";
      os << ",canonicalize,cse";
      os << ")";
    }

    auto npuModule = cloneModule(aieModule.get());
    if (!npuModule) {
      llvm::errs() << "Error: failed to clone module for NPU lowering\n";
      return failure();
    }

    SmallString<256> npuFile(tmpDir);
    sys::path::append(npuFile, "npu." + airMlirFilename);
    if (failed(runPassPipeline(npuPipeline, npuModule.get())))
      return failure();
    if (failed(saveModule(npuModule.get(), npuFile)))
      return failure();

    if (debugIr) {
      addCheckpoint("NPU Instruction Generation Complete", "npu.air.mlir");
      dumpPassLog();
      if (verbose)
        llvm::outs() << "Pass log written to: " << debugIrDir << "/pass.log\n";
    }

    // --- Invoke aiecc for backend compilation ---
    std::string xclbinFile;
    if (!outputFilename.empty()) {
      xclbinFile = outputFilename.getValue();
    } else {
      xclbinFile = "aie.xclbin";
    }

    std::string instsFile;
    if (!instsFilename.empty()) {
      instsFile = instsFilename.getValue();
    } else {
      // Derive from xclbin name
      instsFile = xclbinFile;
      StringRef ref(instsFile);
      if (ref.ends_with(".xclbin"))
        instsFile = ref.drop_back(7).str() + ".insts.bin";
    }

    // Build aiecc command
    std::vector<std::string> aieccCmd;
    aieccCmd.push_back(*aiecc);

    if (verbose)
      aieccCmd.push_back("-v");

    aieccCmd.push_back("--no-aiesim");
    aieccCmd.push_back(xchesscc ? "--xchesscc" : "--no-xchesscc");
    aieccCmd.push_back(xbridge ? "--xbridge" : "--no-xbridge");
    aieccCmd.push_back("--no-compile-host");
    aieccCmd.push_back("--tmpdir=" + tmpDir.getValue());

    // Output format options
    if (outputFormat == OF_elf) {
      aieccCmd.push_back("--generate-full-elf");
      aieccCmd.push_back("--expand-load-pdis");
      aieccCmd.push_back("--full-elf-name=" + elfName.getValue());
    } else if (outputFormat == OF_xclbin) {
      aieccCmd.push_back("--aie-generate-xclbin");
      aieccCmd.push_back("--xclbin-name=" + xclbinFile);
    } else if (outputFormat == OF_txn) {
      aieccCmd.push_back("--aie-generate-txn");
    }
    // OF_none = no output generation options

    // NPU instruction generation (not for ELF mode)
    if (outputFormat != OF_elf) {
      aieccCmd.push_back("--aie-generate-npu-insts");
      aieccCmd.push_back("--npu-insts-name=" + instsFile);
    }

    // Xclbin metadata (not for ELF mode)
    if (outputFormat != OF_elf) {
      if (!kernelName.empty())
        aieccCmd.push_back("--xclbin-kernel-name=" + kernelName.getValue());
      if (!instanceName.empty())
        aieccCmd.push_back("--xclbin-instance-name=" + instanceName.getValue());
      if (!kernelId.empty())
        aieccCmd.push_back("--xclbin-kernel-id=" + kernelId.getValue());
      if (!xclbinInput.empty())
        aieccCmd.push_back("--xclbin-input=" + xclbinInput.getValue());
    }

    // Peano — use --peano=<dir> (equals-joined) so that downstream aiecc
    // never risks consuming the next argument as the peano value.
    if (!peanoInstallDir.empty()) {
      aieccCmd.push_back("--peano=" + peanoInstallDir.getValue());
    }

    aieccCmd.push_back("-O");
    aieccCmd.push_back("3");

    // bf16 emulation
    if (bf16Emulation)
      aieccCmd.push_back("--bf16-emulation");

    // Input file
    aieccCmd.push_back(npuFile.str().str());

    if (verbose) {
      llvm::outs() << "Running aiecc with options: ";
      for (const auto &arg : aieccCmd) {
        llvm::outs() << arg << " ";
      }
      llvm::outs() << "\n";
    }

    if (failed(runCommand(aieccCmd)))
      return failure();

  } else {
    // --- Non-NPU path (Versal/legacy) ---
    // This path generates host-side libraries using aiecc + clang

    // Run air-split-devices
    std::string splitPipeline =
        "builtin.module(air-split-devices{output-prefix=" + tmpDir.getValue() +
        "/})";
    auto splitModule = cloneModule(aieModule.get());
    if (!splitModule) {
      llvm::errs() << "Error: failed to clone module for device split\n";
      return failure();
    }
    if (failed(runPassPipeline(splitPipeline, splitModule.get())))
      return failure();

    // Lower AIRRt to host code
    auto airrtModule = cloneModule(splitModule.get());
    if (!airrtModule) {
      llvm::errs() << "Error: failed to clone module for AIRRt lowering\n";
      return failure();
    }

    SmallString<256> aieCtrlAirrt(tmpDir);
    sys::path::append(aieCtrlAirrt, "airrt." + airMlirFilename);
    {
      std::string pipeline;
      raw_string_ostream os(pipeline);
      os << "builtin.module(";
      os << "convert-vector-to-llvm,convert-math-to-llvm";
      os << ",func.func(air-label-broadcast-channel-with-tile)";
      os << ",lower-affine";
      os << ",func.func(air-opt-shim-dma-bds{device=" << deviceName.getValue()
         << "})";
      os << ",air-to-std,air-lower-linalg-tensors";
      os << ",canonicalize,cse";
      os << ")";
      if (failed(runPassPipeline(pipeline, airrtModule.get())))
        return failure();
      if (failed(saveModule(airrtModule.get(), aieCtrlAirrt)))
        return failure();
    }

    // Lower to LLVM (airrt-to-llvm + bufferize)
    SmallString<256> aieCtrlFile(tmpDir);
    sys::path::append(aieCtrlFile, "aie_ctrl." + airMlirFilename);
    {
      std::string pipeline;
      raw_string_ostream os(pipeline);
      os << "builtin.module(airrt-to-llvm,one-shot-bufferize)";
      if (failed(runPassPipeline(pipeline, airrtModule.get())))
        return failure();
      if (failed(saveModule(airrtModule.get(), aieCtrlFile)))
        return failure();
    }

    // Generate refback (reference backend) by running the full AIRRt→LLVM
    // pipeline on a fresh clone of the placed module.
    {
      auto refbackModule = cloneModule(placedModule.get());
      if (refbackModule) {
        SmallString<256> refbackFile(tmpDir);
        sys::path::append(refbackFile, "refback." + airMlirFilename);
        std::string pipeline;
        raw_string_ostream os(pipeline);
        os << "builtin.module(";
        os << "convert-vector-to-llvm,convert-math-to-llvm";
        os << ",func.func(air-label-broadcast-channel-with-tile)";
        os << ",lower-affine";
        os << ",func.func(air-opt-shim-dma-bds{device=" << deviceName.getValue()
           << "})";
        os << ",air-to-std,air-lower-linalg-tensors";
        os << ",canonicalize,cse";
        os << ",airrt-to-llvm";
        os << ",canonicalize,cse";
        os << ")";
        if (succeeded(runPassPipeline(pipeline, refbackModule.get())))
          (void)saveModule(refbackModule.get(), refbackFile);
        // Failure is non-fatal — refback is a debugging artifact
      }
    }

    SmallString<256> aieCtrlLlvm(tmpDir);
    sys::path::append(aieCtrlLlvm, "llvm." + airMlirFilename);
    {
      std::string pipeline;
      raw_string_ostream os(pipeline);
      os << "builtin.module(";
      os << "expand-strided-metadata,lower-affine,convert-scf-to-cf";
      os << ",finalize-memref-to-llvm,convert-func-to-llvm";
      os << ",convert-arith-to-llvm,convert-cf-to-llvm";
      os << ",canonicalize,cse";
      os << ")";
      if (failed(runPassPipeline(pipeline, airrtModule.get())))
        return failure();
      if (failed(saveModule(airrtModule.get(), aieCtrlLlvm)))
        return failure();
    }

    // Translate to LLVM IR
    SmallString<256> llvmIr(tmpDir);
    sys::path::append(llvmIr, airMlirFilename.str() + ".ll");
    if (failed(runCommand({"aie-translate", "--mlir-to-llvmir",
                           aieCtrlLlvm.str().str(), "-o", llvmIr.str().str()})))
      return failure();

    // Optimize LLVM IR
    SmallString<256> optBc(tmpDir);
    sys::path::append(optBc, airMlirFilename.str() + ".opt.bc");
    if (failed(runCommand(
            {"opt", "-O3", llvmIr.str().str(), "-o", optBc.str().str()})))
      return failure();

    SmallString<256> optIr(tmpDir);
    sys::path::append(optIr, airMlirFilename.str() + ".opt.ll");
    if (failed(runCommand(
            {"llvm-dis", optBc.str().str(), "-o", optIr.str().str()})))
      return failure();

    // Compile to object file
    SmallString<256> objFile(tmpDir);
    sys::path::append(objFile, airMlirFilename.str() + ".o");

    std::vector<std::string> llcCmd = {"llc", "-O3", "--filetype=obj",
                                       "--relocation-model=pic"};
    if (!hostTarget.empty()) {
      StringRef ht = hostTarget.getValue();
      if (ht.contains("x86_64"))
        llcCmd.push_back("-march=x86-64");
      else if (ht.contains("aarch64"))
        llcCmd.push_back("-march=aarch64");
    }
    llcCmd.push_back(optIr.str().str());
    llcCmd.push_back("-o");
    llcCmd.push_back(objFile.str().str());
    if (failed(runCommand(llcCmd)))
      return failure();

    // Get segment metadata via air-translate
    std::string jsonOutput;
    if (failed(
            runCommandCaptureOutput({"air-translate", "--airrt-generate-json",
                                     aieCtrlAirrt.str().str()},
                                    jsonOutput)))
      return failure();

    // Parse JSON to extract segment names
    // The Python version uses eval() on the JSON - we parse it manually
    // For now, this is a simplified version. A full implementation would
    // use llvm::json::parse.
    auto jsonVal = json::parse(jsonOutput);
    if (!jsonVal) {
      llvm::errs() << "Error parsing JSON from air-translate\n";
      return failure();
    }

    std::vector<std::string> segments;
    if (auto *obj = jsonVal->getAsObject()) {
      for (auto &kv : *obj) {
        if (auto *segObj = kv.second.getAsObject()) {
          if (auto symName = segObj->getString("sym_name")) {
            segments.push_back(symName->str());
          }
        }
      }
    }

    // Compile each segment with aiecc
    std::vector<std::string> allObjFiles = {objFile.str().str()};

    for (const auto &segment : segments) {
      if (verbose)
        llvm::outs() << "Compiling segment: " << segment << "\n";

      SmallString<256> segmentFile(tmpDir);
      sys::path::append(segmentFile, "aie." + segment + ".mlir");

      SmallString<256> aieccFile(tmpDir);
      sys::path::append(aieccFile, "aiecc." + segment + ".mlir");

      SmallString<256> aieccDir(tmpDir);
      sys::path::append(aieccDir, segment);

      // Lower segment
      if (failed(runCommand({"air-opt", segmentFile.str().str(),
                             "-air-lower-linalg-tensors", "-lower-affine",
                             "-canonicalize", "-cse", "-o",
                             aieccFile.str().str()})))
        return failure();

      // Determine host target
      std::string aieccTarget;
      if (!hostTarget.empty()) {
        aieccTarget = hostTarget.getValue();
      } else {
#if defined(__x86_64__) || defined(_M_X64)
        aieccTarget = "x86_64-amd-linux-gnu";
#elif defined(__aarch64__) || defined(_M_ARM64)
        aieccTarget = "aarch64-linux-gnu";
#else
        aieccTarget = "x86_64-amd-linux-gnu";
#endif
      }

      // Run aiecc on segment
      std::string sysrootVal = sysroot.empty() ? "/" : sysroot.getValue();
      std::vector<std::string> segAieccCmd;
      segAieccCmd.push_back(*aiecc);
      if (verbose)
        segAieccCmd.push_back("-v");
      segAieccCmd.push_back("--sysroot");
      segAieccCmd.push_back(sysrootVal);
      segAieccCmd.push_back("--host-target");
      segAieccCmd.push_back(aieccTarget);
      segAieccCmd.push_back("--tmpdir");
      segAieccCmd.push_back(aieccDir.str().str());
      segAieccCmd.push_back("--no-aiesim");
      segAieccCmd.push_back("--compile-host");
      segAieccCmd.push_back(xchesscc ? "--xchesscc" : "--no-xchesscc");
      segAieccCmd.push_back(xbridge ? "--xbridge" : "--no-xbridge");
      segAieccCmd.push_back(aieccFile.str().str());

      if (failed(runCommand(segAieccCmd)))
        return failure();

      // Copy and compile wrapper
      SmallString<256> incFile(tmpDir);
      sys::path::append(incFile,
                        airMlirFilename.str() + "." + segment + ".inc");

      SmallString<256> srcIncFile(aieccDir);
      sys::path::append(srcIncFile, "aie_inc.cpp");
      if (failed(copyFile(srcIncFile, incFile)))
        return failure();

      // Generate wrapper cpp
      SmallString<256> cppFile(tmpDir);
      sys::path::append(cppFile,
                        airMlirFilename.str() + "." + segment + ".cpp");

      SmallString<256> segObjFile(tmpDir);
      sys::path::append(segObjFile,
                        airMlirFilename.str() + "." + segment + ".o");

      {
        std::string wrapper;
        raw_string_ostream ws(wrapper);
        ws << "// generated by aircc, do not edit\n";
        ws << "#include \"stdio.h\"\n";
        ws << "#include \"assert.h\"\n";
        ws << "#include \"air_host.h\"\n";
        ws << "#include \"air_host_impl.h\"\n\n";
        ws << "namespace air {\nnamespace segments {\n";
        ws << "namespace " << segment << " {\n";
        ws << "#include \"" << incFile.str() << "\"\n";
        ws << "}\n}\n}\n\n";
        ws << "using namespace air::segments::" << segment << ";\n";
        ws << "extern \"C\" {\n";
        ws << "air_rt_aie_functions_t __airrt_" << segment
           << "_aie_functions {\n";
        ws << "  .configure_cores = &mlir_aie_configure_cores,\n";
        ws << "  .configure_switchboxes = &mlir_aie_configure_switchboxes,\n";
        ws << "  .initialize_locks = &mlir_aie_initialize_locks,\n";
        ws << "  .configure_dmas = &mlir_aie_configure_dmas,\n";
        ws << "  .start_cores = &mlir_aie_start_cores\n";
        ws << "};\n}\n";
        if (failed(writeFile(cppFile, wrapper)))
          return failure();
      }

      // Compile wrapper
      std::vector<std::string> compileCmd;
      compileCmd.push_back(cc.getValue());
      compileCmd.push_back("-std=c++17");
      compileCmd.push_back("-g");
      compileCmd.push_back("-I.");

      if (!sysroot.empty()) {
        compileCmd.push_back("--sysroot=" + sysroot.getValue());
        if (StringRef(aieccTarget).contains("aarch64-linux-gnu"))
          compileCmd.push_back("--gcc-toolchain=" + sysroot.getValue() +
                               "/usr");
      }
      if (!hostTarget.empty())
        compileCmd.push_back("--target=" + hostTarget.getValue());

      // Find include paths relative to the aircc/aiecc executable directory.
      // This mirrors the Python driver's include path setup.
      SmallString<256> exePath(sys::fs::getMainExecutable(nullptr, nullptr));
      sys::path::remove_filename(exePath);

      // AIR host runtime include
      SmallString<256> airHostInclude(exePath);
      sys::path::append(airHostInclude, "..", "runtime_lib", "airhost",
                        "include");
      compileCmd.push_back("-I" + airHostInclude.str().str());

      // aiecc runtime test_lib includes (architecture-specific)
      SmallString<256> aieccPath(*aiecc);
      sys::path::remove_filename(aieccPath);
      if (StringRef(aieccTarget).contains("x86_64")) {
        SmallString<256> testLibInc(aieccPath);
        sys::path::append(testLibInc, "..");
        sys::path::append(testLibInc, "runtime_lib", "x86_64");
        sys::path::append(testLibInc, "test_lib", "include");
        compileCmd.push_back("-I" + testLibInc.str().str());
      }
      if (StringRef(aieccTarget).contains("aarch64")) {
        SmallString<256> testLibInc(aieccPath);
        sys::path::append(testLibInc, "..");
        sys::path::append(testLibInc, "runtime_lib", "aarch64");
        sys::path::append(testLibInc, "test_lib", "include");
        compileCmd.push_back("-I" + testLibInc.str().str());
      }

      // libxaie include (from LIBXAIE_DIR env var if set)
      if (auto libxaiePath = sys::Process::GetEnv("LIBXAIE_DIR")) {
        compileCmd.push_back("-I" + *libxaiePath + "/include");
      }

      // ROCm/HSA include (from ROCM_PATH env var if set)
      if (auto rocmPath = sys::Process::GetEnv("ROCM_PATH")) {
        SmallString<256> hsaInc(*rocmPath);
        sys::path::append(hsaInc, "..", "..", "..", "include");
        compileCmd.push_back("-I" + hsaInc.str().str());
      }

      compileCmd.push_back("-DLIBXAIENGINEV2");
      compileCmd.push_back("-DAIE_LIBXAIE_ENABLE");
      compileCmd.push_back("-fPIC");
      compileCmd.push_back("-c");
      compileCmd.push_back("-o");
      compileCmd.push_back(segObjFile.str().str());
      compileCmd.push_back(cppFile.str().str());

      if (failed(runCommand(compileCmd)))
        return failure();

      allObjFiles.push_back(segObjFile.str().str());
    }

    // Link all object files
    std::string libExt;
#ifdef _WIN32
    libExt = shared ? ".dll" : ".lib";
#else
    libExt = shared ? ".so" : ".a";
#endif

    SmallString<256> libFile(tmpDir);
    sys::path::append(libFile, airMlirFilename.str() + libExt);

    if (shared) {
      std::vector<std::string> linkCmd = {cc.getValue(), "-shared"};
      if (!sysroot.empty()) {
        linkCmd.push_back("--sysroot");
        linkCmd.push_back(sysroot.getValue());
      }
      if (!hostTarget.empty()) {
        linkCmd.push_back("-target");
        linkCmd.push_back(hostTarget.getValue());
      }
      linkCmd.push_back("-fuse-ld=lld");
      linkCmd.push_back("-o");
      linkCmd.push_back(libFile.str().str());
      for (const auto &obj : allObjFiles)
        linkCmd.push_back(obj);
      if (failed(runCommand(linkCmd)))
        return failure();
    } else {
      std::vector<std::string> arCmd = {"llvm-ar", "rc", libFile.str().str()};
      for (const auto &obj : allObjFiles)
        arCmd.push_back(obj);
      if (failed(runCommand(arCmd)))
        return failure();
    }

    // Copy to output file
    if (!outputFilename.empty()) {
      if (failed(copyFile(libFile, outputFilename.getValue())))
        return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(argc, argv,
                              "AIR Compiler Driver\n"
                              "  Compiles AIR MLIR programs for "
                              "AIE/NPU devices.\n");

  // Create tmpdir
  std::error_code ec = sys::fs::create_directory(tmpDir);
  if (ec && ec != std::errc::file_exists) {
    llvm::errs() << "Error creating directory " << tmpDir << ": "
                 << ec.message() << "\n";
    return 1;
  }

  // Resolve --peano: fall back to PEANO_INSTALL_DIR env var when --peano was
  // not specified on the command line. This matches the Python backend behavior
  // (e.g. python/air/backend/xrt.py) where callers read the env var and pass
  // it via --peano. Supporting the env var directly lets callers omit --peano
  // entirely when the value may be empty. Note that if a shell invocation
  // expands to `--peano input.mlir` (e.g. `--peano $EMPTY_VAR input.mlir`
  // with EMPTY_VAR unset), cl::ParseCommandLineOptions will consume the input
  // path as the option value before this fallback runs. To avoid this, callers
  // should either omit --peano entirely or use `--peano=$VAR`.
  //
  // Gate on getNumOccurrences() so that explicit `--peano=` (intentionally
  // empty) is not silently overridden by the env var.
  if (peanoInstallDir.getNumOccurrences() == 0) {
    if (auto envPeano = sys::Process::GetEnv("PEANO_INSTALL_DIR"))
      peanoInstallDir = *envPeano;
  }

  // Resolve conflicting options
  if (noXbridge)
    xbridge = false;
  if (noXchesscc)
    xchesscc = false;

  // Handle --omit-ping-pong-transform with no value (ValueOptional).
  // When the flag is present but no value given, cl::ValueOptional sets the
  // string to "". Map this to "all" for backward compatibility with Python
  // argparse nargs="?" const="all".
  if (omitPingpong.getNumOccurrences() > 0 && omitPingpong.getValue().empty())
    omitPingpong.setValue("all");

  // Track whether --air-runtime-loop-tiling-sizes was explicitly passed
  runtimeLoopTilingSizesPresent =
      runtimeLoopTilingSizes.getNumOccurrences() > 0;

  // Dispatch based on target
  if (target.getValue() == "gpu") {
    return failed(runGpuCompilation()) ? 1 : 0;
  } else {
    return failed(runAieCompilation()) ? 1 : 0;
  }
}

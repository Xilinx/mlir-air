# Current AIR source interfaces

The optional `tools/air_source_profile.py` reads source with Python AST and records hashes for the supported shared interfaces. It imports no AIR builders and performs no compilation, model loading or device execution.

```bash
python agentic/tools/air_source_profile.py
```

`INTERFACES_PRESENT` means the named source symbols exist. It does not certify their signatures, numerical behavior, toolchain installation or runtime compatibility. `INTEGRATION_REVIEW_REQUIRED` identifies a missing or changed interface; inspect that dependency while continuing independent work. This is an engineering diagnostic, not a mandatory startup gate.

The integration uses the shared affine-Q4 deployment interfaces merged by
PR #2198. It was rebased and host-checked against AIR main `ce323dbb`, which pins
MLIR-AIE `1.4.5.dev83` and LLVM-AIE `2026102001`. Follow the checkout's build guide
for supported toolchain installation and record the actual checkout/tools in
each task. Interface hashes identify observed bytes; they allow source changes
to be inspected without a fixed-version startup gate.

The inspected interfaces and their roles are described in [project integration](../references/project-integration.md). Record a fresh source profile when adapting the harness to a different checkout.

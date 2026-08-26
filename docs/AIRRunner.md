# AIR Performance Modeling

`air-runner` is a performance simulator which models the concurrent execution of an MLIR-AIR program.

## Usage

### Command line

```
USAGE: air-runner [options] <input file>

OPTIONS:

Color Options:

  --color                            - Use colors in output (default=autodetect)

General options:

  --disable-i2p-p2i-opt              - Disables inttoptr/ptrtoint roundtrip optimization
  --experimental-assignment-tracking -
  -f <function>                      - top-level function name
  -m <filename>                      - json model filename
  -o <filename>                      - Output filename
  --opaque-pointers                  - Use opaque pointers
  -v                                 - verbose

Generic Options:

  --help                             - Display available options (--help-hidden for more)
  --help-list                        - Display list of available options (--help-list-hidden for more)
  --version                          - Display the version of this program
```

### Python

```
import air.compiler.util

# arch is a json object which describes the target AIE device's resource model
runner = air.compiler.util.Runner(arch)
trace = runner.run(air_module, "your_air_module_name")
```

## How compute is costed

An op's cycle count comes from the arch model, three ways.

**A rate.** The default for a `linalg` body: its scalar arithmetic is counted
and divided by a per-datatype rate.

```json
"kernels": {
  "linalg.matmul": {
    "name": "linalg.matmul",
    "datatypes": { "bf16": { "macs_per_core_per_cycle": 128, "efficiency": 1 } }
  }
}
```

**A constant**, for an `air.custom` — a placeholder for work the IR does not
spell out.

```json
"custom_kernels": { "nn": { "datatypes": { "i8": { "latency": 10480 } } } }
```

**An expression**, when the cost is neither. A weight-stationary block that
streams an activation through fixed weights costs bit-planes per weight tile
and does not care about the nominal MAC count; a rate cannot say that.

```json
"datatypes": { "i8": { "cycles": "ceildiv(volume1, 4096) * (4 + 12*bits1)" } }
```

An expression replaces the whole formula, including
`kernel_invocation_overhead`. It works in both `kernels` and `custom_kernels`.
Available: `+ - * / %`, parentheses, and `ceil`, `floor`, `min`, `max`,
`ceildiv`. Variables, where `N` is an operand number and unsuffixed names alias
operand 0:

| | |
|---|---|
| `ops` | scalar arithmetic the rate model would have counted |
| `iters` | the `linalg` iteration space, 0 for an op with no body |
| `volumeN` | elements in operand N |
| `bitsN` | element bit width of operand N |
| `bytesN` | bytes in operand N |
| `rankN` | rank of operand N |
| `dN_K` | extent of dimension K of operand N |

An expression that cannot be evaluated is an error naming the op and quoting
the expression; the run then reports no latency.

## Machine parameters

How a `linalg` body is priced was once fixed in the runner and chosen for AIE.
These override it; the defaults are the historical values.

```json
"compute_model": {
  "cores_per_kernel_instance": 1,
  "kernel_invocation_overhead": 100,
  "default_ops_per_core_per_cycle": 8
}
```

`kernel_invocation_overhead` is the cost of entering a kernel, which on AIE is
a call to an external function. On a machine where the operator *is* the
instruction it is zero — and it is not merely a tuning value, since a target
whose whole GEMV is 70 cycles cannot be described while a 100-cycle floor
applies.

## What the runner will not price

Scalar ops in a `linalg` body that the rate model does not know are counted as
free, and reported as such. `math.erf` is not `arith.addf`, and a single rate
for a whole body cannot distinguish them — use a `cycles` expression where the
difference matters.

## Time trace user interface

`air-runner` returns the simulated time traces for the MLIR-AIR program as a json file, formatted to be visualized using [Chrome Tracing](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/).

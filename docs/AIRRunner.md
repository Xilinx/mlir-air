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

## Time trace user interface

`air-runner` returns the simulated time traces for the MLIR-AIR program as a json file, formatted to be visualized using [Chrome Tracing](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/).

# AIR

This repository contains tools and libraries for building AIR platforms,
runtimes and compilers.

Basic repository layout:

```
air
├── cmake                     CMake files
├── docs                      Documentation
├── examples                  Example code
├── mlir                      MLIR dialects and passes
├── platforms                 Hardware platforms
│   ├── xilinx_vck190_air     Platform files for vck190 development board
│   └── xilinx_vck5000_air    Platform files for vck5000 PCIe card
├── pynq                      Board repo for building Pynq images
│   └── vck190_air
├── python                    Python libraries and bindings
├── runtime_lib               Runtime libraries for host and controllers
├── test                      In hardware tests of AIR components
├── tools                     aircc.py, air-opt, air-translate
└── utils                     Utility scripts
```

## Documentation

### Getting Started

- [Building](building.md)
- [Running Examples](running.md)
- [AIRCC](aircc.md)

### Generated MLIR Documentation
- [AIR Dialect](AIRDialect.html)
- [AIRRt Dialect](AIRRtDialect.html)
- [AIR Transform Passes](AIRTransformPasses.html)
- [AIR Conversion Passes](AIRConversionPasses.html)
- [Transform Dialect Extensions](AIRTransformOps.html)

### AIR Platform Documentation
- [VCK5000 platform](https://github.com/Xilinx/ROCm-air-platforms) 

### Tutorials
- [AIR Asynchronous Concurrency](AIRAsyncConcurrency.md)
- [AIR Runner](AIRRunner.md)

-----

<p align="center">Copyright&copy; 2019-2023 Advanced Micro Devices, Inc.</p>

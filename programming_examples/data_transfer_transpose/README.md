# Data Transfer Transpose

Transposes a matrix with using either Channels or `dma_memcpy_nd`.

#### Usage (For Both Examples)

To generate AIR MLIR from Python:
```bash
cd <example_dir>
make clean && make print
```

To run:
```bash
cd <example_dir>
make clean && make
```

To run with verbose output:
```bash
cd <example_dir>
python transpose.py -v
```

You can also change some other parameters; to get usage information, run:
```bash
cd <example_dir>
python transpose.py -h
```

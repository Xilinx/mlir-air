# Matrix Scalar Addition

This example focuses on a core concept: processing input as a grid of smaller inputs. In this case, each implementation of the matrix scalar addition program breaks a 2-dimensional matrix of input data (the *image*) into smaller 2-dimensional regions (the *tiles*), and then increments every value in each tile with a constant specific to that tile.

There are several versions of this example that use DMA memcopies, but there are also some versions of this example that use *channels*, the primary abstraction used to represent data movement provided by the mlir-air python bindings. In this example, there is an input channel (`ChanIn`) and an output channel (`ChanOut`). The data is moved into/out of channels to/from the arguments in the mlir-air *launch*; the data is then retrieved from the input channel, processed, and written back to the output channel at the *herd* level.

## Running and Testing

For illustrative purposes, there are several versions of this example: ```single_core_dma```, ```multi_core_dma```, ```single_core_channel```, ```multi_core_channel```, and ```multi_launch_channel```. Note that ```multi_launch_channel``` is a WIP and is not functional as multiple launches are not yet supported.

#### ```single-core-dma```: Tiling using DMA sizes, offsets, and strides

This example ([single_core_dma/single_core_dma.py](single_core_dma/single_core_dma.py)) uses *sizes*, *offsets*, and *strides* to explicitly loop *n* total tiles, and then fetch, increment, and put one tile at a time. The entirety of the work is done by one launch which manages one segment which manages one herd which manages one core.

#### ```multi-core-dma```: Tiling using DMA sizes, offsets, and strides with multiple compute cores

This example ([multi_core_dma/single_core_dma.py](multi_core_dma/multi_core_dma.py)) uses *sizes*, *offsets*, and *strides*. Unlike the `single-core-dma` example, this example uses a herd size that maps to the 2-dimensional number of tiles in the image. No explicit loop is needed to processes each tile as each compute tile will process exactly one tile of data. The entirety of the work is done by one launch which manages one segment which manages one herd which manages *n* cores, where *n* is the number of data tiles.

#### ```single-core-channel```: Tiling using Channel sizes, offsets, and strides

This example ([single_core_dma/single_core_dma.py](single_core_dma/single_core_dma.py)) uses *sizes*, *offsets*, and *strides* to explicitly loop *n* total tiles, and then fetch, increment, and put one tile at a time. L3 data must be written to/from channels at the launch level, so the launch transforms the sequential image data to/from a sequence of sequential tile data using a series of specially constructed `ChannelPut` and `ChannelGet` operations. The compute core is then able to access each data tile by tile using simple `ChannelPut` and `ChannelGet` operations.

#### ```multi-core-channel```: Tiling using Channel sizes, offsets, and strides with multiple compute cores

This example ([multi_core_dma/single_core_dma.py](multi_core_dma/multi_core_dma.py)) uses *sizes*, *offsets*, and *strides* to explicitly loop *n* total tiles, and then fetch, increment, and put one tile at a time. L3 data must be written to/from channels at the launch level, so the launch transforms the sequential image data to/from a sequence of sequential tile data using a series of specially constructed `ChannelPut` and `ChannelGet` operations. Unlike the `single-core-channel` example, this example uses a herd size that maps to the 2-dimensional number of tiles in the image. No explicit loop is needed to processes each tile as each compute tile will process exactly one tile of data. The entirety of the work is done by one launch which manages one segment which manages one herd which manages *n* cores, where *n* is the number of data tiles.

#### [WIP] ```multi-launch-channel```: This example is under construction

This example ([multi_launch_channel/multi_launch_channel.py](multi_launch_channel/multi_launch_channel.py)) uses multiple launches. It is currently a work in progress as multiple launches are not yet fully supported.

#### Usage (For All Examples)

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
python <example_file>.py -v
```

You can also change some other parameters; to get usage information, run:
```bash
cd <example_dir>
python <example_file>.py -h
```

## Recommended Exercises
* Generate the mlir from any example and compare the differences between the generated mlir of each example:
  ```bash
  python (single|multi)_(core|launch)_(dma|channel).py
  ```
* Pick one of the examples and modify it so only the first value in each tile is incremented
* Pick one of the examples and modify it so all data in all tiles is incremented by the same constant value

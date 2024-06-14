# Matrix Scalar Addition

This example focuses on a core concept: processing input as a grid of smaller inputs. In this case, each implementation of the matrix scalar addition program increments every value in a 2-dimensional matrix. To do this, the program divides the input *image* into smaller *tiles* and increments each value in each tile.

This example also introduces the concept of AIR *channels*, the abstraction used to represent data movement in the mlir-air python bindings. In this example, there is an input channel and an output channel. The entire image is written to/read from the respective channel by the test harness ([run.py](run.py)). In contrast, the process on the NPU only reads/writes one tiles worth of data at a time from the respective channel.

Each version of this program uses a different mechanism to decide how to divide the work of handling all the tiles spatially and temporally.
The different ways of dividing up the tiled work largely consists of using controls exposed by different mlir-air abstractions.

The fundamental abstractions of mlir-air, and the ones used in this example, are:
* A *launch(es)*, which manage:
  * Two channels (an input channel and an output channel)
  * One or more segments
* The *segment(s)* manage:
  * One or more herds
* The *herd(s)*, which consist of 
  * One or more compute cores

## Running and Testing

For illustrative purposes, we provide four different ways specify data tiling in this example: ```channel```, ```herd```, ```segment```, and ```launch```.

#### ```channel```: Tiling using channel sizes, offsets, and strides

This example ([matrix_scalar_add_channel.py](matrix_scalar_add_channel.py)) uses channel *sizes*, *offsets*, and *strides* to explicitly loop over the image worth of data to read, increment, and write to one tile at a time. The entirety of the work is done by one launch which manages one segment which manages one herd which manages one core. This is the only example that explicitly uses only one core to do all of the processing.
```bash
make clean && make channel
```

#### ```herd```: Tiling using a herd managing multiple cores

This example ([matrix_scalar_add_herd.py](matrix_scalar_add_herd.py)) uses herds and the internal logic of channels to process all the tiles. Here, there is one launch and one segment and one herd, but a herd of a larger size: one core per tile to process in the image.

TODO: is this example correct? Do we need to specify the offsets, etc. to make sure we aren't doing extra work?

```bash
make clean && make herd
```

#### ```segment```: Tiling using multiple segments

This example ([matrix_scalar_add_segment.py](matrix_scalar_add_segment.py))  uses segments and the internal logic of channels to process all the tiles. Here, there is one launch and several segments: one segment per tile. Each segment consists of a single herd and each of those herds manages a single core which processes one tile. The structure of this example is very similar to the herd example.

TODO: is this example correct? Do we need to specify the offsets, etc. to make sure we aren't doing extra work?

```bash
make clean && make segment
```

#### ```launch```: Tiling using multiple launches

This example ([matrix_scalar_add_launch.py](matrix_scalar_add_launch.py)) uses launches and channel sizes, strides, and offsets. Each launch only reads/writes a single tile from the input/output. Each launch then manages a single segment, herd, and core, and that core processes the single tile of data.

```bash
make clean && make launch
```

## Recommended Exercises
* Generate the mlir from any example and compare the differences between the generated mlir of each example:
  ```bash
  python matrix_scala_add_[channel|herd|segment|launch].py
  ```
* Pick one of the programs and modify it so only the first value in each tile is incremented




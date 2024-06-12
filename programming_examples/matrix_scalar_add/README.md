# Matrix Scalar Addition

This example focuses on a core concept: processing input as a grid of smaller inputs. In this case, each implementation of this program increments every value in a two dimensional matrix. To do this, the program divides the input *image* into smaller *tiles* and increments each value in each tile.

This example also introduces the concept of AIR *channels*, the abstraction used to represent data movement in the AIR compiler Python bindings. In this example, there is an input channel and an output channel. The entire image is written to/read from it's respective channel by the test harness ([run.py](run.py)) but the process on the NPU only reads/writes one tiles worth of data from the respective channel at a time.

In terms of other AIR terminology, this example consists of:
* A *launch*, which manages:
  * Two channels (an input channel and an output channel)
  * A segment

* The *segment* manages:
  * One herd

* The *herd*, which consists of 
  * One or more compute cores

## Running and Testing

For illustrative purposes, we provide five different ways specify data tiling in this example: ```channel```, ```herd```, ```segment```, ```launch```, and ```implicit```.

#### ```channel```: Tiling using channel sizes, offsets, and strides

This example uses channel sizes, offsets, and strides to explicitly loop over the image worth of data to read, increment, and write one tile at a time on one core.
```bash
make clean && make channel
```

#### ```herd```: Tiling using TODO

This example uses TODO
```bash
make clean && make herd
```

#### ```segment```: Tiling using TODO

This example uses TODO
```bash
make clean && make segment
```

#### ```launch```: Tiling using TODO

This example uses TODO
```bash
make clean && make launch
```

#### ```implicit```: Tiling using TODO

This example uses TODO
```bash
make clean && make implicit
```

## Recommended Exercises
* Generate the mlir from any example and compare the differences between the generated MLIR of each example:
  ```bash
  python matrix_scala_add_[channel|herd|segment|launch|implicit].py
  ```
* Pick one of the programs and modify it so only the first value in each tile is incremented




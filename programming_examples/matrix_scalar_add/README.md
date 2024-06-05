# Matrix Scalar Addition

This example focuses on a core concept: processing input as a grid of smaller inputs. In this case, the program defined in [matrix_scalar_add](matrix_scalar_add.py) increments every value in a matrix. To do this, the program divides the input *image* into smaller *tiles* and increments each value in each tile.

This example also introduces the concept of AIR *channels*, the abstraction used to represent data movement in the AIR compiler Python bindings. In this example, there is an input channel and an output channel. The entire image is written to/read from it's respective channel by the test harness ([run.py](run.py)) but the process on the NPU only reads/writes one tiles worth of data from the respective channel at a time.

In terms of other AIR terminology, this example consists of:
* A *launch*, which manages:
  * Two channels (an input channel and an output channel)
  * A segment

* The *segment* manages:
  * One herd

* The herd, which is defined to consist of:
  * One compute core (topology is ```[1,1]```)





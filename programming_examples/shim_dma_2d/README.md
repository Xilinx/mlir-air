# shim_dma_2d

This example demonstrates how data may be moved using shim DMA operations. In this example, a 2-dimensional block of data (referred to in test code as an *image*) is set to have some specific values. The upper corner of the image (referred to in test code as the *tile*) is transferred to a compute core using DMA. The compute core then copies all the data in the tile and sends the data out into an output *image*. When run, the output image is checked to verify that the tile region shows the values from the input image (showing the data transfer was successful) while the remainder of the output image is checked to ensure it retains the original output image values (showing the data is written to the correct tile region in the output image).

The logic in this example is defined in [shim_dma_2d.py](shim_dma_2d.py), and uses Python AIR bindings to generate AIR MLIR.

## Running and Testing

To generate AIR MLIR from the Python bindings, run:
```bash
python shim_dma_2d.py
```

For illustrative purposes, we provide three different ways to run and test this example. The three approaches are functionally equivalent but the implementation of each approach differs. The general workflow of each is:
* Build
  * The AIR Python bindings are used to generate AIR MLIR (generally a file called ```air.mlir```)
  * The AIR MLIR code is transformed and compiled by ```aircc.py```, which may be invoked either on the command line or through a python wrapper. ```aircc.py``` calls ```air-opt``` to run pipelines of passes over the initial AIR MLIR. For more control, there also exist mechanisms (not shown in this example) to customize the passes used. For the curious reader, most of the tests in [```/test/xrt```](/test/xrt) take this approach.
  * The final step is to produce the ```xclbin``` binary which contains one or more kernels that are capable of running on an NPU.
* Test
  * Setup input/output regions
  * Extract the compatible kernel from the ```xclbin``` and load the kernel on the device
  * Run the kernel on the NPU
  * Check that the output(s) contain the expected data

### Method 1: Run and test with AIR utility functions

This is the cleanest and simplest method of specifying a workflow to run AIR MLIR on an NPU, and uses code in the [run.py](run.py) file. The utility class ```XRTRunner``` simplifies setting up input/output data and allows the user to specify the input and an expected output using ```numpy``` ```ndarray```s. Behind the scenes, ```XRTRunner``` calls ```aircc.py``` with the default set of pipelines and passes, but for this and most examples, we use ```experimental_passes``` to signify we also want to run additional passes which should increase efficiency.
```bash
make pyworkflow
```

Note that if you'd like to run this example with verbose output - to see all the commands that are run, for example - you can change the ```VERBOSE``` variable in ```run.py``` to ```True```.

### Method 2: Generate AIR MLIR with python, compile on the command line, and run with python

This method uses the [test.py](test.py) file. While method 1 may be more user-friendly, this method is included as a frame of reference to understand the processes and steps that are abstracted by the AIR XRT backend utility functions used in method 1.

```bash
make
make run_py
```

### Method 3: Generate AIR MLIR with python, compile on the command line, and run with C++

This method uses the [test.cpp](test.cpp) file. While Method 1 may be more user-friendly, this method is used to show how C++ may be used.
```bash
make
make run
```
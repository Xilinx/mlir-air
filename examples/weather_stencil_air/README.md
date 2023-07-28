# SPARTA: Spatial Acceleration for Efficient and Scalable Horizontal Diffusion Weather Stencil Computation

## Introduction
A stencil operation sweeps over an input grid, updating values based on a fixed pattern. High-order stencils are applied to multidimensional grids that have sparse and irregular memory access patterns, limiting the achievable performance. In addition, stencils have a limited cache data reuse which further enhances memory access pressure. 

Real-world climate and weather simulations involve the utilization of complex compound stencil kernels, which are composed of a combination of different stencils. Horizontal diffusion (hdiff) is one such important compound stencil found in many regional global climate and weather prediction models.  It is a mathematical technique to help smooth out small-scale variations in the atmosphere and reduce the impact of numerical errors.  hdiff iterates over an input grid performing Laplacian and flux as depicted in Figure 1 to calculate different grid points. A Laplacian stencil accesses the input grid at five memory offsets in horizontal dimensions. The Lapalacian results together with input data are used to calculate the flux stencil. 


<p align="center">
  <picture>
  	<source media="(prefers-color-scheme: light)" srcset="img/hdiff_comp.png">
  <img alt="hdiff-comp" src="img/hdiff_comp.png" width="400">
  </picture>
  <br>
  <b>Figure 1: Horizontal diffusion (hdiff) kernel composition using Laplacian and flux stencils in a two dimensional plane</b>
</p>

Our goal is to mitigate the performance bottleneck of memory-bound weather stencil computation using AMD-Xilinx Versal AI Engine (AIE). To this end, we introduce SPARTA, a novel spatial accelerator for horizontal diffusion stencil computation. We exploit the two-dimensional spatial architecture to efficiently accelerate horizontal diffusion stencil. We design the first scaled-out spatial accelerator using MLIR (multi-level intermediate representation) compiler framework.

## Mapping onto the AI Engine Cores
We carefully hand-tune the code to overlap memory operations with arithmetic operations, to improve performance. We use the AIE data forwarding interfaces to forward the results from the first AIE core (used for Laplacian calculation) to the subsequent AIE core (used for flux calculation). This approach allows for the concurrent execution of multiple stencil calculations, which can increase the overall performance and throughput of the hdiff design.

We make the following two observations. First, the compute-to-memory intensity ratio of the Laplacian is more balanced compared to the compute to memory intensity ratio of the flux. The flux stencil has a higher compute bound than the memory bound. Third, the non-MAC operations (subtract, compare, and select) in the flux stencil lead to higher compute cycles due to the frequent movement of data between vector registers. 
We conclude that to achieve the maximum throughput, we need to split the hdiff computation over multiple AIE cores. By splitting the hdiff computation over multiple cores, we get two main benefits. First, the compute-bound can be distributed among different cores, allowing for the concurrent execution of multiple stencil calculations, which can increase the overall performance and throughput of the hdiff algorithm. Second, it allows for the use of more number of parallel AIE cores to achieve higher throughput.

Figure 3 shows the multi-AIE design approach for hdiff, where the flux stencil uses the results of the Laplacian stencil and input data to perform its computation. We also show the dataflow sequence from the external DRAM memory to the AIE cores. Instead of waiting for the Laplacian AIE core to complete all five Laplacian stencils required for a single hdiff output, we forward the result for each Laplacian stencil to the flux AIE core, thereby allowing both cores to remain active. In tri-AIE design, we further split flux computation and map MAC operation and non-MAC operations onto different AIE cores.

<p align="center">
  <picture>
  	<source media="(prefers-color-scheme: light)" srcset="img/multi_aie.png">
  <img alt="multi-hdiff" src="img/multi_aie.png" width="400">
  </picture>
  <br>
  <b>Figure 3: Multi-AIE design for hdiff computation to balance the compute and the memory bound. We show the dataflow sequence from the DRAM memory to the AIE cores via shimDMA</b>
</p>
Both Laplacian and flux computations require access to the input data, which is stored in the external DRAM . Therefore, we broadcast the input data onto the local memory of both Laplacian and Flux AIE cores using a single shimDMA channel. In dual-AIE design, a single Flux core is responsible for performing all MAC and non-MAC operations.  As mentioned above, flux operations have an imbalance between compute and memory bounds. Therefore, to further improve the compute performance, we split flux operations over two AIE cores in our tri-AIE design.

## Scaling Accelerator Design
The performance of the hdiff implementation can be maximized by scaling it out across as many AIE cores as possible while avoiding data starvation. As there are only limited shimDMA interfaces, system architects need to develop a design that can balance compute and memory resources without starving the available cores. 

We develop a *bundle* or *B-block*-based design. A B-block is a cluster of AIE cores connected to the same shimDMA input/output channel. As shown in Figure 4, clusters of AIE cores are connected to two channels of a shimDMA (one for input and one for output). Each B-block comprises four lanes (or rows) of our tri-AIE design, with each lane calculating a different offset of output result using a part of the input plane. As each lane requires access to 5 rows of the input grid to perform a single hdiff computation, we use the broadcast  feature of the global interconnect to duplicate the 8 rows of the input data into a circular buffer in the AIE cores of the first column. An 8-element circular buffer allows all the cores in the B-block lanes to work on a different offset of the input data while having 5 input grid rows necessary to perform hdiff computation.


<p align="center">
  <picture>
  	<source media="(prefers-color-scheme: light)" srcset="img/scale_hdiff.png">
  <img alt="scale-hdiff" src="img/scale_hdiff.png" width="400">
  </picture>
  <br>
  <b>Figure 4: Block-based design (B-block) using tri-AIE implementation for hdiff. The b-block-based design allows scaling computation across the AIE cores while bal ancing compute and communication time without getting bottlenecked by limited shimDMA channels</b>
</p>

As AIE architecture lacks support for automatically gathering and ordering of computed outputs, we use physical placement constraints to allow the AIE cores in the last column to access a single shared memory of a dedicated AIE core, enabling data gathering. We refer to this core as the *gather core*. The gather core is responsible for collecting data from all other cores, in addition to processing the results of its own lane. A single B-block operates on a single plane of the input data. Since two B-blocks can be connected to a single shimDMA, two planes can be served per shimDMA. This regular structure can then be repeated for all the shimDMA channels present on a Versal device. 

## More Information
[SPARTA](https://github.com/Xilinx/mlir-aie/tree/main/reference_designs/horizontal_diffusion)

# Citation
>Gagandeep Singh, Alireza Khodamoradi, Kristof Denolf, Jack Lo, Juan Gómez-Luna, Joseph Melber, Andra Bisca, Henk Corporaal, Onur Mutlu.
[**"SPARTA: Spatial Acceleration for Efficient and Scalable Horizontal Diffusion Weather Stencil Computation"**](https://arxiv.org/pdf/2303.03509.pdf)
In _Proceedings of the 37th International Conference on Supercomputing (ICS),_ Orlando, FL, USA, June 2023.

Bibtex entry for citation:

```
@inproceedings{singh2023sparta,
  title={{SPARTA: Spatial Acceleration for Efficient and Scalable Horizontal Diffusion Weather Stencil Computation}},
  author={Singh, Gagandeep and Khodamoradi, Alireza and Denolf, Kristof and Lo, Jack and G{\'o}mez-Luna, Juan and Melber, Joseph and Bisca, Andra and Corporaal, Henk and Mutlu, Onur},
  booktitle={ICS},
  year={2023}
}
```
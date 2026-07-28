# Used in / Cited in

MLIR-AIR is a spatial-compute compiler stack for AMD NPUs and Versal AI Engine
arrays. This page collects how to cite the project and a selection of work that
builds on it. It is curated rather than exhaustive.

If you have built on MLIR-AIR and would like your work considered for this page,
open a pull request.

## How to cite

Work that uses MLIR-AIR should cite the toolchain paper:

> E. Wang, S. Bayliss, A. Bisca, Z. Blair, S. Chowdhary, K. Denolf, J. Fifield,
> B. Freiberger, E. Hunhoff, P. James-Roxby, J. Lo, J. Melber, S. Neuendorffer,
> E. Richter, A. Rosti, J. Setoain, G. Singh, E. Taka, P. Vasireddy, Z. Yu,
> N. Zhang, and J. Zhuang. "From Loop Nests to Silicon: Mapping AI Workloads
> onto AMD NPUs with MLIR-AIR." ACM Transactions on Reconfigurable Technology
> and Systems (TRETS), 19(2), Article 28, June 2026.
> <https://doi.org/10.1145/3785670>

```bibtex
@article{10.1145/3785670,
  author     = {Wang, Erwei and Bayliss, Samuel and Bisca, Andra and Blair, Zachary and Chowdhary, Sangeeta and Denolf, Kristof and Fifield, Jeff and Freiberger, Brandon and Hunhoff, Erika and James-Roxby, Phil and Lo, Jack and Melber, Joseph and Neuendorffer, Stephen and Richter, Eddie and Rosti, Andre and Setoain, Javier and Singh, Gagandeep and Taka, Endri and Vasireddy, Pranathi and Yu, Zhewen and Zhang, Niansong and Zhuang, Jinming},
  title      = {From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR},
  year       = {2026},
  issue_date = {June 2026},
  volume     = {19},
  number     = {2},
  issn       = {1936-7406},
  url        = {https://doi.org/10.1145/3785670},
  doi        = {10.1145/3785670},
  journal    = {ACM Trans. Reconfigurable Technol. Syst.},
  month      = may,
  articleno  = {28},
  numpages   = {36},
  keywords   = {Compiler, dataflow architecture, hardware acceleration, machine learning, reconfigurable technology, spatial architecture.}
}
```

Work that uses the MLIR-AIR **agent skills** for LLM deployment should cite:

> J. Li, E. Wang, Z. Zhang, and S. Bayliss. "From Human Guidance to Autonomy:
> Agent Skill System for End-to-End LLM Deployment on Spatial NPUs." Workshop on
> Machine Learning for Computer Architecture and Systems (MLArchSys), colocated
> with the 53rd International Symposium on Computer Architecture (ISCA), 2026.
> arXiv:2606.07586. <https://arxiv.org/abs/2606.07586v2>

```bibtex
@inproceedings{li2026aiskills,
  title     = {From Human Guidance to Autonomy: Agent Skill System for
               End-to-End LLM Deployment on Spatial NPUs},
  author    = {Li, Jiajie and Wang, Erwei and Zhang, Zhiru and Bayliss, Samuel},
  booktitle = {Workshop on Machine Learning for Computer Architecture and
               Systems (MLArchSys), colocated with the 53rd International
               Symposium on Computer Architecture (ISCA)},
  year      = {2026},
  eprint    = {2606.07586},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2606.07586v2}
}
```

## Repositories

- [amd/IRON](https://github.com/amd/IRON) — IRON operators, AIE kernels, and
  example applications, built on the MLIR-AIE Python bindings.
- [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) — the MLIR-AIE backend
  MLIR-AIR lowers onto (per-tile code, DMA descriptors, hardware locks).
- [Xilinx/llvm-aie (Peano)](https://github.com/Xilinx/llvm-aie) — the LLVM fork
  adding the AI Engine as a target architecture.
- [amd/Triton-XDNA](https://github.com/amd/Triton-XDNA) — a Triton frontend that
  targets MLIR-AIR as its AMD XDNA NPU backend, alongside Triton's native GPU
  backends.

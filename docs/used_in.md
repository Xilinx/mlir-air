# Used in / Cited in

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

## Publications using MLIR-AIR

- **[KernelCraft: Benchmarking for Agentic Close-to-Metal Kernel Generation on
  Emerging Hardware](https://arxiv.org/abs/2603.08721)**<br>
  J. Nie, H. Wu, Y. Lai, Z. Cao, C. Zhang, B. Lou, E. Wang, J. Cheng,
  T. M. Jones, R. Mullins, R. Antonova, and Y. Zhao. International Conference on
  Machine Learning (ICML), 2026.

- **[Practical Formal Verification for MLIR Programs](https://arxiv.org/abs/2605.01124)**<br>
  E. Tucker, L.-N. Pouchet, E. Hunhoff, S. Neuendorffer, and E. Wang.
  International Conference on Parallel Architectures and Compilation Techniques
  (PACT), 2026.

- **[Programming AMD XDNA NPUs with Open-source Compiler Tools: A FlashAttention
  Case Study](https://arxiv.org/abs/2609.21264)**<br>
  E. Wang, E. Wu, V. J. B. Jung, J. Li, A. Rosti, J. Melber, and S. Bayliss.
  arXiv preprint arXiv:2609.21264, September 2026.

## Repositories

- [amd/IRON](https://github.com/amd/IRON) — IRON operators, AIE kernels, and
  example applications, built on the MLIR-AIE Python bindings.
- [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) — the MLIR-AIE backend
  MLIR-AIR lowers onto.
- [Xilinx/llvm-aie (Peano)](https://github.com/Xilinx/llvm-aie) — a fork of the
  LLVM compiler infrastructure ([llvm/llvm-project](https://github.com/llvm/llvm-project))
  adding the AI Engine as a target architecture.
- [amd/Triton-XDNA](https://github.com/amd/Triton-XDNA) — a Triton frontend that
  adds MLIR-AIR as its AMD XDNA NPU backend, alongside Triton's native GPU
  backends.

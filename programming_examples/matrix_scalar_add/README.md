# Build Step-By-Step

First, change the class in `manual_transforms.py` to be the class we'd like to build from.

```bash
mkdir manual_build
cd manual_build

python ../manual_transforms.py
python ../../../mlir-aie/install/bin/aiecc.py --no-aiesim --aie-generate-cdo --aie-generate-npu --no-compile-host --xclbin-name=aie.xclbin --npu-insts-name=insts.txt aie.mlir
python ../manual_run.py
```

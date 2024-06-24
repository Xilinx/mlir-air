# Build Step-By-Step

```bash
mkdir manual_build
cd manual_build

python ../manual_transforms.py
python ../../../mlir-aie/install/bin/aiecc.py --no-aiesim --aie-generate-cdo --aie-generate-npu --no-compile-host --xclbin-name=aie.xclbin --npu-insts-name=insts.txt aie.mlir
python ../manual_run.py
```

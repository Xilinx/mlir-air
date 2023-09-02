#!/usr/bin/env bash
set -xe

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

if [ ! -z "$MLIR_AIR_WHEEL_VERSION" ]; then
  MLIR_AIR_WHEEL_VERSION="==$MLIR_AIR_WHEEL_VERSION"
fi

if [ x"$CIBW_ARCHS" == x"arm64" ] || [ x"$CIBW_ARCHS" == x"aarch64" ]; then
  if [ x"$MATRIX_OS" == x"macos-11" ] && [ x"$CIBW_ARCHS" == x"arm64" ]; then
    PLAT=macosx_11_0_arm64
  elif [ x"$MATRIX_OS" == x"ubuntu-20.04" ] && [ x"$CIBW_ARCHS" == x"aarch64" ]; then
    PLAT=linux_aarch64
  fi

  pip install mlir-air$MLIR_AIR_WHEEL_VERSION --platform $PLAT --only-binary=:all: --target $SITE_PACKAGES  --no-deps --force -U \
    -f https://makslevental.github.io/wheels \
    -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels \
    -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-wheels

else

  pip install mlir-air$MLIR_AIR_WHEEL_VERSION --force -U \
    -f https://makslevental.github.io/wheels \
    -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels \
    -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-wheels

fi

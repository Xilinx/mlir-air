#!/usr/bin/env bash
set -xe
HERE=$(dirname "$(realpath "$0")")

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=linux;;
    Darwin*)    machine=macos;;
    CYGWIN*)    machine=windows;;
    MINGW*)     machine=windows;;
    MSYS_NT*)   machine=windows;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo "${machine}"

export MLIR_WHEEL_VERSION="18.0.0.2023091015+780b046b"
export MLIR_AIE_WHEEL_VERSION="0.0.1.2023091105+a0cc3d0"

if [ "$machine" == "linux" ]; then
  export CIBW_ARCHS=${CIBW_ARCHS:-x86_64}
  export PARALLEL_LEVEL=15
  export MATRIX_OS=ubuntu-20.04
elif [ "$machine" == "macos" ]; then
  export CIBW_ARCHS=${CIBW_ARCHS:-arm64}
  export MATRIX_OS=macos-11
  export PARALLEL_LEVEL=32
else
  export MATRIX_OS=windows-2019
  export CIBW_ARCHS=${CIBW_ARCHS:-AMD64}
fi

ccache --show-stats
ccache --print-stats
ccache --show-config

export HOST_CCACHE_DIR="$(ccache --get-config cache_dir)"

if [ x"$CIBW_ARCHS" == x"aarch64" ]; then
  export PIP_NO_BUILD_ISOLATION="false"
  pip install -r $HERE/../requirements.txt
  $HERE/../scripts/pip_install_mlir.sh

  CMAKE_GENERATOR=Ninja \
  pip wheel $HERE/.. -v -w $HERE/../wheelhouse
else
  cibuildwheel "$HERE"/.. --platform "$machine"
fi

cp -R "$HERE/../scripts" "$HERE/../python_bindings"

pushd "$HERE/../python_bindings"

cibuildwheel --platform "$machine" --output-dir ../wheelhouse

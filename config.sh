CLANG_VER=8

cmake -GNinja \
        -DLLVM_DIR=$1/../peano/lib/cmake/llvm \
        -DMLIR_DIR=$1/../peano/lib/cmake/mlir \
        -DCMAKE_C_COMPILER=clang-${CLANG_VER} \
        -DCMAKE_CXX_COMPILER=clang++-${CLANG_VER} \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_INSTALL_PREFIX=$3 \
        -B$1 -H$2

#        -DAIR_LIBXAIE_ENABLE=On \
#
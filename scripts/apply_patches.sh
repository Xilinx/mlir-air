#!/usr/bin/env bash
set -xe

# note that space before slash is important
PATCHES="\
airtoaie \
boost \
capi \
export_targets \
mscv \
python \
runner \
transformgen \
win32 \
"

if [[ x"${APPLY_PATCHES}" == x"true" ]]; then
  for PATCH in $PATCHES; do
    echo "applying $PATCH"
    ls mlir-air
    git apply --ignore-space-change --ignore-whitespace --verbose --directory mlir-air patches/$PATCH.patch || git apply --ignore-space-change --ignore-whitespace --verbose --directory mlir-air patches/$PATCH.patch -R --check && echo already applied
  done
fi
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install


def check_env(build, default=0):
    return os.getenv(build, str(default)) in {"1", "true", "True", "ON", "YES"}


# Always use forward slashes for CMake paths.
def _cmake_path(p: object) -> str:
    return os.fspath(p).replace("\\", "/")


# ---------------------------------------------------------------------------
# Windows short-path helpers (junction links).  Matches the pattern used in
# mlir-aie's wheel builds to keep all paths well under MAX_PATH (260 chars).
# ---------------------------------------------------------------------------
def _windows_short_root() -> Path:
    return Path(os.getenv("AIR_WHEEL_BUILD_ROOT", "C:/tmp/airwhls")).absolute()


def _windows_short_dir(name: str, *, clean: bool = False) -> Path:
    path = _windows_short_root() / name
    if clean and path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _windows_short_alias(name: str, target: Path) -> Path:
    """Create a junction link at *_windows_short_root()/name* → *target*."""
    link = _windows_short_root() / name
    target = target.absolute()
    if link.exists():
        try:
            if link.resolve() == target.resolve():
                return link
        except OSError:
            pass
        subprocess.run(
            ["cmd", "/c", "rmdir", str(link)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return link


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


def get_cross_cmake_args():
    cmake_args = {}

    def native_tools():
        nonlocal cmake_args

        mlir_tblgen_host = next(
            f.locate()
            for f in files("mlir-native-tools")
            if f.name.startswith("mlir-tblgen")
        )
        mlir_tblgen_target = next(
            f.locate()
            for f in files("mlir" if check_env("ENABLE_RTTI", 1) else "mlir_no_rtti")
            if f.name.startswith("mlir-tblgen")
        )
        os.remove(mlir_tblgen_target)
        shutil.copy(mlir_tblgen_host, mlir_tblgen_target)
        mlir_pdll_host = next(
            f.locate()
            for f in files("mlir-native-tools")
            if f.name.startswith("mlir-pdll")
        )
        mlir_pdll_target = next(
            f.locate()
            for f in files("mlir" if check_env("ENABLE_RTTI", 1) else "mlir_no_rtti")
            if f.name.startswith("mlir-pdll")
        )
        os.remove(mlir_pdll_target)
        shutil.copy(mlir_pdll_host, mlir_pdll_target)

    CIBW_ARCHS = os.getenv("CIBW_ARCHS")
    if CIBW_ARCHS in {"arm64", "aarch64", "ARM64"}:
        ARCH = cmake_args["LLVM_TARGETS_TO_BUILD"] = "AArch64"
    elif CIBW_ARCHS in {"x86_64", "AMD64"}:
        ARCH = cmake_args["LLVM_TARGETS_TO_BUILD"] = "X86"
    else:
        raise ValueError(f"unknown CIBW_ARCHS={CIBW_ARCHS}")
    if CIBW_ARCHS != platform.machine():
        cmake_args["CMAKE_SYSTEM_NAME"] = platform.system()

    if platform.system() == "Darwin":
        if ARCH == "AArch64":
            cmake_args["CMAKE_OSX_ARCHITECTURES"] = "arm64"
            cmake_args["LLVM_HOST_TRIPLE"] = "arm64-apple-darwin21.6.0"
            native_tools()
        elif ARCH == "X86":
            cmake_args["CMAKE_OSX_ARCHITECTURES"] = "x86_64"
            cmake_args["LLVM_HOST_TRIPLE"] = "x86_64-apple-darwin"
    elif platform.system() == "Linux":
        if ARCH == "AArch64":
            cmake_args["LLVM_HOST_TRIPLE"] = "aarch64-linux-gnu"
            cmake_args["CMAKE_C_COMPILER"] = "aarch64-linux-gnu-gcc"
            cmake_args["CMAKE_CXX_COMPILER"] = "aarch64-linux-gnu-g++"
            cmake_args["CMAKE_CXX_FLAGS"] = "-static-libgcc -static-libstdc++"
            cmake_args["SysrootAarch64"] = "/usr/aarch64-linux-gnu"
            native_tools()
        elif ARCH == "X86":
            cmake_args["LLVM_HOST_TRIPLE"] = "x86_64-unknown-linux-gnu"
            cmake_args["LLVM_TARGET_ARCH"] = "X86"

    return cmake_args


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        install_dir = extdir / "mlir_air"
        cfg = "Release"

        cmake_generator = os.getenv("CMAKE_GENERATOR", "Ninja")

        # Get MLIR install path from downloaded wheel
        MLIR_INSTALL_ABS_PATH = Path(
            os.getenv(
                "MLIR_INSTALL_ABS_PATH",
                Path(__file__).parent
                / ("mlir" if check_env("ENABLE_RTTI", 1) else "mlir_no_rtti"),
            )
        ).absolute()

        # Get mlir-aie install path from environment variable
        MLIR_AIE_INSTALL_PATH = Path(
            os.getenv(
                "MLIR_AIE_INSTALL_PATH",
                Path(__file__).parent
                / ("mlir_aie" if check_env("ENABLE_RTTI", 1) else "mlir_aie_no_rtti"),
            )
        ).absolute()

        cmake_source_dir = Path(ext.sourcedir)

        if platform.system() == "Windows":
            # Create junction links to keep source path short, avoiding
            # MSVC linker issues from long paths in .rsp files.
            cmake_source_dir = _windows_short_alias("src", cmake_source_dir).absolute()

        cmake_args = [
            "-G",
            cmake_generator,
            f"-DCMAKE_INSTALL_PREFIX={_cmake_path(install_dir)}",
            f"-DPython3_EXECUTABLE={_cmake_path(sys.executable)}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DLLVM_DIR={_cmake_path(MLIR_INSTALL_ABS_PATH)}/lib/cmake/llvm",
            f"-DMLIR_DIR={_cmake_path(MLIR_INSTALL_ABS_PATH)}/lib/cmake/mlir",
            f"-DAIE_DIR={_cmake_path(MLIR_AIE_INSTALL_PATH)}/lib/cmake/aie",
            f"-DCMAKE_MODULE_PATH={_cmake_path(cmake_source_dir)}/cmake/modules",
            # Prevent symbol collision
            "-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON",
            "-DCMAKE_C_VISIBILITY_PRESET=hidden",
            "-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DLLVM_VERSION_SUFFIX=",
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
            "-DLLVM_CCACHE_BUILD=ON",
            f"-DLLVM_ENABLE_RTTI={os.getenv('ENABLE_RTTI', 'ON')}",
            "-DAIE_ENABLE_BINDINGS_PYTHON=ON",
            "-DMLIR_DETECT_PYTHON_ENV_PRIME_SEARCH=ON",
            "-DAIR_RUNTIME_TARGETS=x86_64",
            f"-Dx86_64_TOOLCHAIN_FILE={_cmake_path(cmake_source_dir)}/cmake/modules/toolchain_x86_64.cmake",
            "-DPython_FIND_VIRTUALENV=ONLY",
            "-DPython3_FIND_VIRTUALENV=ONLY",
        ]

        # Add XRT support if available
        if os.getenv("XRT_ROOT"):
            xrt_dir = Path(os.getenv("XRT_ROOT")).absolute()
            cmake_args.append(f"-DXRT_ROOT={xrt_dir}")
            cmake_args.append("-DENABLE_RUN_XRT_TESTS=ON")

        if shutil.which("ccache"):
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
                "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
                "-DCMAKE_C_FLAGS=/MT",
                "-DCMAKE_CXX_FLAGS=/MT",
            ]
        else:
            cmake_args += [
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
            ]

        if shutil.which("lld") and platform.system() != "Windows":
            cmake_args.append("-DLLVM_USE_LINKER=lld")

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.getenv("CMAKE_ARGS").split(" ") if item]

        cmake_args_dict = get_cross_cmake_args()
        cmake_args += [f"-D{k}={v}" for k, v in cmake_args_dict.items()]

        build_args = [f"-j{os.getenv('PARALLEL_LEVEL', 2 * os.cpu_count())}"]

        build_temp = Path(self.build_temp) / ext.name
        if platform.system() == "Windows":
            build_temp = _windows_short_dir("bld", clean=True)
        elif not build_temp.exists():
            build_temp.mkdir(parents=True)

        print("ENV", pprint(os.environ), file=sys.stderr)
        print("cmake", " ".join(cmake_args), file=sys.stderr)

        subprocess.run(
            ["cmake", _cmake_path(cmake_source_dir), *cmake_args],
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", "--target", "install", *build_args],
            cwd=build_temp,
            check=True,
        )


class DevelopWithPth(develop):
    """Custom develop command to create a .pth file into the site-packages directory."""

    def run(self):
        super().run()
        pth_target = os.path.join(self.install_dir, "air.pth")
        with open(pth_target, "w") as pth_file:
            pth_file.write("mlir_air/python")


class InstallWithPth(install):
    """Custom install command to create a .pth file into the site-packages directory."""

    def run(self):
        super().run()
        pth_target = os.path.join(self.install_lib, "air.pth")
        with open(pth_target, "w") as pth_file:
            pth_file.write("mlir_air/python")


def get_version():
    if "AIR_WHEEL_VERSION" in os.environ and os.environ["AIR_WHEEL_VERSION"].lstrip(
        "v"
    ):
        return os.environ["AIR_WHEEL_VERSION"].lstrip("v")
    release_version = "0.0.1"
    commit_hash = os.environ.get("AIR_PROJECT_COMMIT", "deadbeef")
    now = datetime.now()
    timestamp = os.environ.get(
        "DATETIME", f"{now.year}{now.month:02}{now.day:02}{now.hour:02}"
    )
    suffix = "" if check_env("ENABLE_RTTI", 1) else "_no_rtti"
    return f"{release_version}.{timestamp}+{commit_hash}{suffix}"


MLIR_AIR_SOURCE_DIR = Path(
    os.getenv("MLIR_AIR_SOURCE_DIR", Path(__file__).parent.parent.parent)
).absolute()


def parse_requirements(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
        return [
            line.strip() for line in lines if line.strip() and not line.startswith("#")
        ]


setup(
    version=get_version(),
    license="MIT",
    include_package_data=True,
    ext_modules=[CMakeExtension("_mlir_air", sourcedir=MLIR_AIR_SOURCE_DIR)],
    cmdclass={
        "build_ext": CMakeBuild,
        "develop": DevelopWithPth,
        "install": InstallWithPth,
    },
    zip_safe=False,
    packages=find_packages(exclude=["wheelhouse", "mlir-air"]),
    python_requires=">=3.10",
    install_requires=parse_requirements(Path(__file__).parent / "requirements.txt"),
)

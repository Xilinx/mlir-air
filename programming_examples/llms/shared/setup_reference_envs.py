# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Create pinned CPU-reference environments, sharing the caller's Torch install.

Usage: python3 setup_reference_envs.py /path/to/reference-envs
Run this after installing the general LLM runtime requirements.
"""

from pathlib import Path
import subprocess
import sys
import sysconfig


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1]).resolve()
    llms = Path(__file__).resolve().parents[1]
    for name, model, version in (
        ("current", "phi2_2_7b_q4nx", "5.3.0"),
        ("openelm", "openelm_1b_q4nx", "4.39.3"),
    ):
        dest = root / name
        subprocess.run([sys.executable, "-m", "venv", str(dest)], check=True)
        python = str(dest / "bin/python")
        site = Path(
            subprocess.check_output(
                [
                    python,
                    "-c",
                    "import sysconfig; print(sysconfig.get_path('purelib'))",
                ],
                text=True,
            ).strip()
        )
        # A child venv does not inherit the invoking venv. Add its packages
        # after the child's site-packages so pinned Transformers/Tokenizers/Hub
        # take precedence while the large CPU Torch installation is shared.
        (site / "air-runtime.pth").write_text(sysconfig.get_path("purelib") + "\n")
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "-r",
                str(llms / model / "requirements-reference.txt"),
            ],
            check=True,
        )
        subprocess.run(
            [
                python,
                "-c",
                "import torch, transformers; "
                f"assert transformers.__version__ == {version!r}; "
                "print('Reference environment:', torch.__version__, transformers.__version__)",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

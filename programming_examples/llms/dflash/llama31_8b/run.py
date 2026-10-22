# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Stream one response or enter a resident prompt session."""

from paths import HERE
from common.run_8b import main, run_prompt

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]", flush=True)
        raise SystemExit(130)

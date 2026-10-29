# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
def _main():
    from air_examples.llms.shared.deployment_cli import main

    return main("granite33_2b_q4nx")


if __name__ == "__main__":
    _main()

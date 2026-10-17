# ./python/test/compiler/xrt_backend_lock_race_fix.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""XRTBackend refuses lock race fix combinations aircc would refuse.

use_lock_race_condition_fix_auto is tri-state: None leaves the choice to
aircc, False turns auto mode off, True turns it on and so cannot be combined
with either design-wide fix.
"""

from air.backend.abc import AirBackendError
from air.backend.xrt import XRTBackend


def check(**options):
    try:
        XRTBackend(**options)
        print("accepted")
    except AirBackendError as e:
        print("rejected:", e)


# CHECK: accepted
check()
# CHECK: accepted
check(use_lock_race_condition_fix=True)
# CHECK: accepted
check(use_lock_race_condition_fix_v2=True)
# CHECK: accepted
check(use_lock_race_condition_fix_auto=True)
# CHECK: accepted
check(use_lock_race_condition_fix=True, use_lock_race_condition_fix_auto=False)
# CHECK: accepted
check(use_lock_race_condition_fix_v2=True, use_lock_race_condition_fix_auto=False)

# CHECK: rejected: {{.*}}mutually exclusive
check(use_lock_race_condition_fix=True, use_lock_race_condition_fix_v2=True)
# CHECK: rejected: {{.*}}mutually exclusive
check(use_lock_race_condition_fix=True, use_lock_race_condition_fix_auto=True)
# CHECK: rejected: {{.*}}mutually exclusive
check(use_lock_race_condition_fix_v2=True, use_lock_race_condition_fix_auto=True)

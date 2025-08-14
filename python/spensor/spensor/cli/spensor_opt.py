# ./python/spensor/spensor/cli/spensor_opt.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse

from xdsl.xdsl_opt_main import xDSLOptMain

from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl.dialects.test import Test
from xdsl.dialects.memref import MemRef

from spensor.dialects.spensor_dialect import SpensorDialect, NDSpensorDialect


class OptMain(xDSLOptMain):
    def register_all_dialects(self):
        self.ctx.register_dialect(Arith.name, lambda: Arith)
        self.ctx.register_dialect(Builtin.name, lambda: Builtin)
        self.ctx.register_dialect(Func.name, lambda: Func)
        self.ctx.register_dialect(Comb.name, lambda: Comb)
        self.ctx.register_dialect(Test.name, lambda: Test)
        self.ctx.register_dialect(MemRef.name, lambda: MemRef)
        self.ctx.register_dialect(SpensorDialect.name, lambda: SpensorDialect)
        self.ctx.register_dialect(NDSpensorDialect.name, lambda: NDSpensorDialect)
        self.ctx.load_registered_dialect(SpensorDialect.name)
        self.ctx.load_registered_dialect(NDSpensorDialect.name)

    def register_all_passes(self):
        super().register_all_passes()

    def register_all_targets(self):
        super().register_all_targets()

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        super().register_all_arguments(arg_parser)


def main():
    xdsl_main = OptMain()
    xdsl_main.run()


if __name__ == "__main__":
    main()

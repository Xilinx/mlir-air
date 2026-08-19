#!/usr/bin/env python3
"""Decode-latency bench for the qwen2.5-3B fused decode.

bench_decode.exe cannot be reused: qwen binds 4 data BOs in a different order
(x, w, kv, y -- no separate rms BO) than the llama/gemma path (x, w, rms, y, kv).
Argument order and every BO size are taken verbatim from the production
hand-off driver (fused_decode/qwen_prefill_to_decode.py:183-186, :245), so
nothing here is hand-derived.

Synthetic contents -- LATENCY ONLY, never a correctness gate, same caveat as
bench_decode.cpp. Reports the device dispatch span (the comparable number) and,
separately, the per-token host KV-window upload this design needs and the
llama/gemma device-append design does not.
"""

import os, sys, time, importlib

sys.path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "fused_decode")
)
ctx = int(sys.argv[1])
nl = int(os.environ.get("QWEN_NLAYERS", "36"))
d = sys.argv[2]
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 32
warm = 8
os.environ.update(ATTN_L=str(ctx), QWEN_NLAYERS=str(nl), W_DUAL_CHAN="1")
import numpy as np, pyxrt as xrt
from ml_dtypes import bfloat16

m = importlib.import_module("fused_decode_qwen")

X = max(m.X_CHUNKS * 2 * m.COL_BLOCK, nl * m.XLAYER)
WSZ = nl * m.W_LAYER
KVL = m.N_ATTN_CU * 2 * m.ATTN_ROUNDS * m.KVBLK
KVN = nl * KVL
YN = max(m.DEST_TOTAL * m.PAYLOAD, m.K + m.DQ)
print(
    f"  geometry: ATTN_MAXL={m.ATTN_MAXL} X={X} W={WSZ} KV={KVN} ({KVN*2/2**20:.0f} MiB) Y={YN}"
)

dev = xrt.device(0)
xb = xrt.xclbin(f"{d}/decode_L{ctx}.xclbin")
dev.register_xclbin(xb)
hwc = xrt.hw_context(dev, xb.get_uuid())
kern = xrt.kernel(
    hwc, [q for q in xb.get_kernels() if "MLIR_AIE" in q.get_name()][0].get_name()
)
g, HO = kern.group_id, xrt.bo.host_only
x_bo = xrt.bo(dev, X * 2, HO, g(3))
w_bo = xrt.bo(dev, WSZ * 2, HO, g(4))
kv_bo = xrt.bo(dev, KVN * 2, HO, g(5))
y_bo = xrt.bo(dev, YN * 2, HO, g(6))
insts = np.fromfile(f"{d}/decode_L{ctx}.insts.bin", dtype=np.uint32)
ib = xrt.bo(dev, insts.nbytes, xrt.bo.cacheable, g(1))
ib.write(insts, 0)
ib.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
one = np.full(1, 1.0, bfloat16).view(np.uint16)[0]
for bo, n in ((x_bo, X), (w_bo, WSZ), (kv_bo, KVN)):
    bo.write(np.full(n, one, np.uint16), 0)
    bo.sync(TO)
kvd = np.full(KVN, one, np.uint16)

disp, upl = [], []
for i in range(warm + iters):
    t0 = time.perf_counter()
    kv_bo.write(kvd, 0)
    kv_bo.sync(TO)
    x_bo.sync(TO)
    t1 = time.perf_counter()
    st = kern(3, ib, insts.size, x_bo, w_bo, kv_bo, y_bo).wait(60000)
    if "COMPLETED" not in str(st):
        print(f"DISPATCH DID NOT COMPLETE: state={st}")
        sys.exit(1)
    t2 = time.perf_counter()
    if i >= warm:
        upl.append((t1 - t0) * 1e3)
        disp.append((t2 - t1) * 1e3)
mean = sum(disp) / len(disp)
sd = (sum((v - mean) ** 2 for v in disp) / len(disp)) ** 0.5
um = sum(upl) / len(upl)
print(
    f"  ctx {ctx:<6} dispatch {mean:8.3f} ms  sd {sd:.3f}  min {min(disp):.3f}  "
    f"({1000/mean:.2f} tok/s)   + host KV upload {um:.1f} ms -> {1000/(mean+um):.2f} tok/s e2e"
)
print(
    f"CSV,{ctx},{mean:.3f},{sd:.3f},{min(disp):.3f},{1000/mean:.2f},{um:.3f},{1000/(mean+um):.2f}"
)

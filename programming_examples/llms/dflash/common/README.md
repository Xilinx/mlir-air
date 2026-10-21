# Shared DFlash implementation

The model examples share the greedy draft/verify/commit loop in
[speculative.py](speculative.py), incremental output in
[streaming.py](streaming.py), and bounded-memory packed-weight upload in
[weight_io.py](weight_io.py).

[kernels/](kernels/) contains the compute-tile arithmetic. The AIR attention
subgraph is in [fused_attention_graph.py](fused_attention_graph.py).
Model geometry, residual ownership, activation storage and the complete
dataflow remain explicit in each model's mapping. Common source is compiled
separately for each target and drafter configuration.

[build_identity.py](build_identity.py) includes shared sources in artifact
invalidation. A model's `paths.py` locates this package for standalone script
execution; examples do not import another model's implementation directory.

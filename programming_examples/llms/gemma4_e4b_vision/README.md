# Gemma4-E4B image + text on NPU2

Image turns for Gemma4-E4B-IT. This is [`gemma4_e2b_vision`](../gemma4_e2b_vision)
at `VARIANT=e4b`. The two models share the vision tower; E4B projects its soft
tokens to 2560 instead of 1536 and runs them through the text model of
[`gemma4_e4b_q4nx`](../gemma4_e4b_q4nx).

The targets are the same as in `gemma4_e2b_vision`:

```bash
export PEANO_INSTALL_DIR=/path/to/llvm-aie   # >= 22.0.0
pip install -r requirements.txt
make compile          # gemma4_e4b_q4nx's fused prefill + decode
make vision           # NPU tower vs HF, cosine >= 0.99
make run IMAGE=photo.jpg PROMPT="What is in this picture?"
make verify           # image-turn top-k gate vs HF bf16 + the shapes keyword gate
make profile-image-sweep   # TTFT and decode tok/s per image-token budget
```

Weights: the vision tower and the multimodal projection come from the bf16
`unsloth/gemma-4-E4B-it` checkpoint, and the text model is the
`FastFlowLM/Gemma4-E4B-IT-NPU2` q4nx bundle.

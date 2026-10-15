# Released models

Setting `uniform_rho5_r0`: the paper's main run (5% temporal uniform keyframes,
first run, 33 keyframes, seed 0). The tensors are exported unchanged from the
training checkpoints; no retraining was done.

- `dino_decoder.safetensors`: decoder state (2,056,197 parameters plus BatchNorm
  statistics), float32. MIT license, see `DECODER_LICENSE`. The frozen encoder is
  `facebook/dinov2-large` at revision `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`
  (Apache-2.0, downloaded separately).
- `sam3_trainable-00001-of-00001.safetensors`: the 304 SAM 3 tracker tensors
  (11,742,338 parameters) trained in the paper, float32. The frozen image backbone,
  point embeddings and buffers are not included and come from the official SAM 3
  checkpoint (commit `46957e47805eaa273f4aa7bbbd25a88bca9108ce`). Subject to the
  SAM License, see `SAM_LICENSE`.
- `index.json`: SAM 3 parameter names and their shard.
- `manifest.json`: file sizes and SHA-256 hashes, training steps (decoder 2000,
  SAM 3 1000) and model revisions.

Load them with `peercheck/release.py` or the `--release uniform_rho5_r0` option of
`scripts/run_peercheck.py` and `demo/full_demo.py`.

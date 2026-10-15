# Third-party notices

- **SAM 3**: code and models are governed by the official repository's
  [SAM License](https://github.com/facebookresearch/sam3/blob/46957e47805eaa273f4aa7bbbd25a88bca9108ce/LICENSE).
  No SAM 3 source tree or official checkpoint is bundled. `weights/` contains SAM 3
  tracker parameters fine-tuned in this study; they are distributed under the SAM
  License, a copy of which is in `weights/SAM_LICENSE`. The compatibility diff
  contains upstream context and is not relicensed under this repository's MIT
  license. Obtain access to the gated official weights separately.
- **DINOv2**: [official code and model license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE)
  is Apache-2.0. The release downloads the frozen model through Transformers;
  the encoder weights are not bundled. Only the decoder trained in this study is
  released (MIT).
- **Ultralytics**: the optional YOLO26 baseline requires the separately installed
  AGPL-3.0 package. See the [official repository](https://github.com/ultralytics/ultralytics)
  and its license. It is not a dependency of the main package or CPU demo.
- **SAM 2 loss formulation**: the single-device loss in `sam3_propagation.py`
  independently implements the focal, Dice, mask-quality and presence objectives
  used by the [SAM 2 training code](https://github.com/facebookresearch/sam2).
  No SAM 2 source files are bundled. The release audit checks numerical and
  gradient equivalence to the experiment's single-device loss adapter.

Other dependencies retain their respective licenses. The MIT grant applies only
to original PeerCheck code and does not supersede third-party terms.

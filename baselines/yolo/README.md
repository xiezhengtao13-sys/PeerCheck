# Optional YOLO26 comparator

Install the GPU requirements and this folder's requirements separately. Ultralytics
is AGPL-3.0; consult its official licensing terms. It is not required by PeerCheck.

```bash
python -m pip install -r baselines/yolo/requirements.txt
python baselines/yolo/run.py --input my_data --work private_yolo --output yolo_output
```

The official `yolo26m-seg.pt` is downloaded by Ultralytics when absent. Paper
settings are 1280 pixels, batch 6, 1000 optimizer-hook calls, AdamW, confidence
0.25, retina masks enabled, and the last fixed-step EMA checkpoint. The budget
is iterations, not 1000 epochs. AMP can skip actual parameter updates on overflow.
Polygons use external contours, fill holes, and discard components below 50 pixels;
this conversion is a limitation of this baseline. Only keyframes are
provided to training; validation is disabled. The paper reports this fixed-step
setting; longer 300-epoch training was examined only as a diagnostic.

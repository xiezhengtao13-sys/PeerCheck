"""Build the static web demo in docs/ from the demonstration clip in demo/.

Writes reduced-size photographs, single-channel label masks (0-4, 255 ignore)
for the SAM 3, DINOv2, consensus and ground-truth layers, and docs/demo/data.js
with the per-frame agreement and IoU values of the paper run.

Usage: python scripts/build_web_demo.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from peercheck.io import read_mask
from peercheck.metrics import evaluate
from peercheck.verification import agreement

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'demo'
OUT = ROOT / 'docs' / 'demo'
SIZE = (880, 660)
LAYERS = {'sam': DEMO / 'precomputed' / 'V3', 'dino': DEMO / 'precomputed' / 'VD',
          'consensus': DEMO / 'precomputed' / 'V8', 'gt': DEMO / 'data' / 'gt'}


def main():
    ref = pd.read_csv(DEMO / 'paper_reference.csv')
    (OUT / 'rgb').mkdir(parents=True, exist_ok=True)
    for layer in LAYERS:
        (OUT / 'masks' / layer).mkdir(parents=True, exist_ok=True)
    frames = []
    for row in ref.itertuples():
        stem = Path(row.frame_id).stem
        name = f'{row.frame_index:03d}'
        with Image.open(DEMO / 'data' / 'images' / row.sequence / row.frame_id) as im:
            im.convert('RGB').resize(SIZE, Image.Resampling.LANCZOS).save(OUT / 'rgb' / f'{name}.jpg', quality=85)
        masks = {}
        for layer, folder in LAYERS.items():
            masks[layer] = read_mask(folder / row.sequence / f'{stem}.png')
            small = Image.fromarray(masks[layer]).resize(SIZE, Image.Resampling.NEAREST)
            small.save(OUT / 'masks' / layer / f'{name}.png', optimize=True)
        if row.is_keyframe:
            # keyframes are annotated, not routed; their scores are shown for reference only
            c = agreement(masks['sam'], masks['dino'])
            iou = {k: evaluate(masks['gt'], masks[k])['foreground_iou'] for k in ('consensus', 'sam', 'dino')}
        else:
            c = row.paper_C_i
            iou = {'consensus': row.paper_V8_iou, 'sam': row.paper_V3_iou, 'dino': row.paper_VD_iou}
        frames.append({'index': int(row.frame_index), 'name': name, 'keyframe': bool(row.is_keyframe),
                       'C': round(float(c), 4), 'iou': round(float(iou['consensus']), 4),
                       'iou_sam': round(float(iou['sam']), 4), 'iou_dino': round(float(iou['dino']), 4)})
    data = {'sequence': 'Sequence 3 of the paper, frames 141 to 168',
            'size': list(SIZE), 'tau': 0.85, 'q': 0.85, 't_ann': 157, 't_rev': 7, 'frames': frames}
    (OUT / 'data.js').write_text('window.PEERCHECK_DATA = ' + json.dumps(data, indent=1) + ';\n')
    print(f'{len(frames)} frames written to {OUT}')


if __name__ == '__main__':
    main()

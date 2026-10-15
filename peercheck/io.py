"""Portable sequence discovery and indexed mask I/O."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

COLORS = np.array([[45,45,45],[0,180,230],[255,195,0],[225,70,65],[155,85,220]], dtype=np.uint8)


def discover(data):
    data = Path(data)
    frames = []
    for folder in sorted((data / 'images').iterdir()):
        if not folder.is_dir():
            continue
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
        for index, image in enumerate(images):
            mask = data / 'keyframes' / folder.name / (image.stem + '.png')
            frames.append({'sequence': folder.name, 'frame_index': index, 'frame_id': image.name,
                           'image': image, 'mask': mask if mask.exists() else None})
    if not frames:
        raise ValueError('Expected images/<sequence>/<frame>.jpg under the input directory.')
    return frames


def read_mask(path):
    with Image.open(path) as im:
        a = np.asarray(im).copy()
    if a.ndim != 2 or not np.isin(a, [0,1,2,3,4,255]).all():
        raise ValueError('Expected indexed labels 0-4 or 255.')
    return a.astype(np.uint8)


def save_mask(path, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(mask, dtype=np.uint8)
    palette = np.zeros((256,3), dtype=np.uint8)
    palette[:5] = COLORS
    palette[255] = 245
    im = Image.fromarray(a).convert('P')
    im.putpalette(palette.ravel().tolist())
    im.save(path, optimize=True)


def overlay(image, mask):
    with Image.open(image) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
    if rgb.shape[:2] != mask.shape:
        raise ValueError('Overlay image and mask dimensions must match.')
    colored = rgb.copy()
    for c in range(1,5):
        colored[mask == c] = (0.5*rgb[mask == c] + 0.5*COLORS[c]).astype(np.uint8)
    # Ignore stays visible as the original photograph.
    return Image.fromarray(colored)

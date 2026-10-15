"""Equations 1-3: nested proportional quotas and within-sequence k-center."""
import math
import numpy as np


def quota_order(lengths, total):
    lengths = np.asarray(lengths, dtype=int)
    if (lengths <= 0).any() or not len(lengths) <= total <= lengths.sum():
        raise ValueError('Budget must give every nonempty sequence at least one frame.')
    counts = np.ones(len(lengths), dtype=int)
    order = list(range(len(lengths)))
    for k in range(len(lengths) + 1, total + 1):
        deficits = k * lengths / lengths.sum() - counts
        eligible = np.flatnonzero(counts < lengths)
        winner = min(eligible, key=lambda s: (-deficits[s], -int(lengths[s]), s))
        order.append(int(winner))
        counts[winner] += 1
    return order


def select(frames, rho_percent=5, strategy='uniform', seed=0, embeddings=None):
    """Frames are sorted by (sequence, frame_index); returned indices are nested.

    `uniform` is temporal farthest-point selection, not evenly spaced linspace.
    `coverage-aware` uses cosine distance of normalized DINOv2 CLS tokens.
    The same seed selects identical initial frames for both methods.
    """
    if strategy not in ('uniform', 'coverage-aware') or not 0 < rho_percent <= 100:
        raise ValueError('Invalid strategy or ratio.')
    sequences = sorted({r['sequence'] for r in frames})
    groups = [[i for i, r in enumerate(frames) if r['sequence'] == s] for s in sequences]
    budget = math.ceil(rho_percent * len(frames) / 100)
    quotas = quota_order([len(g) for g in groups], budget)
    rng = np.random.default_rng(seed)
    starts = [int(rng.choice(g)) for g in groups]
    if strategy == 'coverage-aware':
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(frames) or not np.isfinite(embeddings).all():
            raise ValueError('One finite embedding per frame is required.')
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if (norm == 0).any():
            raise ValueError('Zero embeddings are invalid.')
        embeddings = embeddings / norm
    orders = []
    for s, group in enumerate(groups):
        positions = np.array([frames[i]['frame_index'] for i in group])
        chosen = [group.index(starts[s])]
        minimum = np.full(len(group), np.inf)
        matrix = embeddings[group] if strategy == 'coverage-aware' else None
        for _ in range(quotas.count(s)):
            c = chosen[-1]
            distance = np.abs(positions - positions[c]) if matrix is None else np.clip(1 - matrix @ matrix[c], 0, 2)
            minimum = np.minimum(minimum, distance)
            minimum[chosen] = -np.inf
            if len(chosen) < quotas.count(s):
                chosen.append(int(np.argmax(minimum)))
        orders.append([group[i] for i in chosen])
    used = [0] * len(groups)
    result = []
    for s in quotas:
        result.append(orders[s][used[s]])
        used[s] += 1
    return result


def appearance_embeddings(frames, config, device='cuda'):
    from .dino_decoder import load_encoder, normalized_rgb
    import torch
    from PIL import Image, ImageOps
    encoder = load_encoder(config, device)
    vectors = []
    with torch.inference_mode():
        for row in frames:
            with Image.open(row['image']) as im:
                image = ImageOps.exif_transpose(im).convert('RGB').resize((518, 518), Image.Resampling.BICUBIC)
                rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].to(device).float()/255
            token = encoder(pixel_values=normalized_rgb(rgb)).last_hidden_state[:, 0]
            vectors.append(torch.nn.functional.normalize(token, dim=1)[0].cpu().numpy())
    return np.stack(vectors)

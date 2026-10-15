"""Equations 9-12: semantic foreground agreement and routing."""
import numpy as np


def confusion(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or a.shape != b.shape or not a.size:
        raise ValueError('Masks must be nonempty, equally shaped 2-D arrays.')
    if not np.isin(a, [0, 1, 2, 3, 4, 255]).all() or not np.isin(b, [0, 1, 2, 3, 4, 255]).all():
        raise ValueError('Allowed labels: 0-4 and 255 (ignore).')
    encoded = np.minimum(a, 5).astype(np.int64) * 6 + np.minimum(b, 5)
    return np.bincount(encoded.ravel(), minlength=36).reshape(6, 6)


def apply_tolerance(matrix, a_min=0.0005):
    """Collapse a map's total foreground to background if 0 < area < a_min*N.

    This is a whole-map metric rule, not per-class filtering; labels stay intact.
    The strict boundary preserves a foreground area exactly equal to a_min*N.
    """
    if not 0 <= a_min <= 1:
        raise ValueError('a_min must be a fraction between zero and one.')
    m = np.asarray(matrix, dtype=np.int64).copy()
    if m.shape != (6, 6) or (m < 0).any():
        raise ValueError('Expected nonnegative 6-by-6 confusion counts.')
    threshold = a_min * m.sum()
    small_a = 0 < m[1:5, :].sum() < threshold
    small_b = 0 < m[:, 1:5].sum() < threshold
    if small_a:
        m[0, :] += m[1:5, :].sum(0)
        m[1:5, :] = 0
    if small_b:
        m[:, 0] += m[:, 1:5].sum(1)
        m[:, 1:5] = 0
    return m


def pooled_foreground_iou(a, b, a_min=0.0005):
    """Eq. 9: sum of correct class intersections / sum of class unions.

    Different foreground classes count as errors. Ignore does not erase GT
    false negatives. An empty foreground union has IoU one.
    """
    m = apply_tolerance(confusion(a, b), a_min)
    tp = np.diag(m)[1:5].sum()
    union = m[1:5, :].sum() + m[:, 1:5].sum() - tp
    return float(tp / union) if union else 1.0


def agreement(sam, dino, a_min=0.0005):
    """Eq. 10: symmetric model agreement C_i."""
    return pooled_foreground_iou(sam, dino, a_min)


def route(confidence, tau=0.85, is_keyframe=False):
    """Eq. 11: keyframes are human inputs, excluded from accept/review."""
    if not np.isfinite(confidence) or not 0 <= confidence <= 1 or not 0 <= tau <= 1:
        raise ValueError('Confidence and tau must be finite values in [0,1].')
    return 'keyframe' if is_keyframe else ('accept' if confidence >= tau else 'review')


def intersection_labels(sam, dino):
    """Eq. 12: retain equal labels (including background), else ignore=255."""
    confusion(sam, dino)
    return np.where(sam == dino, sam, 255).astype(np.uint8)

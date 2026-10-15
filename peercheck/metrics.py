"""Evaluation at the supplied mask resolution; no GT enters routing."""
import numpy as np
from .verification import confusion, pooled_foreground_iou


def evaluate(gt, prediction, a_min=0.0005):
    m = confusion(gt, prediction)
    result = {'foreground_iou': pooled_foreground_iou(gt, prediction, a_min)}
    for c in range(5):
        union = m[c, :].sum() + m[:, c].sum() - m[c, c]
        result[f'class_{c}_iou'] = float(m[c, c] / union) if union else float('nan')
    labeled = m[:, 1:5].sum()
    result['labeled_precision'] = float(np.diag(m)[1:5].sum() / labeled) if labeled else float('nan')
    result['ignore_fraction'] = float(m[:, 5].sum() / m.sum())
    return result

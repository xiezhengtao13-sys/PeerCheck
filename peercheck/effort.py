"""Equation 13: manual effort from the measured mean times per frame."""


def effort(total, keyframes, review, reannotated=0, t_ann=157.0, t_rev=7.0):
    """E = (|K| + |R_f|) t_ann + |R| t_rev.

    `review` counts every reviewed frame, each checked once; `reannotated` counts
    the reviewed frames whose label is rejected and annotated again from scratch.
    The defaults are the mean times measured on the paper's dataset: 157 s to
    annotate one frame and 7 s to check one automatic label.
    """
    if total <= 0 or min(keyframes, review, reannotated) < 0 or keyframes + review > total \
            or reannotated > review:
        raise ValueError('Counts must partition the dataset; re-annotated frames are reviewed frames.')
    seconds = (keyframes + reannotated) * t_ann + review * t_rev
    return {'seconds': seconds, 'hours': seconds / 3600,
            'relative_to_full_annotation': seconds / (total * t_ann),
            'accepted': total - keyframes - review}

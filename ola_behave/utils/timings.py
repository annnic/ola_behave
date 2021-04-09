import numpy as np


def infer_tv(fps, filechunk):
    """Takes fps and the length of the video to compute an inferred time vector that starts from 0
    >>> infer_tv(1, 3)
    array([0., 1., 2.])
    """
    tv = np.arange(0, filechunk, 1 / fps)
    return tv

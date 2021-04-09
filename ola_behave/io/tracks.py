import numpy as np


def load_track(csv_file_path):
    """Takes file path, loads  the csv track, computes speed from this, returns  both
    """
    track_internal = np.genfromtxt(csv_file_path, delimiter=',')

    if track_internal.size == 0:
        # if empty return empty
        displacement_internal = []
        track_internal = []
        return displacement_internal, track_internal
    else:
        # find displacement
        b = np.diff(track_internal[:, 1])
        c = np.diff(track_internal[:, 2])
        displacement_internal = np.sqrt(b ** 2 + c ** 2)
        return displacement_internal, track_internal

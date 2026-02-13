import numpy as np


def triangle_area(a, b, c):
    """
    """
    lab = np.linalg.norm(a - b, axis=-1)
    lac = np.linalg.norm(a - c, axis=-1)
    lbc = np.linalg.norm(b - c, axis=-1)
    s = 0.5*(lab + lac + lbc)
    area = (s*(s - lab)*(s - lac)*(s - lbc))**0.5
    return area
import itertools

import numpy as np




# chatgpt generated function
def merge_meshes_with_weld(V0, F0, V1, F1, tol=1e-9):
    """
    Объединяет 2 меша и 'сваривает' совпадающие вершины.
    V*: (N,3) float, F*: (M,3) int (0-based).
    tol: геом. допуск (в тех же единицах, что и координаты).

    Возвращает: V_merged, F_merged, (map0, map1)
      map0: отображение старых индексов V0 -> новых индексов
      map1: отображение старых индексов V1 -> новых индексов
    """
    # 1) складываем вершины в один массив
    V = np.vstack([V0, V1])

    # 2) квантизация (устойчивая сварка для float)
    #    ключ = округление координат к сетке tol
    if tol <= 0:
        raise ValueError("tol must be > 0")
    Q = np.round(V / tol).astype(np.int64)

    # 3) unique по строкам: получаем новые вершины и отображение old->new
    #    inv[i] = индекс уникальной вершины для старой вершины i
    uniqQ, uniq_idx, inv = np.unique(Q, axis=0, return_index=True, return_inverse=True)

    V_merged = V[uniq_idx]

    # 4) перенумеровываем грани
    F0_new = inv[F0]
    F1_shifted = F1 + len(V0)
    F1_new = inv[F1_shifted]

    F_merged = np.vstack([F0_new, F1_new])

    # (опционально) убрать вырожденные треугольники, которые могли появиться после сварки
    # например, если два или три индекса в треугольнике совпали
    good = (F_merged[:,0] != F_merged[:,1]) & (F_merged[:,1] != F_merged[:,2]) & (F_merged[:,0] != F_merged[:,2])
    F_merged = F_merged[good]

    map0 = inv[:len(V0)]
    map1 = inv[len(V0):]

    return V_merged, F_merged, (map0, map1)



def rotate_over_x(points, angle):
    """
    Rotates the cloud of points over the axis 0

    Parameters:
    -----------
    points: np.array shape (..., 3)

    angle: float
    """
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s,  c]])
    return points @ R.T


def rotate_over_y(points, angle):
    """
    Rotates the cloud of points over the axis 1

    Parameters:
    -----------
    points: np.array shape (..., 3)

    angle: float
    """
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
    return points @ R.T


def rotate_over_z(points, angle):
    """
    Rotates the cloud of points over the axis 2

    Parameters:
    -----------
    points: np.array shape (..., 3)

    angle: float
    """
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    return points @ R.T


def get_torus_vertices(phi, psi, R=2, r=1):
    """
    Returns the coordinates of the torus based on the 2 angle arrays

    Pramaneters:
    phi: float np.array in [0, 2pi] length n

    psi: float np.array in [0, 2pi] length n
    
    r: float
        ...

    R: float
        ...

    Returns:
    --------
    vertices: np.array shape (n, 3)

    """
    x = (R + r*np.cos(psi)) * np.cos(phi)
    y = (R + r*np.cos(psi)) * np.sin(phi)
    z = r * np.sin(psi)
    vertices = np.column_stack([x, y, z])
    return vertices

def get_torus_faces(n, m):
    """

    """
    faces = []
    for x, y, in itertools.product(range(n), range(m)):
        v00 = m*x + y
        v01 = m*((x + 1)%n) + y
        v10 = m*x + (y + 1)%m
        v11 = m*((x + 1)%n) + (y + 1)%m
        if (x + y)%2 == 0:
            faces.append([v00, v01, v11])
            faces.append([v00, v10, v11])
        else:
            faces.append([v00, v01, v10])
            faces.append([v01, v10, v11])
    
    faces = np.array(faces)
    return faces
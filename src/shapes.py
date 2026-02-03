import itertools
import numpy as np
import scipy as sp


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
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s,  c]])
    return points @ R.T

def rotate_over_y(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
    return points @ R.T

def rotate_over_z(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    return points @ R.T

def get_torus_vertices(phi, psi, R=2, r=1):
    """
    """
    x = (R + r*np.cos(psi)) * np.cos(phi)
    y = (R + r*np.cos(psi)) * np.sin(phi)
    z = r * np.sin(psi)
    return np.column_stack([x, y, z])


def get_torus_triangulation(n, m):
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


def get_torus(n, m, R=2, r=1, glue_out=True, glue_in=False, tol=1e-6):
    """
    """
    phi = np.arange(n)/n*2*np.pi
    psi = np.arange(m)/m*2*np.pi - np.pi
    phi = np.repeat(phi, m)
    psi = np.tile(psi, n)

    vertices = get_torus_vertices(phi, psi, R, r)
    faces = get_torus_triangulation(n, m)

    if glue_out:
        disk_border_idx = np.argwhere(abs(np.linalg.norm(vertices, axis=1) - r) < tol).ravel()
        disk_border_pts = vertices[disk_border_idx][:, [0, 1]]
        disk_triangulation_local = sp.spatial.Delaunay(disk_border_pts, furthest_site=True, qhull_options="QJ").simplices
        disk_triangulation_global = disk_border_idx[disk_triangulation_local]
        faces = np.vstack([faces, disk_triangulation_global])
    if glue_in:
        disk_border_idx = np.argwhere((abs(vertices[:, 1]) <= 1e-5) & (vertices[:, 0] > 0)).ravel()
        disk_border_pts = vertices[disk_border_idx][:, [0, 2]]
        disk_triangulation_local = sp.spatial.Delaunay(disk_border_pts, furthest_site=True, qhull_options="QJ").simplices
        disk_triangulation_global = disk_border_idx[disk_triangulation_local]
        faces = np.vstack([faces, disk_triangulation_global])

    return vertices, faces


def get_couple_linked_tori(n0, n1, r0=0.5, r1=0.5):
    """
    """
    alpha0 = 2*np.pi/n0
    alpha1 = 2*np.pi/n1
    phi0 = alpha0*np.arange(n0)
    phi1 = alpha1*np.arange(n1)

    circle0 = np.column_stack([-r0*np.cos(phi0) + r0, r0*np.sin(phi0), np.zeros(n0)])
    circle1 = np.column_stack([+r1*np.cos(phi1) - r1, np.zeros(n1), r1*np.sin(phi1)])

    torus0_vertices = np.concatenate([rotate_over_y(circle0 + np.array([r1, 0, 0]), i*alpha1) - np.array([r1, 0, 0]) for i in range(n1)])
    torus1_vertices = np.concatenate([rotate_over_z(circle1 + np.array([-r0, 0, 0]), i*alpha0) + np.array([r0, 0, 0])for i in range(n0)])

    torus0_faces = get_torus_triangulation(n1, n0)
    torus1_faces = get_torus_triangulation(n0, n1)

    linked_tori_vertices, linked_tori_faces, (map0, map1) = merge_meshes_with_weld(torus0_vertices, torus0_faces, torus1_vertices, torus1_faces, tol=1e-8)

    return linked_tori_vertices, linked_tori_faces


def get_halftorus(r=1, R=2, l0=1, l1=2, n=24, m=36, glue=True, tol=1e-6):
    """
    """
    phi = np.arange(n)/(n - 1)*np.pi
    psi = np.arange(m)/m*2*np.pi

    vertices_torus = []
    for p in phi:
        for s in psi:
            x = (R + r * np.cos(s)) * np.cos(p)
            y = (R + r * np.cos(s)) * np.sin(p)
            z = r * np.sin(s)
            vertices_torus.append((x, y, z))

    faces_torus = []
    for i in range(n - 1):
        for j in range(m):
            next_i = (i + 1) % n
            next_j = (j + 1) % m
            v0 = i*m + j
            v1 = next_i*m + j
            v2 = next_i*m + next_j
            v3 = i*m + next_j
            if (i + j) % 2 == 0:
                faces_torus.append([v0, v1, v2])
                faces_torus.append([v0, v2, v3])
            else:
                faces_torus.append([v1, v2, v3])
                faces_torus.append([v0, v1, v3])

    vertices = np.array(vertices_torus)
    faces = np.array(faces_torus)

    # Продлеваю край (геометрия для phi=0)
    x = np.cos(psi)
    z = np.sin(psi)
    y = -(np.abs(z)*l0 + (1 - np.abs(z))*l1)
    x = r*x + R
    z = r*z

    vertices_extra0 = np.stack([ x, y, z], axis=1)   # для phi=0
    vertices_extra1 = np.stack([-x, y, z], axis=1)   # для phi=pi (зеркалим по x)

    base0 = n * m
    base1 = base0 + m

    # индексы колец края
    edge0 = 0 * m
    edge1 = (n - 1) * m

    j  = np.arange(m)
    jn = (j + 1) % m

    # --- лента для phi=0 ---
    v_edge_j0  = edge0 + j
    v_edge_jn0 = edge0 + jn
    v_ex_j0    = base0 + j
    v_ex_jn0   = base0 + jn

    faces_extra0 = np.vstack([
        np.stack([v_edge_j0,  v_ex_j0,  v_ex_jn0], axis=1),
        np.stack([v_edge_j0,  v_ex_jn0, v_edge_jn0], axis=1),
    ])

    # --- лента для phi=pi ---
    v_edge_j1  = edge1 + j
    v_edge_jn1 = edge1 + jn
    v_ex_j1    = base1 + j
    v_ex_jn1   = base1 + jn

    faces_extra1 = np.vstack([
        np.stack([v_edge_j1,  v_ex_j1,  v_ex_jn1], axis=1),
        np.stack([v_edge_j1,  v_ex_jn1, v_edge_jn1], axis=1),
    ])

    # собираем
    vertices_extra = np.vstack([vertices_extra0, vertices_extra1]).astype(float)
    faces_extra    = np.vstack([faces_extra0, faces_extra1]).astype(int)

    vertices = np.concatenate([vertices, vertices_extra], axis=0)
    faces    = np.concatenate([faces, faces_extra], axis=0)


    if glue:
        # glue the hole
        disk_border_idx = np.argwhere(abs(np.linalg.norm(vertices, axis=1) - r) < tol).ravel()
        disk_border_idx = np.append(disk_border_idx, n*m + m//2)
        disk_border_idx = np.append(disk_border_idx, n*m + m + m//2)
        
        disk_border_pts = vertices[disk_border_idx]
        disk_border_pts = disk_border_pts[:, [0, 1]]

        disk_triangulation_local = sp.spatial.Delaunay(disk_border_pts, furthest_site=False, qhull_options="QJ").simplices
        disk_triangulation_global = disk_border_idx[disk_triangulation_local]
        faces = np.vstack([faces, disk_triangulation_global])


    vertices[:, 1] += l1

    return vertices, faces


def get_halftori_bouquet(leaves=3, r=1, R=2, l0=1, n=6, m=6, glue=True, tol=1e-6):
    l1 = l0 + r/np.tan(np.pi/leaves)

    vertices0, faces0 = get_halftorus(r, R, l0, l1, n, m, glue, tol)
    vertices, faces = vertices0.copy(), faces0.copy()
    for i in range(1, leaves):
        angle = i * (2 * np.pi / leaves)
        rotated_vertices = rotate_over_x(vertices0, angle)
        vertices, faces, _ = merge_meshes_with_weld(vertices, faces, rotated_vertices, faces0, tol)
    return vertices, faces

def split_edge(vertices, faces, e0, e1):
    """
    """
    new_vertex = 0.5*(vertices[e0] + vertices[e1])
    new_index = len(vertices)
    new_vertices = np.concatenate([vertices, [new_vertex]])

    faces_cond = (faces == e0).any(axis=1) & (faces == e1).any(axis=1)
    updating_simplices_vertices = np.unique(faces[faces_cond])
    updating_simplices_vertices = updating_simplices_vertices[~np.isin(updating_simplices_vertices, [e0, e1])]
    new_faces = np.concatenate([faces[~faces_cond], 
                                [(e0, new_index, v) for v in updating_simplices_vertices], 
                                [(e1, new_index, v) for v in updating_simplices_vertices]])
    return new_vertices, new_faces


def split_large_edges(vertices, faces, max_length=1.0):
    """
    """
    new_vertices, new_faces = vertices.copy(), faces.copy()

    def get_max_edge_length(V, F):
        edges = np.concatenate([F[:, [0, 1]], F[:, [0, 2]], F[:, [1, 2]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        edges_lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
        max_length_idx = np.argmax(edges_lengths)
        e0, e1 = edges[max_length_idx]
        l = edges_lengths[max_length_idx]
        return e0, e1, l
    
    e0, e1, l = get_max_edge_length(new_vertices, new_faces)
    while l > max_length:
        new_vertices, new_faces = split_edge(new_vertices, new_faces, e0, e1)
        e0, e1, l = get_max_edge_length(new_vertices, new_faces)
    return new_vertices, new_faces


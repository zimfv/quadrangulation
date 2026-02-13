import itertools
import numpy as np
import networkx as nx


def compact_mesh(V, F):
    """
    Remove vertices not referenced by any face and reindex faces.
    Returns V2, F2, old2new, new2old.
    """
    V = np.asarray(V)
    F = np.asarray(F)

    used = np.zeros(len(V), dtype=bool)
    used[F.reshape(-1)] = True

    new2old = np.nonzero(used)[0]
    old2new = -np.ones(len(V), dtype=int)
    old2new[new2old] = np.arange(len(new2old))

    V2 = V[new2old]
    F2 = old2new[F]

    return V2, F2, old2new, new2old


def count_new_vertices(faces, new_face):
    """
    """
    vertices = np.unique(faces)
    cnt_vertices = np.sum(~np.isin(new_face, vertices))
    return cnt_vertices


def count_new_edges(faces, new_face):
    """
    """
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])
    edges = np.sort(edges, axis=1)

    new_edges = np.array([
        [new_face[0], new_face[1]],
        [new_face[1], new_face[2]],
        [new_face[2], new_face[0]]
    ])
    new_edges = np.sort(new_edges, axis=1)

    matches = (edges[:, None] == new_edges).all(axis=2)
    edge_exists = matches.any(axis=0)

    cnt_edges = np.sum(~edge_exists)

    return cnt_edges


def is_homotopy_preserving_face_addition(faces, new_face):
    """
    """
    if (np.sort(faces, axis=1) == np.sort(new_face)).all(axis=1).any():
        return False
    cnt_v = count_new_vertices(faces, new_face)
    if cnt_v == 3:
        return False
    cnt_e = count_new_edges(faces, new_face)
    return cnt_e - cnt_v == 1

def get_faces_components(faces):
    """
    """
    if (faces.ndim != 2) or (faces.shape[-1] != 3):
        raise ValueError(f'Expected faces be shape (n, 3)')
    g = nx.Graph()
    g.add_nodes_from(range(faces.shape[0]))
    g.add_edges_from([(i, j) for i, j in itertools.combinations(range(faces.shape[0]), 2) if np.intersect1d(faces[i], faces[j]).size == 2])
    return  list(map(lambda i: faces[list(i)], nx.connected_components(g)))
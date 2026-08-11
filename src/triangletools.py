import itertools
import numpy as np
import networkx as nx
import scipy as sp
import igl


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


def get_neighborhood_graph(faces, center, with_center=False):
    """
    """
    neighborhood_faces = faces[(faces == center).any(axis=1)]
    neighborhood_edges = neighborhood_faces[neighborhood_faces != center].reshape(-1, 2)
    neighborhood_graph = nx.Graph()
    neighborhood_graph.add_edges_from(neighborhood_edges)
    if with_center:
        neighborhood_graph.add_edges_from([(center, node) for node in neighborhood_graph.nodes()])
    return neighborhood_graph


def merging_paths_intersects(path0, path1, faces):
    """
    Returns True if 2 merging edge paths intersect 
    """
    if np.intersect1d(path0, path1).size == 0:
        raise ValueError(f'Paths does not merge.')
    if len(np.unique([path0[[0, -1]], path1[[0, -1]]])) != 4:
        raise ValueError(f'Paths does not merge in the middle.')

    # detect the disk arround paths
    faces_new = faces[np.isin(faces, np.concatenate([path0, path1])).any(axis=1)]
    #faces_new = faces[(np.isin(faces, np.concatenate([path0, path1])).sum(axis=1) > 1) | np.isin(faces, np.concatenate([path0[1:-1], path1[1:-1]])).any(axis=1)]
    faces_new = faces_new[~((np.isin(faces_new, np.concatenate([path0, path1])).sum(axis=1) == 1) & np.isin(faces_new, np.concatenate([path0[[0, -1]], path1[[0, -1]]])).any(axis=1)) ]
    
    edges = np.concatenate(faces_new[:, [[0, 1], [0, 2], [1, 2]]])

    g = nx.Graph()
    g.add_edges_from(edges)

    #import matplotlib.pyplot as plt
    #fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    #fig.suptitle(f'path0: {path0}\npath1: {path1}')
    #nx.draw_networkx(g, with_labels=True, pos=nx.kamada_kawai_layout(g), ax=axs[0])

    # cut the disk by path0
    g.remove_nodes_from(path0)

    print('path0:', path0)
    print('path1:', path1)

    #nx.draw_networkx(g, with_labels=True, pos=nx.kamada_kawai_layout(g), ax=axs[1])
    #plt.show()

    # check if the ends of path1 in one connected component
    try:
        shortest_path = nx.shortest_path(g, path1[0], path1[-1])
        print('Intersect', False, 'shortest_path:', shortest_path)
        return False
    except nx.NetworkXNoPath:
        print('Intersect', True)
        return True


def face_adjacency(faces, restricted_edges=None):
    """
    Returns the face adjacency matrix, where 2 faces are adjacent if they share a common not restricted edge

    Parameters:
    -----------
    faces: np.array shape(n, 3)
        The faces of the triangulation

    restricted_edges: np.array shape (k, 2)
        The list of restricted edges

    Returns:
    --------
    A : sp.sparse.csr_matrix
        Face adjacency matrix
    """
    if restricted_edges is None:
        restricted_edges = np.zeros([0, 2], dtype=int)
    n = len(faces)

    restricted_edges = np.sort(restricted_edges, axis=-1)

    rows = []
    cols = []
    for (i0, face0), (i1, face1) in itertools.combinations(enumerate(faces), 2):
        if np.intersect1d(face0, face1).size == 2:
            if not (restricted_edges == np.intersect1d(face0, face1)).all(axis=-1).any():
                rows.extend([i0, i1])
                cols.extend([i1, i0])

    A = sp.sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(n, n),
    )
    return A


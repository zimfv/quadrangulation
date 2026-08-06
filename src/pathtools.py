import numpy as np
import scipy as sp
import itertools
import igl




def get_path_sides(path, faces):
    """
    Returns sides of the path

    Returns:
    --------
    sides: list[np.array[int]]
        The indices of faces from each side
    """
    path_edges = np.sort(np.transpose([path[:-1], path[1:]]), axis=1)
    touching_faces_indices = np.argwhere((faces[..., None] == path[1:-1]).any(axis=1).any(axis=1)).ravel()
    touching_faces = faces[touching_faces_indices]
    n = len(touching_faces)

    TT, TTi = igl.triangle_triangle_adjacency(touching_faces)
    face_adjacency_matrix = sp.sparse.csc_matrix((n, n), dtype=int)
    for i, row in enumerate(TT):
        for j in row:
            if not ((np.array([i, j]) == path_edges).all(axis=-1).any() and (np.array([j, i]) == path_edges).all(axis=-1).any()):
                face_adjacency_matrix[i, j] = 1
    n_components, labels = sp.sparse.csgraph.connected_components(csgraph=face_adjacency_matrix, directed=False, return_labels=True)
    sides = [touching_faces_indices[labels == label] for label in range(n_components)]
    return sides


def paths_intersect(path0, path1, faces):
    """
    """
    sides0 = get_path_sides(path0, faces)
    sides1 = get_path_sides(path1, faces)
    if (len(sides0) < 2) or (len(sides1) < 2):
        return False
    intersection_matrix = np.array([[np.intersect1d(side0, side1).size for side0 in sides0] for side1 in sides1])
    return (intersection_matrix != 0).all()


def merge_paths_at_nodes(paths, nodes=[]):
    """
    """
    node_path_indices = [[i for i, path in enumerate(paths) if (path[[0, -1]] == node).any()] for node in nodes]
    node_touching_counts = np.array([len(i) for i in node_path_indices])
    if (node_touching_counts < 2).all():
        return paths

    node = nodes[node_touching_counts >= 2][0]
    node_path_indices = node_path_indices[np.argwhere(node_touching_counts >= 2).ravel()[0]]

    new_paths = [path for i, path in enumerate(paths) if i not in node_path_indices]
    paths_to_merge = [path for i, path in enumerate(paths) if i in node_path_indices]

    for path0, path1 in itertools.combinations(paths_to_merge, 2):
        if path0[-1] != node:
            path0 = np.array(path0)[::-1]
        if path1[0] != node:
            path1 = np.array(path1)[::-1]
            new_paths.append(np.concatenate([path0, path1]))
    
    return merge_paths_at_nodes(new_paths, nodes)
    
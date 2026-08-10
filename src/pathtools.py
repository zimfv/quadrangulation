import numpy as np
import scipy as sp
import itertools
import igl

from pygeodesic import geodesic

from src import triangletools




def get_path_sides(path, faces):
    """
    Returns sides of the path

    Returns:
    --------
    sides: list[np.array[int]]
        The indices of faces from each side
    """
    
    #touching_faces_indices = np.argwhere((faces[..., None] == path[1:-1]).any(axis=1).any(axis=1)).ravel()  
    touching_faces_indices = np.argwhere((faces[..., None] == path[1:-1]).any(axis=1).any(axis=1) | \
                                         ((faces[..., None] == path[1:]).any(axis=1) & (faces[..., None] == path[:-1]).any(axis=1)).any(axis=1)).ravel()

    touching_faces = faces[touching_faces_indices]
    
    face_adjacency_matrix = triangletools.face_adjacency(touching_faces, restricted_edges=np.transpose([path[:-1], path[1:]]))

    n_components, labels = sp.sparse.csgraph.connected_components(csgraph=face_adjacency_matrix, directed=False, return_labels=True)

    sides = [touching_faces_indices[labels == label] for label in range(n_components)]
    return sides


def paths_intersect(path0, path1, faces):
    """
    """
    if np.intersect1d(path0, path1).size == 0:
        return False
    sides0 = get_path_sides(path0, faces)
    sides1 = get_path_sides(path1, faces)
    if (len(sides0) < 2) or (len(sides1) < 2):
        return False
    intersection_matrix = np.array([[np.intersect1d(side0, side1).size for side0 in sides0] for side1 in sides1])
    return (intersection_matrix != 0).all()


def concatenate_paths(path0, path1):
    """
    """
    if path0[0] == path1[0]:
        return np.concatenate([path0[::-1], path1])
    if path0[-1] == path1[-1]:
        return np.concatenate([path0, path1[::-1]])
    if path0[-1] == path1[0]:
        return np.concatenate([path0, path1])
    if path0[0] == path1[-1]:
        return np.concatenate([path1, path0])
    raise ValueError('Paths are inconcatenable.')
        


def merge_paths_at_nodes(paths, nodes=[]):
    """
    """
    # FIXME: Does not add new merged paths (also does not remove paths used to merge)
    n_paths = len(paths)
    paths_indices_in_nodes = [[i for i in range(n_paths) if (paths[i][[1, -1]] == node).any()] for node in nodes]
    print(nodes, paths_indices_in_nodes)
    merged_in_nodes = [
        concatenate_paths(paths[i], paths[j])
        for paths_indices_in_node in paths_indices_in_nodes
        for i, j in itertools.combinations(paths_indices_in_node, 2)
    ]

    new_paths = paths + merged_in_nodes
    return new_paths



def get_path_close_geodesic(path, faces, vertices):
    """
    """
    if len(path) == 2:
        return vertices[path]

    sides = get_path_sides(path, faces)

    face_indices = np.concatenate(sides)

    V, F, old2new, new2old = triangletools.compact_mesh(vertices, faces[face_indices])
    source_vid = old2new[path[0]]
    target_vid = old2new[path[-1]]

    geo = geodesic.PyGeodesicAlgorithmExact(V, F)
    geo_distance, geopath = geo.geodesicDistance(source_vid, target_vid)

    return geopath
    
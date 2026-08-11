import numpy as np


def get_boundary_edges(faces):
    """
    """
    face_edges = np.concatenate([faces[:, j] for j in [[0, 1], [0, 2], [1, 2]]])
    face_edges = np.sort(face_edges, axis=1)
    edges = np.unique(face_edges, axis=0)
    edges = edges[(edges[:, None, :] == face_edges).all(axis=-1).sum(axis=-1) != 2]
    return edges


def get_conic_vertices(faces):
    """
    """
    conic_vertices = []
    for vertex in np.unique(faces):
        faces_vertex = faces[(faces == vertex).any(axis=1)]
        connections = faces_vertex[:, None, :, None] == faces_vertex[None, :, None, :]
        connections = connections.any(axis=-1).sum(axis=-1) == 2
        n_components, _ = sp.sparse.csgraph.connected_components(connections, directed=False)
        if n_components > 1:
            conic_vertices.append(vertex)
    conic_vertices = np.array(conic_vertices)
    return conic_vertices

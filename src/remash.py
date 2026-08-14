import numpy as np
import igl

from src.shapes import split_edge

def flip_ears_defining_boundary(faces):
    """
    """
    ears, ear_opp = igl.ears(faces)

    mask = ear_opp.reshape(-1, 1) != np.arange(3)
    edges_to_flip = faces[ears][mask].reshape(-1, 2)

    faces_contains_fliping_edges = (faces[:, np.array(list(itertools.combinations(range(3), 2)))][..., None, :] == edges_to_flip).all(axis=-1).any(axis=-2)
    assert (faces_contains_fliping_edges.sum(axis=0) == 2).all()
    _face_idx, _edge_idx = np.where(faces_contains_fliping_edges)
    faces_fliping_edges = _face_idx.reshape(-1, 2)
    faces_contains_fliping_edges = faces_contains_fliping_edges.any(axis=1)

    opposite_vertives = faces[faces_fliping_edges].reshape(-1, 6)
    mask = (opposite_vertives[:, None, :] != edges_to_flip[:, :, None]).all(axis=-2)
    opposite_vertives = opposite_vertives[mask].reshape(-1, 2)
    
    new_faces = np.vstack([np.hstack([opposite_vertives, edges_to_flip[:, i].reshape(-1, 1)]) for i in range(2)])
    new_faces = np.vstack([faces[~faces_contains_fliping_edges], new_faces])
    new_faces = np.sort(new_faces, axis=1)
    
    return new_faces


def flip_ears(faces, boundary_edges=None):
    """
    """
    if boundary_edges is None:
        return flip_ears_defining_boundary(faces)
    pass


def split_edges_arroud_saddles(vertices, faces, saddles, edge_choice_strategy='longest'):
    """
    """
    bnd_edges, bnd_face_indices, bnd_local_edge_indices = igl.boundary_facets(faces)
    #assert np.isin(saddles, np.unique(bnd_edges)).all()
    bnd_vertices = np.unique(bnd_edges)

    bnd_vertices_nonsaddles = bnd_vertices[~np.isin(bnd_vertices, saddles)]

    adj_full = igl.adjacency_matrix(faces)
    
    saddle_neighbors = np.argwhere(adj_full[saddles])
    saddle_neighbor_is_far_from_boundary = (adj_full[:, bnd_vertices_nonsaddles][saddle_neighbors[:, 1]].toarray() == 0).all(axis=1)

    good_saddles = saddles[saddle_neighbors[:, 0][saddle_neighbor_is_far_from_boundary]]
    

    if set(good_saddles) == set(saddles):
        return vertices, faces
    
    if isinstance(edge_choice_strategy, str):
        if edge_choice_strategy == 'longest':
            f_edge_choice_strategy = lambda edges: np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1).argmax()
        if edge_choice_strategy == 'shortest':
            f_edge_choice_strategy = lambda edges: np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1).argmin()
    else:
        f_edge_choice_strategy = edge_choice_strategy

    saddles_to_split_arround = saddles[~np.isin(saddles, good_saddles)]

    new_vertices, new_faces = vertices.copy(), faces.copy()
    for saddle in saddles_to_split_arround:
        saddle_faces = faces[(faces == saddle).any(axis=1)]
        mask = saddle_faces != saddle
        saddle_edges = saddle_faces[mask].reshape(-1, 2)

        e0, e1 = saddle_edges[f_edge_choice_strategy(saddle_edges)]
        new_vertices, new_faces = split_edge(new_vertices, new_faces, e0, e1)
    return split_edges_arroud_saddles(new_vertices, new_faces, saddles, edge_choice_strategy=edge_choice_strategy)
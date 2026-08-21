import itertools
import numpy as np
import networkx as nx
import scipy as sp
import igl

from collections import defaultdict


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


def cut_mesh_along_edges(V, F, cut_edges):
    """
    Cut a triangle mesh along specified edges.

    Parameters
    ----------
    V : (n, 3) array
    F : (m, 3) int array
    cut_edges : iterable of (i, j)
        Edges in the indexing of V. Orientation does not matter.

    Returns
    -------
    SV : (n_new, 3) array
        Vertices after cutting.
    SF : (m, 3) int array
        Faces after cutting.
    SVI : (n_new,) int array
        SVI[new_vertex] = original vertex in V.
    """

    cut_edges = {
        (min(a, b), max(a, b))
        for a, b in cut_edges
    }

    nV = len(V)
    nF = len(F)

    # Faces incident to every vertex
    vertex_faces = [[] for _ in range(nV)]

    for fi, face in enumerate(F):
        for v in face:
            vertex_faces[v].append(fi)

    # For each vertex v, construct adjacency between incident faces.
    #
    # Two faces are connected around v if they share an edge containing v
    # AND that edge is not a cut edge.
    face_neighbors_at_vertex = [
        defaultdict(set) for _ in range(nV)
    ]

    # edge -> incident faces
    edge_faces = defaultdict(list)

    for fi, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            e = (min(u, v), max(u, v))
            edge_faces[e].append(fi)

    for (u, v), faces in edge_faces.items():

        # Skeleton edge: don't connect triangle fans through it
        if (u, v) in cut_edges:
            continue

        if len(faces) == 2:
            f0, f1 = faces

            face_neighbors_at_vertex[u][f0].add(f1)
            face_neighbors_at_vertex[u][f1].add(f0)

            face_neighbors_at_vertex[v][f0].add(f1)
            face_neighbors_at_vertex[v][f1].add(f0)

    # new_index[(old_vertex, face)] gives the copy of old_vertex
    # used by that particular face.
    new_index = {}

    new_vertices = []
    SVI = []

    for v in range(nV):

        incident = vertex_faces[v]

        if not incident:
            continue

        unvisited = set(incident)

        while unvisited:

            seed = unvisited.pop()
            component = [seed]
            stack = [seed]

            while stack:
                f = stack.pop()

                for g in face_neighbors_at_vertex[v].get(f, ()):
                    if g in unvisited:
                        unvisited.remove(g)
                        stack.append(g)
                        component.append(g)

            # One vertex copy for this connected triangle fan
            new_v = len(new_vertices)

            new_vertices.append(V[v])
            SVI.append(v)

            for fi in component:
                new_index[(v, fi)] = new_v

    # Rewrite faces
    SF = np.empty_like(F)

    for fi, face in enumerate(F):
        for j, v in enumerate(face):
            SF[fi, j] = new_index[(v, fi)]

    SV = np.asarray(new_vertices)
    SVI = np.asarray(SVI, dtype=np.int64)

    return SV, SF, SVI


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


'''
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
'''


def face_adjacency(faces, restricted_edges=None):
    """
    Returns the face adjacency matrix, where two faces are adjacent
    if they share a common non-restricted edge.

    Parameters
    ----------
    faces : np.ndarray, shape (n, 3)
        Triangle vertex indices.

    restricted_edges : np.ndarray, shape (k, 2), optional
        Edges across which faces should not be considered adjacent.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (n, n)
        Symmetric face adjacency matrix.
    """
    faces = np.asarray(faces, dtype=int)
    n = len(faces)

    if restricted_edges is None:
        restricted_edges = np.empty((0, 2), dtype=int)
    else:
        restricted_edges = np.asarray(restricted_edges, dtype=int)

    # Canonical orientation for restricted edges.
    restricted_edges = np.sort(restricted_edges, axis=1)
    restricted = {tuple(edge) for edge in restricted_edges}

    # edge -> list of incident face indices
    edge_to_faces = {}

    for face_idx, (a, b, c) in enumerate(faces):
        edges = (
            (min(a, b), max(a, b)),
            (min(b, c), max(b, c)),
            (min(c, a), max(c, a)),
        )

        for edge in edges:
            if edge not in restricted:
                edge_to_faces.setdefault(edge, []).append(face_idx)

    rows = []
    cols = []

    for incident_faces in edge_to_faces.values():
        # Normal manifold case: an edge belongs to two faces.
        # This also handles non-manifold edges with >2 incident faces.
        for i in range(len(incident_faces)):
            for j in range(i + 1, len(incident_faces)):
                f0 = incident_faces[i]
                f1 = incident_faces[j]

                rows.extend((f0, f1))
                cols.extend((f1, f0))

    data = np.ones(len(rows), dtype=np.uint8)

    return sp.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n, n),
        dtype=np.uint8,
    )

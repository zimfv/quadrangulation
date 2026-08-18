import numpy as np
import scipy as sp


import itertools
from functools import cached_property, cache

from src import mesh_topology, triangletools




class Stratified:
    def __init__(self, faces, vertices, with_conic_vertices: bool=True):
        """
        """
        self.faces = np.unique(np.sort(faces, axis=1), axis=0)
        self.vertices = np.array(vertices)

        self.skeleton_graph = mesh_topology.get_skeleton_graph(self.faces, with_boundary_edges=True, with_conic_vertices=with_conic_vertices)        

        face_adjacency_without_skeleton = triangletools.face_adjacency(self.faces, restricted_edges=self.skeleton_graph.edges())
        self.n_strats, self.strats_labels = sp.sparse.csgraph.connected_components(face_adjacency_without_skeleton)

        pass
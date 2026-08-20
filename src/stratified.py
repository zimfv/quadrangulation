import numpy as np
import scipy as sp
import igl


import itertools
from functools import cached_property, cache

from src import mesh_topology, triangletools
from src import skeleton_function



class Stratified:
    def __init__(self, faces, vertices, with_conic_vertices: bool=True):
        """
        """
        self.faces = np.unique(np.sort(faces, axis=1), axis=0)
        self.vertices = np.array(vertices)

        self.skeleton_graph = mesh_topology.get_skeleton_graph(self.faces, with_boundary_edges=True, with_conic_vertices=with_conic_vertices)        

        face_adjacency_without_skeleton = triangletools.face_adjacency(self.faces, restricted_edges=self.skeleton_graph.edges())
        self.n_strats, self.strats_labels = sp.sparse.csgraph.connected_components(face_adjacency_without_skeleton)

    @property
    def n_vertices(self):
        """
        """
        return len(self.vertices)

    @property
    def n_faces(self):
        return len(self.faces)


    @cached_property
    def edges(self):
        """
        """
        return np.unique(np.concatenate(self.faces[:, [[0, 1], [0, 2], [1, 2]]]), axis=0)

    @property
    def n_edges(self):
        """
        """
        return len(self.edges)


    @property
    def skeleton_indices(self):
        """
        """
        return np.array(list(self.skeleton_graph.nodes()))

    @property
    def skeleton_edges(self):
        """
        """
        return np.array(list(self.skeleton_graph.edges()))


    def define_values_on_skeleton(self, skeleton_values_function=skeleton_function.get_skeleton_values, **kwargs):
        """
        """
        skeleton_values = skeleton_values_function(self.skeleton_graph, self.vertices, **kwargs)

        self.values = np.zeros(len(self.vertices))
        self.values[self.skeleton_indices] = skeleton_values
        return self.values

    
    @cache
    def get_stratum_parameters(self, stratum_label):
        """
        """
        # FIXME: This use a exponential memory by some reason
        # TODO: There should be an another solution with igl.split_nonmanifold

        if (stratum_label < 0) or (stratum_label >= self.n_strats):
            raise KeyError(f'Unexpected stratum_label = {stratum_label}. There are only {self.n_strats} strats.')

        V_whole, F_whole, old2new_whole, new2old_whole = triangletools.compact_mesh(self.vertices, self.faces[self.strats_labels == stratum_label])
        print(f'V_whole.shape = {V_whole.shape}')
        print(f'F_whole.shape = {F_whole.shape}')
        print(f'old2new_whole.shape = {old2new_whole.shape}')
        print(f'new2old_whole.shape = {new2old_whole.shape}')

        edges_to_cut = old2new_whole[self.skeleton_edges]
        edges_to_cut = edges_to_cut[(edges_to_cut != -1).all(axis=-1)]
        print(f'edges_to_cut.shape = {edges_to_cut.shape}')
        
        V_cut, F_cut, old2new_cut, new2old_cut = triangletools.cut_mesh(V_whole, F_whole, edges_to_cut)
        print(f'V_cut.shape = {V_cut.shape}')
        print(f'F_cut.shape = {F_cut.shape}')
        print(f'old2new_cut.shape = {old2new_cut.shape}')
        print(f'new2old_cut.shape = {new2old_cut.shape}')
        assert False
            
        old2new_global = np.full_like(old2new_whole, -1)
        old2new_global[old2new_whole != -1] = old2new_cut[old2new_whole[old2new_whole != -1]]
        print(f'old2new_global.shape = {old2new_global.shape}')
        new2old_global = new2old_whole[new2old_cut]
        print(f'new2old_global.shape = {new2old_global.shape}')

        return V_cut, F_cut, old2new_global, new2old_global



        
        
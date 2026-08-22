import numpy as np
import scipy as sp
import igl


import itertools
from functools import cached_property, cache

from src import mesh_topology, triangletools
from src import skeleton_function
from src import function_extension
from src import remash

import warnings






class Stratified:
    def define_strats(self):
        """
        """
        self.skeleton_graph = mesh_topology.get_skeleton_graph(self.faces, with_boundary_edges=True, with_conic_vertices=True)        

        face_adjacency_without_skeleton = triangletools.face_adjacency(self.faces, restricted_edges=self.skeleton_graph.edges())
        self.n_strats, self.strats_labels = sp.sparse.csgraph.connected_components(face_adjacency_without_skeleton)


    def __init__(self, faces, vertices, with_conic_vertices: bool=True):
        """
        """
        self.faces = np.unique(np.sort(faces, axis=1), axis=0)
        self.vertices = np.array(vertices)
        self.define_strats()


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

        self.values = np.nan*np.zeros(len(self.vertices))
        self.values[self.skeleton_indices] = skeleton_values
        return self.values


    def get_stratum(self, stratum_label, return_all_copies: bool=False):
        """
        """
        V, F, old2new, new2old = triangletools.compact_mesh(self.vertices, self.faces[self.strats_labels == stratum_label])
        cut_edges = old2new[self.skeleton_edges]
        cut_edges = cut_edges[(cut_edges != -1).all(axis=1)]
        SV, SF, SVI = triangletools.cut_mesh_along_edges(V, F, cut_edges)

        stratum2global = new2old[SVI]
        global2stratum = [[] for _ in range(len(self.vertices))]
        for split_vid, global_vid in enumerate(stratum2global):
            global2stratum[global_vid].append(split_vid)
        if not return_all_copies:
            global2stratum = np.array([-1 if len(copies) == 0 else copies[0] for copies in global2stratum])

        return SV, SF, global2stratum, stratum2global


    def get_stratum_boundary(self, stratum_label, with_values: bool=True):
        """
        """
        SV, SF, global2stratum, stratum2global = self.get_stratum(stratum_label, return_all_copies=True)
        stratum_boundary_indices = np.concatenate([copies for i, copies in enumerate(global2stratum) if i in self.skeleton_indices])
        if with_values:
            stratum_boundary_values = np.concatenate([np.full_like(copies, self.values[i]) for i, copies in enumerate(global2stratum) if i in self.skeleton_indices])
            return stratum_boundary_indices, stratum_boundary_values
        return stratum_boundary_indices

    def get_stratum_boundary_edges(self, stratum_label):
        """
        """
        SV, SF, global2stratum, stratum2global = self.get_stratum(stratum_label, return_all_copies=True)
        stratum_edges = np.unique(np.concatenate(SF[:, [[0, 1], [0, 2], [1, 2]]]), axis=0)
        stratum_edges_global = np.sort(stratum2global[stratum_edges], axis=1)
        stratum_edges_in_skeleton = (stratum_edges_global[:, None, :] == self.skeleton_edges).all(axis=2).any(axis=1)
        return stratum_edges[stratum_edges_in_skeleton]


    def remash(self, flip_ears: bool=True, 
               split_minima: bool=True, split_maxima: bool=False, split_cons: bool=True, 
               edge_choice_strategy='longest'):
        """
        """
        remashed_vertices = []
        remashed_indices = []
        remashed_faces = []
        last_new_vertex_index = self.n_vertices
        for stratum_label in range(self.n_strats):
            V, F, global2stratum, stratum2global = self.get_stratum(stratum_label, return_all_copies=True)
            
            stratum_boundary_indices, stratum_boundary_values = self.get_stratum_boundary(stratum_label, with_values=True)
            stratum_boundary_edges = self.get_stratum_boundary_edges(stratum_label)
            
            if flip_ears:
                #F = remash.flip_ears(F, stratum_boundary_edges)
                F = remash.flip_ears(F, None)
            if split_minima:
                boundary_mins = function_extension.detect_mins(F, stratum_boundary_indices, stratum_boundary_values)
                V, F = remash.split_edges_arroud_saddles(V, F, boundary_mins, edge_choice_strategy)
            if split_maxima:
                boundary_maxs = function_extension.detect_maxs(F, stratum_boundary_indices, stratum_boundary_values)
                V, F = remash.split_edges_arroud_saddles(V, F, boundary_maxs, edge_choice_strategy)
            if split_cons:
                cons = function_extension.detect_cons(F, stratum_boundary_indices)
                V, F = remash.split_edges_arroud_saddles(V, F, cons, edge_choice_strategy)

            n_new_vertices = len(V) - len(stratum2global) 
            remashed2global = np.concatenate([stratum2global, last_new_vertex_index + np.arange(n_new_vertices)])
            last_new_vertex_index += n_new_vertices

            remashed_vertices.append(V.copy())
            remashed_indices.append(remashed2global.copy())
            remashed_faces.append(remashed2global[F].copy())

        remashed_vertices = np.concatenate(remashed_vertices, axis=0)
        remashed_indices = np.concatenate(remashed_indices, axis=0)
        remashed_faces = np.concatenate(remashed_faces, axis=0)

        _, remashed_indices_idx = np.unique(remashed_indices, return_index=True)
        assert len(remashed_indices_idx) == last_new_vertex_index
        assert (np.arange(last_new_vertex_index) == remashed_indices[remashed_indices_idx]).all()
        remashed_vertices = remashed_vertices[remashed_indices_idx]

        self.vertices = remashed_vertices
        self.faces = remashed_faces
        if hasattr(self, 'values'):
            new_values = np.zeros(len(self.vertices))
            new_values[:len(self.values)] = self.values
            self.values = new_values
        self.__dict__.pop("edges", None)
        self.define_strats()
        


    def define_values_on_stratum(self, stratum_label, extension_values_function=function_extension.second_boundary_dirichlet_laplacian_eigenfunction_plus_harmonic, **kwargs):
        """
        """
        stratum_vertices, stratum_faces, global2stratum, stratum2global = self.get_stratum(stratum_label, return_all_copies=True)


        stratum_boundary_indices, stratum_boundary_values = self.get_stratum_boundary(stratum_label, with_values=True)

        stratum_values = extension_values_function(stratum_vertices, stratum_faces, stratum_boundary_indices, stratum_boundary_values, **kwargs)

        self.values[stratum2global] = stratum_values
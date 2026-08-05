import numpy as np
import scipy as sp
import igl

import itertools
from functools import cached_property, cache
from collections import deque

class MorseSmale:
    def __init__(self, faces, values, vertices=None):
        r"""
        Initialize the 2-dimensional simplicilial complex with N vertices and M 2-faces to quadrangulate

        Parameters:
        -----------
        faces: array shape (M, 3)
            The list of 2-faces of the simplicial complex

        values: array shape (N, )
            The filtration values of the vertices
        
        vertices: array shape (N, d) or None
            If the complex has an embeding in d-dimensional eucledean space, we can define the cords of the vertices
        """
        self.faces = np.unique(np.sort(faces, axis=1), axis=0)
        self.values = np.array(values)
        if vertices is None:
            self.vertices = None
        else:
            self.vertices = np.array(vertices)
            if (self.vertices.shape[0] != self.values.shape[0]) or (self.vertices.ndim != 2):
                raise ValueError(f'Expected vertices length ({self.values.shape[0]}, d)')

            
    @cached_property
    def edges(self):
        """
        """
        edges = self.faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        return edges


    @property
    def n_vertices(self):
        """
        """
        return self.values.shape[0]


    @property
    def n_edges(self):
        """
        """
        return self.edges.shapoe[0]


    @property
    def n_faces(self):
        """
        """
        return self.faces.shape[0]


    @cached_property
    def local_minima(self):
        """
        """
        adjacency_matrix = igl.adjacency_matrix(self.faces)
        is_min = lambda i: (self.values[i] < self.values[adjacency_matrix.getrow(i).indices]).all()
        mins = np.argwhere(np.vectorize(is_min)(np.arange(self.n_vertices))).ravel()
        return mins
    
    @cached_property
    def local_maxima(self):
        """
        """
        adjacency_matrix = igl.adjacency_matrix(self.faces)
        is_max = lambda i: (self.values[i] > self.values[adjacency_matrix.getrow(i).indices]).all()
        maxs = np.argwhere(np.vectorize(is_max)(np.arange(self.n_vertices))).ravel()
        return maxs


    @cache
    def get_saddles_and_directions(self):
        """
        ...

        Returns:
        --------
        saddles : list[int]

        increasing_steps: list[list[int]]

        decreasing_steps: list[list[int]]
        """
        saddles = []
        increasing_steps = []
        decreasing_steps = []

        adjacency_matrix = igl.adjacency_matrix(self.faces)

        for i in range(self.n_vertices):
            neighbour_indices = adjacency_matrix.getrow(i).indices
            neighbour_values = self.values[neighbour_indices]

            adjacency_submatrix = adjacency_matrix[neighbour_indices][:, neighbour_indices]
                        
            higher_mask = neighbour_values >= self.values[i]
            higher_indices = neighbour_indices[higher_mask]
            higher_values = neighbour_values[higher_mask]
            adjacency_submatrix_higher = adjacency_submatrix[higher_mask][:, higher_mask]
            higher_n_components, higher_labels = sp.sparse.csgraph.connected_components(csgraph=adjacency_submatrix_higher, directed=False, return_labels=True)

            lower_mask = neighbour_values <= self.values[i]
            lower_indices = neighbour_indices[lower_mask]
            lower_values = neighbour_values[lower_mask]
            adjacency_submatrix_lower = adjacency_submatrix[lower_mask][:, lower_mask]
            lower_n_components, lower_labels = sp.sparse.csgraph.connected_components(csgraph=adjacency_submatrix_lower, directed=False, return_labels=True)


            # maybe there should be some other condition...
            if (higher_n_components > 1) and (lower_n_components > 1):
                saddles.append(i)

                increasing_steps.append([])
                for label in range(higher_n_components):
                    increasing_steps[-1].append(higher_indices[higher_labels == label][higher_values[higher_labels == label].argmax()])

                decreasing_steps.append([])
                for label in range(lower_n_components):
                    decreasing_steps[-1].append(lower_indices[lower_labels == label][lower_values[lower_labels == label].argmin()])
                                
        return saddles, increasing_steps, decreasing_steps


    @property
    def saddles(self):
        """
        """
        saddles = np.array(self.get_saddles_and_directions()[0])
        return saddles


    def drections_queue(self, how='increasing-decreasing'):
        """
        """
        if how == 'increasing-decreasing':
            saddles, increasing_steps, decreasing_steps = self.get_saddles_and_directions()
            drections_queue = deque(
                (saddle, int(step))
                for saddle, steps in itertools.chain(
                    zip(saddles, increasing_steps),
                    zip(saddles, decreasing_steps),
                )
                for step in steps
            )
            
        if how == 'decreasing-increasing':
            saddles, increasing_steps, decreasing_steps = self.get_saddles_and_directions()
            drections_queue = deque(
                (saddle, int(step))
                for saddle, steps in itertools.chain(
                    zip(saddles, decreasing_steps),
                    zip(saddles, increasing_steps),
                )
                for step in steps
            )

        return drections_queue
            



            

import numpy as np
import scipy as sp
import igl

import itertools
from functools import cached_property, cache
from collections import deque

from src import pathtools

from tqdm import tqdm





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
    def local_maxima(self):
        """
        """
        adjacency_matrix = igl.adjacency_matrix(self.faces)
        is_max = lambda i: (self.values[i] > self.values[adjacency_matrix.getrow(i).indices]).all()
        maxs = np.argwhere(np.vectorize(is_max)(np.arange(self.n_vertices))).ravel()
        return maxs

    
    @cached_property
    def local_minima(self):
        """
        """
        adjacency_matrix = igl.adjacency_matrix(self.faces)
        is_min = lambda i: (self.values[i] < self.values[adjacency_matrix.getrow(i).indices]).all()
        mins = np.argwhere(np.vectorize(is_min)(np.arange(self.n_vertices))).ravel()
        return mins


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
            if (higher_n_components > 1) or (lower_n_components > 1):
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
        return np.array(self.get_saddles_and_directions()[0])


    @property
    def n_paths(self):
        """
        """
        return np.concatenate(self.get_saddles_and_directions()[1] + self.get_saddles_and_directions()[2]).size


    def iterate_increasing_directions(self):
        """
        """
        saddles, increasing_steps, decreasing_steps = self.get_saddles_and_directions()
        for saddle, steps in zip(saddles, increasing_steps):
            for step in steps:
                yield saddle, step
        

    def iterate_decreasing_directions(self):
        """
        """
        saddles, increasing_steps, decreasing_steps = self.get_saddles_and_directions()
        for saddle, steps in zip(saddles, decreasing_steps):
            for step in steps:
                yield saddle, step


    def get_directions_queue(self, how='increasing-decreasing') -> deque:
        """
        """
        if how == 'increasing-decreasing':
            drections_queue = deque(itertools.chain(self.iterate_increasing_directions(), 
                                                    self.iterate_decreasing_directions()))
            
        elif how == 'decreasing-increasing':
            drections_queue = deque(itertools.chain(self.iterate_decreasing_directions(), 
                                                    self.iterate_increasing_directions()))

        else:
            pass

        return drections_queue



    def get_available_next_vertices_for_the_path(self, path, old_paths=[]):
        """
        """
        nexts = self.edges[(self.edges == path[-1]).any(axis=1)]
        nexts = nexts[~np.isin(nexts, path)]
        does_continuation_intersect_old = lambda next: np.any([pathtools.paths_intersect(np.append(path, next), old_path, self.faces) for old_path in old_paths])
        nexts = nexts[~np.vectorize(does_continuation_intersect_old)(nexts)]
        
        return nexts


    def continue_path(self, path, old_paths=[]):
        """
        """
        # FIXME: Defined incorrect destinations, and they are achived
        nexts = self.get_available_next_vertices_for_the_path(path, old_paths)
        dirrection_coeff = (self.values[path[-1]] - self.values[path[0]])
        next = nexts[np.argmax(dirrection_coeff*(self.values[nexts] - self.values[path[-1]]))]
        return np.append(path, next)


    def get_paths(self, how='increasing-decreasing', cache: bool=True, with_bar: bool=True):
        """
        """
        # FIXME: Defined incorrect destinations, and they are achived
        if cache and hasattr(self, '_paths'):
            return self._paths
        
        directions_queue = self.get_directions_queue(how=how)

        directions_returned_to_queue = set()
        if with_bar:
            pbar = tqdm(total=self.n_paths, desc=f'Searching paths')
            
        paths = []
        while directions_queue:
            merged_paths = pathtools.merge_paths_at_nodes(paths, self.saddles)

            new_path = np.array(directions_queue.popleft())
            increasing = self.values[new_path[1]] > self.values[new_path[0]]

            destinations = self.local_maxima if increasing else self.local_minima
            
            while new_path[-1] not in destinations:
                new_path = self.continue_path(new_path, old_paths=merged_paths)
                if new_path[-1] in self.saddles:
                    # Check, if saddle have opposite paths, and add this direction to the end of queue if not
                    saddle = new_path[-1]
                    saddles, increasing_steps, decreasing_steps = self.get_saddles_and_directions()
                    opposite_steps = (decreasing_steps if increasing else increasing_steps)[saddles.index(saddle)]
                    step_is_done = lambda step: np.any([(path[:2] == np.array([saddle, step])).all() for path in paths])

                                        
                    if not np.vectorize(step_is_done)(opposite_steps).all():
                        directions_queue.append(tuple(new_path[:2]))
                        directions_returned_to_queue.add(tuple(new_path[:2]))
                        
            if not new_path[-1] in self.saddles:
                paths.append(np.array(new_path))
                directions_returned_to_queue -= {tuple(new_path[:2])}
                if with_bar:
                    pbar.update()
                
            if with_bar:
                pbar.set_postfix({'Returned directions': directions_returned_to_queue})
                pbar.refresh()

        if cache:
            self._paths = paths
        return paths




    def iterate_paths_close_geodesics(self, how='increasing-decreasing', cache: bool=True, with_bar: bool=True):
        """
        """
        for path in self.get_paths(how=how, cache=cache, with_bar=with_bar):
            geopath = pathtools.get_path_close_geodesic(path, self.faces, self.vertices)
            yield geopath

    


import numpy as np
import scipy as sp
import igl

import itertools
from functools import cached_property, cache
from collections import deque

from src import pathtools
from src import triangletools
from src import mesh_topology

from tqdm import tqdm
import warnings

from src.timing import Timer





class MorseSmale:
    def __init__(self, faces, values, vertices=None, protected=None):
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

        if protected is None:
            self.protected = mesh_topology.get_boundary_edges(self.faces)
        else:
            self.protected = np.asarray(protected)

            
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



    def get_available_next_vertices_for_the_path(self, path, old_paths=None):
        """
        """
        if old_paths is None:
            old_paths = []
        nexts = self.edges[(self.edges == path[-1]).any(axis=1)]
        nexts = nexts[~np.isin(nexts, path)]

        old_paths_to_check = [old_path for old_path in old_paths if (old_path == path[-1]).any()]
        forking_saddles = np.intersect1d(self.saddles, path[1:])
        old_paths_to_check = pathtools.merge_paths_at_nodes(old_paths_to_check, forking_saddles)

        continuation_intersects_old = [[pathtools.paths_intersect(np.append(path, next_vertex), old_path, self.faces) for old_path in old_paths_to_check] for next_vertex in nexts]
        continuation_intersects_old = np.any(continuation_intersects_old, axis=1)

        nexts = nexts[~continuation_intersects_old]

        return nexts


    def continue_path(self, path, old_paths=None):
        """
        """
        nexts = self.get_available_next_vertices_for_the_path(path, old_paths)
        if len(nexts) == 0:
            if old_paths is None:
                old_paths_blocking = []
            else:
                old_paths_blocking = [old_path for old_path in old_paths if (old_path == path[-1]).any()]
            msg = f"The path {np.array2string(path, separator=", ")} can't be continued.\nIt's blocked by\n"
            msg += "[" + ",\n ".join([np.array2string(old_path, separator=", ") for old_path in old_paths_blocking]) + "\n]"
            raise ValueError(msg)

        dirrection_coeff = (self.values[path[-1]] - self.values[path[0]])
        next = nexts[np.argmax(dirrection_coeff*(self.values[nexts] - self.values[path[-1]]))]
        return np.append(path, next)


    def get_paths(self, how='increasing-decreasing', cache: bool=True, with_bar: bool=True):
        """
        """
        if cache and hasattr(self, '_paths'):
            return self._paths
        
        directions_queue = self.get_directions_queue(how=how)

        directions_returned_to_queue = set()
        if with_bar:
            pbar = tqdm(total=self.n_paths, desc=f'Searching paths')
            
        paths = []
        j = 0
        while directions_queue:
            new_path = np.array(directions_queue.popleft())
            increasing = self.values[new_path[1]] > self.values[new_path[0]]

            destinations = self.local_maxima if increasing else self.local_minima
            
            while new_path[-1] not in destinations:
                new_path = self.continue_path(new_path, old_paths=paths)
                if new_path[-1] in self.saddles:
                    # Check, if saddle have opposite paths, and add this direction to the end of queue if not
                    saddle = new_path[-1]
                    saddles, increasing_steps, decreasing_steps = self.get_saddles_and_directions()
                    opposite_steps = (decreasing_steps if increasing else increasing_steps)[saddles.index(saddle)]
                    step_is_done = lambda step: np.any([(path[:2] == np.array([saddle, step])).all() for path in paths])

                    if not np.vectorize(step_is_done)(opposite_steps).all():
                        directions_queue.append(tuple(new_path[:2]))
                        directions_returned_to_queue.add(tuple(new_path[:2]))
                        break

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


    def get_protected_critical_points(self):
        """
        """
        return np.intersect1d(np.concatenate([self.saddles, self.local_maxima, self.local_minima]), self.protected)


    def detect_cancelling_pairs(self, paths):
        """
        """
        path_starts = np.array([path[0] for path in paths])
        path_ends = np.array([path[-1] for path in paths])
        extremums = np.concatenate([self.local_maxima, self.local_minima])
        extremums = np.setdiff1d(extremums, self.protected)
        extremums_paths_count = (extremums[:, None] == path_ends).sum(axis=1)
        canceling_extremums = extremums[(extremums_paths_count == 1) | (extremums_paths_count == 2)]

        cancelling_pairs = []
        for extremum in canceling_extremums:
            extremum_value = self.values[extremum]

            connected_saddles = np.unique(path_starts[np.argwhere(path_ends == extremum).ravel()])
            connected_saddles = np.setdiff1d(connected_saddles, self.protected)
            # TODO: Monkey saddle case. In this case we can be able to cancel the saddle from the boundary
            pass

            if len(connected_saddles) == 0:
                connected_saddles = np.unique(path_starts[np.argwhere(path_ends == extremum).ravel()])
                msg = f"The extrmum {extremum} with just {len(connected_saddles)} relations is uncancellable:\n"
                msg += f"All related saddles {connected_saddles} are protected from cancellation."
                raise ValueError(msg)

            connected_saddles_values = self.values[connected_saddles]

            saddle = connected_saddles[np.argmin(abs(connected_saddles_values - extremum_value))]
            cancelling_pairs.append((saddle, extremum))
        
        return cancelling_pairs



    def cancel_pair(self, paths, saddle, extremum):
        """
        """
        saddle_paths = [path for path in paths if path[0] == saddle]
        extremum_paths = [path for path in paths if path[-1] == extremum]

        if len(saddle_paths) == 4:
            paths_after_cancelation = [path for path in paths if (path[0] != saddle) and (path[-1] != extremum)]
            if len(extremum_paths) == 2:
                opposite_path = [path for path in saddle_paths if (tuple(path[[0, -1]]) != (saddle, extremum)) and (self.values[path[0]] - self.values[path[1]])*(self.values[saddle] - self.values[extremum]) > 0][0]
                new_path = pathtools.concatenate_paths(pathtools.concatenate_paths(extremum_paths[0], extremum_paths[1]), opposite_path)
                if new_path[0] not in self.saddles:
                    new_path = new_path[::-1]
                paths_after_cancelation.append(new_path)
            return paths_after_cancelation
        else:
            # TODO: Monkey Saddle Case
            pass


    def get_paths_after_cancellations(self, how='increasing-decreasing', cache: bool=True, with_bar: bool=True, cancellation_failure_strategy="raise"):
        """
        """
        if cache and hasattr(self, '_paths_after_cancellations'):
            return self._paths_after_cancellations

        paths = self.get_paths(how=how, cache=cache, with_bar=with_bar)


        def cancellation_failure(paths, err: ValueError):
            if cancellation_failure_strategy == 'raise':
                raise err
            if cancellation_failure_strategy == 'warn':
                warnings.warn(str(err))
                return paths
            if cancellation_failure_strategy == 'return':
                return paths

        try:
            cancelling_pairs = self.detect_cancelling_pairs(paths)
        except ValueError as err:
            cancellation_failure(paths, err)

        while len(cancelling_pairs) > 0:
            try:
                saddle, extremum = cancelling_pairs[0]
                paths = self.cancel_pair(paths, saddle, extremum)
                cancelling_pairs = self.detect_cancelling_pairs(paths)
            except ValueError as err:
                cancellation_failure(paths, err)

        if cache:
            self._paths_after_cancellations = paths
        return paths


    def get_labels_separated_by_paths(self, paths):
        """
        """
        paths_edges = np.concatenate([np.transpose([path[1:], path[:-1]]) for path in paths], axis=0)
        paths_edges = np.unique(np.sort(paths_edges, axis=1), axis=0)

        face_adjacency = triangletools.face_adjacency(self.faces, paths_edges)
        n_comps, comp_labels = sp.sparse.csgraph.connected_components(csgraph=face_adjacency, directed=False, return_labels=True)

        return n_comps, comp_labels


    def get_quadrangle_labels(self, how='increasing-decreasing', cache: bool=True, with_bar: bool=True, cancellation_failure_strategy="raise"):
        """
        """
        if cache and hasattr(self, 'quadrangle_labels'):
            return self.quadrangle_labels

        paths = self.get_paths_after_cancellations(how=how, cache=cache, with_bar=with_bar, 
                                                   cancellation_failure_strategy=cancellation_failure_strategy)
        paths_sides = [pathtools.get_path_sides(path, self.faces) for path in paths]
        paths_sides_paired = [path_sides for path_sides in paths_sides if len(path_sides) == 2]
        sides = list(itertools.chain(*paths_sides))

        n_comps, comp_labels = self.get_labels_separated_by_paths(paths)

        # Unite components from one quadrangle, detecting corresponding boundary paths
        sides_groups = [np.unique(comp_labels[side]) for side in sides]
        sides_antigroups_pairs = [(np.unique(comp_labels[side0]), np.unique(comp_labels[side1])) for side0, side1 in paths_sides_paired]

        rows, cols = np.concatenate([[group[1:], group[:-1]] for group in sides_groups], axis=1)
        #rows, cols = np.concatenate([list(itertools.combinations(group, 2)) for group in sides_groups if group.size > 1], axis=0).transpose()

        for antigroup0, antigroup1 in sides_antigroups_pairs:
            drop_condition = (np.isin(rows, antigroup0) & np.isin(cols, antigroup1)) | (np.isin(rows, antigroup1) & np.isin(cols, antigroup0))
            rows = rows[~drop_condition]
            cols = cols[~drop_condition]


        adjacency_matrix = sp.sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_comps, n_comps), dtype=np.uint8)

        n_quadrangles, labels2quadrangles = sp.sparse.csgraph.connected_components(csgraph=adjacency_matrix, directed=False, return_labels=True)

        quadrangles_labels = labels2quadrangles[comp_labels]

        return quadrangles_labels


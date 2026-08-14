import itertools

import numpy as np
import scipy as sp
import networkx as nx

from pygeodesic import geodesic

from src import graph_methods, geometry, triangletools
from src.graph_simplification import simplify_graph

import warnings


class MorseSmale:
    def __init__(self, faces, values, vertices=None, forest_method='steepest', gradient_respects_distance=False):
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

        forest_method: str
            ...
            ``'steepest'``: ...
            ``'spaning'``: ...
        
        gradient_respects_distance : bool
            Determines how the gradient between the filtration values of two
            adjacent vertices is computed.

            If ``True``, the gradient is normalized by the Euclidean distance
            between the vertices:

            .. math::

                \frac{f(v_1) - f(v_0)}{\lVert v_1 - v_0 \rVert}

            Otherwise, the gradient is computed as the filtration value difference:

            .. math::

                f(v_1) - f(v_0)
        """
        self.faces = np.unique(np.sort(faces, axis=1), axis=0)
        self.values = np.array(values)
        if vertices is None:
            self.vertices = None
        else:
            self.vertices = np.array(vertices)
            if (self.vertices.shape[0] != self.values.shape[0]) or (self.vertices.ndim != 2):
                raise ValueError(f'Expected vertices length ({self.values.shape[0]}, d)')
        
        self.n_vertices = self.values.shape[0]
        self.n_edges = np.unique(np.sort(np.concatenate(self.faces[:, [[0, 1], [0, 2], [1, 2]]]), axis=1), axis=0).shape[0]
        self.n_faces = self.faces.shape[0]
    
        if forest_method.lower() == 'steepest':
            self._get_increasing_graph_method = graph_methods.get_steepest_increasing_graph
        elif forest_method.lower() == 'spaning':
            self._get_increasing_graph_method = graph_methods.get_spaning_increasing_graph
        
        self.gradient_respects_distance = gradient_respects_distance

    def distance(self, index0, index1):
        """
        Return the distance between the vertices 2 vertices

        If embedding is not defined then distance between any distinct vertices is 1
        """
        if self.vertices is not None:
            return np.linalg.norm(self.vertices[index1] - self.vertices[index0], axis=-1)
        else:
            return np.array(index0 != index1).astype(int)

    def gradient(self, index0, index1):
        """
        Return the gradient between the filtration values of two adjacent vertices.
        """
        if self.gradient_respects_distance:
            val = (self.values[index1] - self.values[index0])/self.distance(index0, index1)
        else:
            val = (self.values[index1] - self.values[index0])
        return val
    
    def get_edge_graph(self) -> nx.Graph:
        """
        Return the 1-skeleton graph of the simplicial complex

        Return:
        -------
        self.edge_graph : nx.Graph
        """
        if not hasattr(self, 'edge_graph'):
            self.edge_graph = nx.Graph()
            self.edge_graph.add_nodes_from(range(self.n_vertices))
            edges = np.unique(np.sort(np.concatenate(self.faces[:, [[0, 1], [0, 2], [1, 2]]]), axis=1), axis=0)
            self.edge_graph.add_edges_from(edges)
        return self.edge_graph.copy()
    
    def get_face_graph(self):
        """
        Return the face adjacency graph.

        Returns:
        --------
        self.face_graph: nx.Graph
            Nodes are indices of the faces
            Edges exists between 2 nodes, if 2 coresponding faces share a common edge
        """
        if not hasattr(self, 'face_graph'):
            self.face_graph = nx.Graph()
            self.face_graph.add_nodes_from(range(self.n_faces))
            for (i0, face0), (i1, face1) in itertools.combinations(enumerate(self.faces), 2):
                intersection = np.intersect1d(face0, face1)
                if len(intersection) == 2:
                    self.face_graph.add_edge(i0, i1, intersection=intersection)
        return self.face_graph.copy()
    
    def get_increasing_graph(self) -> nx.DiGraph:
        """
        Return the directed graph: edges are from nodes to their maximum higher in the filtration neighbour.

        Return:
        -------
        self.increasing_graph : nx.DiGraph
            Directed subgraph of self.edge_graph
        """
        if not hasattr(self, 'increasing_graph'):
            self.increasing_graph = self._get_increasing_graph_method(self.get_edge_graph(), 
                                                                      gradient_function=self.gradient, 
                                                                      distance_function=self.distance)
        return self.increasing_graph.copy()

    def get_decreasing_graph(self) -> nx.DiGraph:
        """
        Return the directed graph: edges are from nodes to their minimum lower in the filtration neighbour.

        Return:
        -------
        self.decreasing_graph : nx.DiGraph
            Directed subgraph of self.edge_graph
        """
        if not hasattr(self, 'decreasing_graph'):
            self.decreasing_graph = self._get_increasing_graph_method(self.get_edge_graph(), 
                                                                      gradient_function=lambda i0, i1: self.gradient(i1, i0), 
                                                                      distance_function=self.distance)
        return self.decreasing_graph.copy()
    
    def define_critical_points(self):
        """
        Define critical points

        Attributes:
        -----------
        mins: list
            Indicises of local minimum vertices

        maxs: list
            Indicises of local maximum vertices
            
        saddles: list
            Indicises of saddle vertices
        """
        if (hasattr(self, 'mins') and hasattr(self, 'maxs') and hasattr(self, 'saddles')):
            return None
        
        self.mins = []
        self.maxs = []
        self.saddles = []
        for node in range(self.n_vertices):
            neighborhood_faces = self.faces[(self.faces == node).any(axis=1)]
            neighborhood_edges = neighborhood_faces[neighborhood_faces != node].reshape(-1, 2)
            neighborhood_nodes = np.unique(neighborhood_edges)
            neighborhood_grads = self.gradient(node, neighborhood_nodes)
            if (neighborhood_grads > 0).all():
                self.mins.append(node)
            elif (neighborhood_grads < 0).all():
                self.maxs.append(node)
            else:
                graph_neighborhood = nx.Graph()
                graph_neighborhood.add_nodes_from(neighborhood_nodes)
                graph_neighborhood.add_edges_from(neighborhood_edges)
                graph_lower_neighborhood = graph_neighborhood.subgraph(neighborhood_nodes[neighborhood_grads < 0])
                graph_higher_neighborhood = graph_neighborhood.subgraph(neighborhood_nodes[neighborhood_grads > 0])
                regular = nx.is_connected(graph_lower_neighborhood) and nx.is_connected(graph_higher_neighborhood)
                if not regular:
                    self.saddles.append(node)
                
    
    def iterate_saddles_and_increasing_directions(self):
        """
        Iterate the origins (first 2 nodes) of increasing path

        Yields:
        -------
        saddle: int
            The index of the 1st node in the increasing path
            This is always a saddle

        next_node: int
            The index of the 2nd node in the increasing path
        """
        if not hasattr(self, 'saddles'):
            self.define_critical_points()
        for saddle in self.saddles:
            #neighbors = np.array(list(self.get_edge_graph().neighbors(saddle)))
            #neighbors_gradients = self.gradient(saddle, neighbors) 
            #graph_higher_neighborhood = self.get_edge_graph().subgraph(neighbors[neighbors_gradients > 0])
            neighborhood_graph = triangletools.get_neighborhood_graph(self.faces, saddle, with_center=False)
            neighbors = np.array(list(neighborhood_graph.nodes()))
            neighbors_gradients = self.gradient(saddle, neighbors)
            graph_higher_neighborhood = neighborhood_graph.subgraph(neighbors[neighbors_gradients > 0])
            for component in nx.connected_components(graph_higher_neighborhood):
                next_node = list(component)[self.gradient(saddle, list(component)).argmax()]
                yield (saddle, next_node)
                

    def iterate_saddles_and_decreasing_directions(self):
        """
        Iterate the origins (first 2 nodes) of decreasing path

        Yields:
        -------
        saddle: int
            The index of the 1st node in the decreasing path
            This is always a saddle

        next_node: int
            The index of the 2nd node in the decreasing path
        """
        if not hasattr(self, 'saddles'):
            self.define_critical_points()
        
        for saddle in self.saddles:
            #neighbors = np.array(list(self.edge_graph.neighbors(saddle)))
            #neighbors_gradients = self.gradient(saddle, neighbors) 
            #graph_lower_neighborhood = self.edge_graph.subgraph(neighbors[neighbors_gradients < 0])
            neighborhood_graph = triangletools.get_neighborhood_graph(self.faces, saddle, with_center=False)
            neighbors = np.array(list(neighborhood_graph.nodes()))
            neighbors_gradients = self.gradient(saddle, neighbors)
            graph_lower_neighborhood = neighborhood_graph.subgraph(neighbors[neighbors_gradients < 0])
            for component in nx.connected_components(graph_lower_neighborhood):
                next_node = list(component)[self.gradient(saddle, list(component)).argmin()]
                yield (saddle, next_node)
    

#    def is_correct_saddle_on_path(self, path, saddle, other_paths=[]) -> bool:
#        """
#        Return True if the saddle on path has a correct direction out
#        
#        Parameters:
#        -----------
#        path : np.array[int]
#
#        saddle: int
#
#        other_paths: list[np.array[int]]
#
#        Returns:
#        --------
#        path_turns_correctly: bool
#        """
#        if (saddle not in path[1:-1]) or (saddle not in self.saddles):
#            raise ValueError("saddle expected to be a not booundary saddle on the path")
#        
#        # get a structure of faces surrounding the saddle
#        surrounding_faces_indices = np.argwhere((self.faces == saddle).any(axis=1)).ravel()
#        surrounding_faces_graph = self.get_face_graph().subgraph(nodes=surrounding_faces_indices).copy()
#
#        # path splits the faces surrounding the saddle into 2 halfspace
#        path_edges = np.sort(np.transpose([path[1:], path[:-1]]), axis=1)
#        graph_path_edges = [(e0, e1) for e0, e1, data in surrounding_faces_graph.edges(data=True) if (data['intersection'] == path_edges).all(axis=1).any()]
#        surrounding_faces_graph.remove_edges_from(graph_path_edges)
#        assert nx.number_connected_components(surrounding_faces_graph) <= 2
#        halfspace_graphs = [surrounding_faces_graph.subgraph(component).copy() for component in nx.connected_components(surrounding_faces_graph)]
#       
#        # split halfspaces by other paths
#        other_paths_edges = np.sort(np.transpose(np.hstack([[chain[1:], chain[:-1]] for chain in other_paths])), axis=1)
#        graph_other_path_edges = [(e0, e1) for e0, e1, data in surrounding_faces_graph.edges(data=True) if (data['intersection'] == other_paths_edges).all(axis=1).any()]
#        for i in range(len(halfspace_graphs)):
#            halfspace_graphs[i].remove_edges_from(graph_other_path_edges)
#        
#        # check if there is a halfspace not splited by other paths
#        halfspace_graph_splits = np.array([nx.number_connected_components(g) for g in halfspace_graphs])
#        path_turns_correctly = (halfspace_graph_splits == 1).any()
#
#
#        print(f"Path through saddle: {path[0]} - {saddle} - {path[-1]}")
#        print(f"Suurounding halfspaces: {[g.nodes() for g in halfspace_graphs]}")
#        print(f"Suurounding halfspaces splits: {halfspace_graph_splits}")
#        print(f"Verdict: {path_turns_correctly}")
#        return path_turns_correctly




    def get_chain_halfspaces(self, chain):
        """
        """
        surrounding_faces_indices = np.isin(self.faces, chain[1:-1]).any(axis=1) 
        surrounding_faces_indices = surrounding_faces_indices | ((self.faces == chain[0]).any(axis=1) & (self.faces == chain[1]).any(axis=1))
        surrounding_faces_indices = surrounding_faces_indices | ((self.faces == chain[-1]).any(axis=1) & (self.faces == chain[-2]).any(axis=1))
        surrounding_faces_indices = np.argwhere(surrounding_faces_indices).ravel()
        #print(f'surrounding_faces_indices: {surrounding_faces_indices}')

        surrounding_faces_graph = self.get_face_graph().subgraph(surrounding_faces_indices).copy()
        #print(f'surrounding_faces_graph.nodes (before removing paths): {surrounding_faces_graph.nodes}')
        #print(f'surrounding_faces_graph.edges (before removing paths): {surrounding_faces_graph.edges}')
        edges_in_chain = np.sort(np.transpose([chain[1:], chain[:-1]]), axis=1)
        edges_to_remove = [(e0, e1) for e0, e1, data in surrounding_faces_graph.edges(data=True) if (data['intersection'] == edges_in_chain).all(axis=1).any()]
        surrounding_faces_graph.remove_edges_from(edges_to_remove)
        #print(f'surrounding_faces_graph.nodes (after removing paths): {surrounding_faces_graph.nodes}')
        #print(f'surrounding_faces_graph.edges (after removing paths): {surrounding_faces_graph.edges}')
        halfspaces_face_indices = [list(comp) for comp in nx.connected_components(surrounding_faces_graph)]
        #print(f'halfspaces_face_indices: {halfspaces_face_indices}')
        assert len(halfspaces_face_indices) in {1, 2}
        return halfspaces_face_indices


    def is_correct_saddle_on_path(self, path, saddle, other_paths=[]) -> bool:
        """
        Return True if the saddle on path has a correct direction out
        
        Parameters:
        -----------
        path : np.array[int]

        saddle: int

        other_paths: list[np.array[int]]

        Returns:
        --------
        path_turns_correctly: bool
        """
        if (saddle not in path[1:-1]) or (saddle not in self.saddles):
            raise ValueError("saddle expected to be a not booundary saddle on the path")

        saddle_index = list(path).index(saddle)
        halfspaces = self.get_chain_halfspaces(path[:saddle_index + 2])


        face_graph_without_paths = self.get_face_graph()
        edges_in_paths = np.sort(np.transpose(np.hstack([[chain[1:], chain[:-1]] for chain in other_paths])), axis=1)
        edges_to_remove = [(e0, e1) for e0, e1, data in face_graph_without_paths.edges(data=True) if (data['intersection'] == edges_in_paths).all(axis=1).any()]
        face_graph_without_paths.remove_edges_from(edges_to_remove)
        components_in_halfspaces = np.array([nx.number_connected_components(face_graph_without_paths.subgraph(hfaces)) for hfaces in halfspaces])
        path_turns_correctly = (components_in_halfspaces == 1).any()

        print(f"Path through saddle: {path[0]} - {saddle} - {path[-1]}")
        print(f"Surrounding halfspaces: {[hf for hf in halfspaces]}")
        print(f"Components in halfspaces: {components_in_halfspaces}")
        print(f"Verdict: {path_turns_correctly}")
        return path_turns_correctly
        



    def iterate_incorrect_saddles_on_paths(self, paths=None, first=True):
        """
        ...

        Yields:
        --------
        path_index: int

        saddle: int
        """
        if paths is None:
            paths = self.get_paths()
        for path_index, path in enumerate(paths):
            path_saddles = path[1:-1][np.isin(path[1:-1], self.saddles)]
            for saddle in path_saddles:
                if not self.is_correct_saddle_on_path(path, saddle, paths):
                    yield path_index, saddle
                    if first:
                        break


    
    def path_is_increasing(self, path) -> bool:
        """
        """
        path_vals = self.values[path]
        status = path_vals[1] > path_vals[0]
        return status
    
    def fix_path(self, path, saddle, other_paths=[]):
        """
        """
        if self.is_correct_saddle_on_path(path, saddle, other_paths):
            return path
        path_direction = self.path_is_increasing(path)
        suffixes = [chain for chain in other_paths if chain[0] == saddle and self.path_is_increasing(chain) == path_direction]
        prefix = path[:list(path).index(saddle)]
        for suffix in suffixes:
            new_path = np.concatenate([prefix, suffix])
            if self.is_correct_saddle_on_path(new_path, saddle, other_paths):
                return new_path



    def get_paths(self, fix_incorrect_paths=True):
        """
        Return list of paths: increasing (from sadles to local maxima) and decreasing (from saddles to local minima)

        Parameters:
        -----------
        fix_incorrect_paths : bool
            Fix incorrect crossing paths, which could appear if there is a saddle somewhere in the middle of the path

        Returns:
        --------
        self.paths: list[np.array[int]]
            List of paths, represented as arrays of consequtive vertex indices from saddles to local maxima/minima
        """
        # FIXME: The second saddles along paths can generate path intersections
        # TODO: Rewrite fixing incorrect paths
        if not hasattr(self, 'paths'):

            self.paths = []
            for saddle, next_node in self.iterate_saddles_and_increasing_directions():
                path = graph_methods.get_chain_from(self.get_increasing_graph(), next_node)
                path = np.append(saddle, path)
                self.paths.append(path)
            for saddle, next_node in self.iterate_saddles_and_decreasing_directions():
                path = graph_methods.get_chain_from(self.get_decreasing_graph(), next_node)
                path = np.append(saddle, path)
                self.paths.append(path)
            
            if fix_incorrect_paths:
                incorrect_path_saddle_pairs = list(self.iterate_incorrect_saddles_on_paths(self.paths, first=True))
                while len(incorrect_path_saddle_pairs) > 0:
                    print(f'incorrect_path_saddle_pairs: {len(incorrect_path_saddle_pairs)}:\n{incorrect_path_saddle_pairs}')
                    path_index, saddle = incorrect_path_saddle_pairs[0]
                    path = self.paths.pop(path_index)
                    new_path = self.fix_path(path, saddle, self.paths)
                    self.paths.append(new_path)
                    incorrect_path_saddle_pairs = list(self.iterate_incorrect_saddles_on_paths(self.paths, first=True))


            #if fix_incorrect_paths:
            #    def common_suffix(a, b):
            #        a = np.asarray(a)
            #        b = np.asarray(b)
            #        
            #        n = min(len(a), len(b))
            #        a_rev = a[-n:][::-1]
            #        b_rev = b[-n:][::-1]
            #        
            #        diff = np.flatnonzero(a_rev != b_rev)
            #        k = diff[0] if len(diff) else n
            #        
            #        return a[-k:] if k > 0 else np.array([])
            #    
            #    possibly_wrong_paths_indices = [i for i, path in enumerate(self.paths) if np.isin(path[1:-1], self.saddles).any()]
            #    for i in possibly_wrong_paths_indices:
            #        path = self.paths[i]
            #
            #        # this could work only in the case when there is only 1 intersection
            #        saddle = path[1:-1][np.isin(path[1:-1], self.saddles)][0]
            #        short_path = path[:list(path).index(saddle) + 1]
            #        print(f'short_path: {short_path}')
            #        if path[-1] in self.maxs:
            #            saddle_directions = [next_node for s, next_node in self.iterate_saddles_and_increasing_directions() if  s == saddle]
            #            saddle_counter_directions = [next_node for s, next_node in self.iterate_saddles_and_decreasing_directions() if  s == saddle]
            #            
            #            direction_graph = self.get_increasing_graph()
            #        else:
            #            saddle_directions = [next_node for s, next_node in self.iterate_saddles_and_decreasing_directions() if  s == saddle]
            #            saddle_counter_directions = [next_node for s, next_node in self.iterate_saddles_and_increasing_directions() if  s == saddle]
            #            direction_graph = self.get_decreasing_graph()
            #        saddle_counter_directions = np.setdiff1d(saddle_counter_directions, path)
            #        
            #        # possible path trajectories
            #        optional_paths = [graph_methods.get_chain_from(direction_graph, next_node) for next_node in saddle_directions]
            #        print('optional_paths:', optional_paths)
            #        optional_paths = [np.concatenate([short_path, optional_path]) for optional_path in optional_paths]
            #        print('optional_paths:', optional_paths)
            #        
            #        counter_paths = [counter_path[::-1] for counter_path in self.paths if counter_path[0] == saddle]
            #        counter_paths_merges = [common_suffix(counter_path, short_path) for counter_path in counter_paths]
            #        counter_path = counter_paths[np.argmax([len(counter_path_merged) for counter_path_merged in counter_paths_merges])]
            #        # the paths which should not be intersected
            #        constraining_paths = [np.append(counter_path, next_node) for next_node in saddle_counter_directions]
            #        print('constraining_paths:', constraining_paths)
            #
            #        for optional_path in optional_paths:
            #
            #            if not np.any([triangletools.merging_paths_intersects(optional_path, constraining_path, self.faces) for constraining_path in constraining_paths]):
            #                self.paths[i] = optional_path
            #                break
        return self.paths
        
    
    
    def iterate_paths(self):
        """
        Iterate paths: increasing (from sadles to local maxima) and decreasing (from saddles to local minima)

        Yield:
        ------
        path : np.array[int]
            Paths represented as list of consequtive vertex indices from saddle to local maximum/minimum
        """
        for path in self.get_paths():
            yield path


    def define_decomposition_by_paths(self):
        """
        Defines which face is in which quadrangle

        Returns:
        --------
        self.faces_components_by_paths: integer np.array length M
            The indices are face indices
            The values are quadrangle indices
        """
        # FIXME: This method can split one quadrangle into many when increasing and deceasing paths intersect in one nodes
        # TODO: There should be another solution (probably, without the face adjacency graph)
        if hasattr(self, 'faces_components_by_paths'):
            return self.faces_components_by_paths
        
        # represent face_graph edges as pairs of vertex ids triplets
        edges = list(self.get_face_graph().edges)
        edges_face_repr = self.faces[np.array(edges)]

        # define the edges of the complex coresponding the edges of the graph
        edges_edge_repr = -1*np.ones([edges_face_repr.shape[0], 2], dtype=int)
        for j0, j1 in itertools.product(itertools.combinations(range(3), 2), repeat=2):
            cond = (edges_face_repr[:, 0, list(j0)] == edges_face_repr[:, 1, list(j1)]).all(axis=1)
            edges_edge_repr[cond] = edges_face_repr[cond][:, 0, (list(j0))]
        edges_edge_repr = np.sort(edges_edge_repr, axis=1)

        # edges of the complex inclued into paths
        paths_edges = np.concatenate([np.transpose([path[:-1], path[1:]]) for path in self.get_paths()])
        paths_edges = np.unique(np.sort(paths_edges, axis=1), axis=0)

        # remove edges from graph, which are included into paths
        remove_conds = edges_edge_repr[:, None, :, None] == paths_edges[None, :, None, :]
        remove_conds = (remove_conds[:, :, 0, 0] & remove_conds[:, :, 1, 1]).any(axis=-1)

        edges_to_remove = [edge for edge, cond in zip(edges, remove_conds) if cond]

        face_graph_reduced = self.get_face_graph()
        face_graph_reduced.remove_edges_from(edges_to_remove)

        # define components for faces
        self.faces_components_by_paths = -np.ones(self.n_faces, dtype=int)
        for i, comp in enumerate(nx.connected_components(face_graph_reduced)):
            self.faces_components_by_paths[list(comp)] = i

        return self.faces_components_by_paths
    

    def get_surrounding_faces(self, chain, level=0):
        """
        Returns the indices of faces surrounding the chain

        Parameters:
        -----------
        chain: np.array[int] or list[int]
            The indices of the chain vertices

        level: int
            How far triangles from the chain we take

        Returns:
        --------
        surrounding_faces: np.array[int]
            The indices of the faces surrounding the chain.
        """
        faces_vertex_permutations = self.faces[:, [list(perm) for perm in itertools.permutations(range(3), 2)]][..., None]
        chain_edges = np.array([chain[:-1], chain[1:]])
        surrounding_faces0 = np.argwhere((faces_vertex_permutations == chain_edges).all(axis=-2).any(axis=(-1, -2))).reshape(-1)
        dist = nx.multi_source_dijkstra_path_length(self.get_face_graph(), sources=set(surrounding_faces0))
        surrounding_faces = np.array([key for key, value in dist.items() if value <= level])
        return surrounding_faces
    

    def get_face_distances_from_chain(self, chain, weight_function='area'):
        """
        Compute the distances of faces from the chain.
        1. Defines the face adjacency graph
        2. For each edge in this graph define the weight in the given way
        3. Define the weighted path length as the distance

        Parameters:
        -----------
        chain: np.array[int] or list[int]
            The indices of the chain vertices

        weight_function : str or function
            The way how to compute the edge weight

            Labeled options:
            ``'area'``: area of the quadrangle of centers of the faces and vertices of the common edge

            ``'common-edge-length'``: the distance between vertices of the common edge

            ``'centers-distance'``: the distance between centers of the faces (along the surface)

        Returns:
        --------
        dist: np.array length M
            The distance from the chain for each vertex
        """
        if type(weight_function) is str:
            if weight_function.lower() == 'area':
                def weight_function(face0, face1):
                    a, b = self.vertices[np.intersect1d(face0, face1)]
                    c0 = self.vertices[face0].mean(axis=0)
                    c1 = self.vertices[face1].mean(axis=0)
                    return geometry.triangle_area(a, b, c0) + geometry.triangle_area(a, b, c1)
            elif weight_function.lower() in ['length', 'common-edge-length']:
                def weight_function(face0, face1):
                    a, b = self.vertices[np.intersect1d(face0, face1)]
                    return np.linalg.norm(a - b)
            elif weight_function.lower() in ['distance', 'centers-distance']:
                def weight_function(face0, face1):
                    a, b = self.vertices[np.intersect1d(face0, face1)]
                    m = 0.5*(a + b)
                    c0 = self.vertices[face0].mean(axis=0)
                    c1 = self.vertices[face1].mean(axis=0)
                    return np.linalg.norm(c0 - m) + np.linalg.norm(c1 - m)
            else:
                raise ValueError("Expected weight_function parameter be None, str from ['area', 'length', 'distance'] or the function of 2 parameters")
        if weight_function is None:
            weight = 'weight'
        else:
            def weight(u, v, *args, **kwargs):
                return weight_function(self.faces[u], self.faces[v]) if u != v else 0

        faces_vertex_permutations = self.faces[:, [list(perm) for perm in itertools.permutations(range(3), 2)]][..., None]
        chain_edges = np.array([chain[:-1], chain[1:]])
        surrounding_faces0 = np.argwhere((faces_vertex_permutations == chain_edges).all(axis=-2).any(axis=(-1, -2))).reshape(-1)
        dist = nx.multi_source_dijkstra_path_length(self.get_face_graph(), sources=set(surrounding_faces0), weight=weight)
        dist = np.array([dist[i] for i in range (self.n_faces)])
        return dist
    
    
    def get_surrounding_disks_face_indices(self, chain, weight_function='area', max_distance=np.inf, ignore_disk_condition=True):
        """
        Returns the surrounding faces which is a (shellable) disk

        Parameters:
        -----------
        chain: np.array[int] or list[int]
            The indices of the chain vertices

        weight_function : str
            The way how to compute the edge weight
            ``'area'``: area of the quadrangle of centers of the faces and vertices of the common edge

            ``'common-edge-length'``: the distance between vertices of the common edge

            ``'centers-distance'``: the distance between centers of the faces (along the surface)

        max_distance : int or np.inf
            The maximal distance from the chain

        ignore_disk_condition : bool
            If ``False`` there is no check that the surrounding area is homotopically equivalent to a disk.
            This is incorrect, but much faster. Need for tests.

        Returns:
        --------
        surrounding_disks_face_indices: np.array[int]

        """
        face_distances = self.get_face_distances_from_chain(chain, weight_function)
        face_order = np.argsort(face_distances)
        if ignore_disk_condition:
            face_add_status = np.ones_like(face_order, dtype=bool)
        else:
            face_add_status = np.zeros_like(face_order, dtype=bool)
            face_add_status[face_distances == 0] = True
            # shelling construction
            for i in face_order[face_distances <= max_distance]:
                if triangletools.is_homotopy_preserving_face_addition(self.faces[face_add_status], self.faces[i]):
                    face_add_status[i] = True
        surrounding_disks_face_indices = np.argwhere(face_add_status & (face_distances <= max_distance)).reshape(-1)
        
        return surrounding_disks_face_indices


    def get_geodesic_homotopic_to_edge_chain(self, chain, weight_function='area', max_distance=np.inf, with_distance=False):
        """
        Returns the geodesic, homotopic to a given chain

        Parameters:
        -----------
        chain: np.array[int] or list[int]
            The indices of the chain vertices

        weight_function : str
            The way how to compute the edge weight
            ``'area'``: area of the quadrangle of centers of the faces and vertices of the common edge

            ``'common-edge-length'``: the distance between vertices of the common edge

            ``'centers-distance'``: the distance between centers of the faces (along the surface)

        max_distance : int or np.inf
            The maximal distance from the chain

        with_distance: bool
            if ``True`` also returns the length of the geodesic

        Returns:
        --------
        geopath: np.array shape (:, D)
            The consequtive vertices along the geodesic
        
        geo_distance: float
            The length of geodesic
        """
        surrounding_disk_faces = self.faces[self.get_surrounding_disks_face_indices(chain, weight_function, max_distance)]
        
        #face_components = triangletools.get_faces_components(surrounding_disk_faces)
        #if (len(face_components) != 1) or np.intersect1d(np.unique(surrounding_disk_faces), chain).size != len(chain):
        #    print('Wow! Uncomputable geodesic, some faces are missed. Returning the original path')
        #    geopath = self.vertices[chain]
        #    if with_distance:
        #        geo_distance = np.linalg.norm(geopath[1:] - geopath[:-1], axis=1).sum()
        #        return geopath, geo_distance
        #    return geopath
        
        V, F, old2new, new2old = triangletools.compact_mesh(self.vertices, surrounding_disk_faces)
        source_vid = old2new[chain[0]]
        target_vid = old2new[chain[-1]]

        geo = geodesic.PyGeodesicAlgorithmExact(V, F)
        geo_distance, geopath = geo.geodesicDistance(source_vid, target_vid)

        if len(geopath) == 0:
            warnings.warn('Wow! Uncomputable geodesic, some faces are missed. Returning the original path')
            geopath = self.vertices[chain]
            geo_distance = np.linalg.norm(geopath[1:] - geopath[:-1], axis=1).sum()
            
        if with_distance:
            return geopath, geo_distance
        return geopath
    

    def iterate_geodesics_homotopic_to_paths(self, weight_function='area', max_distance=np.inf, with_distance=False):
        """
        Iterate the geodesics, homotopic to paths

        Parameters:
        -----------
        weight_function : str
            The way how to compute the edge weight
            ``'area'``: area of the quadrangle of centers of the faces and vertices of the common edge

            ``'common-edge-length'``: the distance between vertices of the common edge

            ``'centers-distance'``: the distance between centers of the faces (along the surface)

        max_distance : int or np.inf
            The maximal distance from the chain

        with_distance: bool
            if ``True`` also returns the length of the geodesic

        Returns:
        --------
        geopath: np.array shape (:, D)
            The consequtive vertices along the geodesic
        
        geo_distance: float
            The length of geodesic
        """
        for path in self.get_paths():
            yield self.get_geodesic_homotopic_to_edge_chain(path, weight_function, max_distance, with_distance)


    def get_paths_graph(self) -> nx.MultiGraph:
        """
        Returns the graph relating the critical points and paths

        Returns:
        --------
        g_paths: nx.MultiGraph
            A multigraph where:

            - Nodes are the vertex indices of critical points.
            - Node attribute ``critical_type`` is one of
            ``"min"``, ``"max"``, or ``"saddle"``.
            - Each edge represents a path connecting two critical points.
            - Edge attribute ``path`` contains a copy of the NumPy array of
            vertex indices defining the path, including both endpoints.
        """
        self.define_critical_points()

        g_paths = nx.MultiGraph()
        g_paths.add_nodes_from(self.mins, critical_type="min")
        g_paths.add_nodes_from(self.maxs, critical_type="max")
        g_paths.add_nodes_from(self.saddles, critical_type="saddle")

        for path in self.get_paths():
            v0, v1 = path[[0, -1]]
            g_paths.add_edge(v0, v1, path=path.copy())
        return g_paths
    

    def get_paths_graph_after_cancellations(self, protected_nodes=[]) -> nx.MultiGraph:
        """
        Returns the graph relating the critical points and paths after cancellations

        Parameters:
        -----------
        protected_nodes : list[int]
            The nodes, which can't be canceled

        Returns:
        --------
        g_paths : nx.MultiGraph
            A multigraph where:

            - Nodes are the vertex indices of critical points.
            - Node attribute ``critical_type`` is one of
            ``"min"``, ``"max"``, or ``"saddle"``.
            - Each edge represents a path connecting two critical points.
            - Edge attribute ``path`` contains a copy of the NumPy array of
            vertex indices defining the path, including both endpoints.
        """
        g_simplifyed = simplify_graph(self.get_paths_graph(), self.values, protected_nodes=protected_nodes)
        
        return g_simplifyed



    def get_geodesics_graph(self, weight_function='area', max_distance=np.inf, simplify=True, protected_nodes=[]) -> nx.MultiGraph:
        """

        Parameters:
        -----------
        weight_function : str
            The way how to compute the edge weight
            ``'area'``: area of the quadrangle of centers of the faces and vertices of the common edge

            ``'common-edge-length'``: the distance between vertices of the common edge

            ``'centers-distance'``: the distance between centers of the faces (along the surface)

        max_distance : int or np.inf
            The maximal distance from the chain

        simplify: bool
            If ``True`` do cancellations
            
        protected_nodes : list[int]
            The nodes, which can't be canceled
        
        Returns:
        --------
        g_paths : nx.MultiGraph
            A multigraph where:

            - Nodes are the vertex indices of critical points.
            - Node attribute ``critical_type`` is one of
            ``"min"``, ``"max"``, or ``"saddle"``.
            - Each edge represents a path connecting two critical points.
            - Edge attribute ``path`` contains a copy of the NumPy array of
            vertex indices defining the path, including both endpoints.
            - Edge attribute ``geopath`` contains a geodesic points homotopic to path.
        """
        if simplify:
            g = self.get_paths_graph_after_cancellations(protected_nodes=protected_nodes)
        else:
            g = self.get_paths_graph()
        
        for u, v, key, data in g.edges(keys=True, data=True):
            if np.isin(data['path'], protected_nodes).all():
                data["geopath"] = self.vertices[data["path"]]
            else:
                data["geopath"] = self.get_geodesic_homotopic_to_edge_chain(data["path"], weight_function, max_distance, with_distance=False)
        return g

        

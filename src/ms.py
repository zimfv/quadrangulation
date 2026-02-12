import itertools

import numpy as np
import scipy as sp
import networkx as nx

from src import graph_methods




class MorseSmale:
    def __init__(self, faces, values, vertices=None, forest_method='steepest'):
        """
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
        if forest_method.lower() == 'spaning':
            self._get_increasing_graph_method = graph_methods.get_spaning_increasing_graph

    def distance(self, index0, index1):
        """
        """
        if self.vertices is not None:
            return np.linalg.norm(self.vertices[index1] - self.vertices[index0], axis=-1)
        else:
            return np.array(index0 != index1).astype(int)

    def gradient(self, index0, index1):
        """
        """
        #val = self.values[index1] - self.values[index0]
        #if self.vertices is not None:
        #    val /= np.linalg.norm(self.vertices[index1] - self.vertices[index0], axis=1)
        val = (self.values[index1] - self.values[index0])/self.distance(index0, index1)
        return val
    
    def get_edge_graph(self) -> nx.Graph:
        """
        """
        if not hasattr(self, 'edge_graph'):
            self.edge_graph = nx.Graph()
            self.edge_graph.add_nodes_from(range(self.n_vertices))
            edges = np.unique(np.sort(np.concatenate(self.faces[:, [[0, 1], [0, 2], [1, 2]]]), axis=1), axis=0)
            self.edge_graph.add_edges_from(edges)
        return self.edge_graph.copy()
    
    def get_increasing_graph(self) -> nx.DiGraph:
        """
        """
        if not hasattr(self, 'increasing_graph'):
            self.increasing_graph = self._get_increasing_graph_method(self.get_edge_graph(), 
                                                                      gradient_function=self.gradient, 
                                                                      distance_function=self.distance)
        return self.increasing_graph.copy()

    def get_decreasing_graph(self) -> nx.DiGraph:
        """
        """
        if not hasattr(self, 'decreasing_graph'):
            self.decreasing_graph = self._get_increasing_graph_method(self.get_edge_graph(), 
                                                                      gradient_function=lambda i0, i1: self.gradient(i1, i0), 
                                                                      distance_function=self.distance)
        return self.decreasing_graph.copy()
    
    def define_critical_points(self):
        """
        """
        self.get_edge_graph()

        self.mins = []
        self.maxs = []
        self.saddles = []
        for node in range(self.n_vertices):
            neighbors = np.array(list(self.edge_graph.neighbors(node)))
            neighbors_gradients = self.gradient(node, neighbors) 
            if (neighbors_gradients > 0).all():
                self.mins.append(node)
            elif (neighbors_gradients < 0).all():
                self.maxs.append(node)
            else:
                graph_lower_neighborhood = self.edge_graph.subgraph(neighbors[neighbors_gradients < 0])
                graph_higher_neighborhood = self.edge_graph.subgraph(neighbors[neighbors_gradients > 0])
                regular = nx.is_connected(graph_lower_neighborhood) and nx.is_connected(graph_higher_neighborhood)
                if not regular:
                    self.saddles.append(node)
    
    def iterate_saddles_and_increasing_directions(self):
        """
        """
        try:
            self.saddles
        except AttributeError:
            self.define_critical_points()
        for saddle in self.saddles:
            neighbors = np.array(list(self.edge_graph.neighbors(saddle)))
            neighbors_gradients = self.gradient(saddle, neighbors) 
            graph_higher_neighborhood = self.edge_graph.subgraph(neighbors[neighbors_gradients > 0])
            for component in nx.connected_components(graph_higher_neighborhood):
                next_node = list(component)[self.gradient(saddle, list(component)).argmax()]
                yield (saddle, next_node)
                
    def iterate_saddles_and_decreasing_directions(self):
        """
        """
        try:
            self.saddles
        except AttributeError:
            self.define_critical_points()
        for saddle in self.saddles:
            neighbors = np.array(list(self.edge_graph.neighbors(saddle)))
            neighbors_gradients = self.gradient(saddle, neighbors) 
            graph_lower_neighborhood = self.edge_graph.subgraph(neighbors[neighbors_gradients < 0])
            for component in nx.connected_components(graph_lower_neighborhood):
                next_node = list(component)[self.gradient(saddle, list(component)).argmin()]
                yield (saddle, next_node)
        
    def iterate_paths(self):
        """
        """
        for saddle, next_node in self.iterate_saddles_and_increasing_directions():
            path = graph_methods.get_chain_from(self.get_increasing_graph(), next_node)
            path = np.append(saddle, path)
            yield path
        for saddle, next_node in self.iterate_saddles_and_decreasing_directions():
            path = graph_methods.get_chain_from(self.get_decreasing_graph(), next_node)
            path = np.append(saddle, path)
            yield path

    def get_paths(self):
        """
        """
        if not hasattr(self, 'paths'):
            self.paths = list(self.iterate_paths())
        return self.paths

    def get_face_graph(self):
        """
        """
        if not hasattr(self, 'face_graph'):
            self.face_graph = nx.Graph()
            self.face_graph.add_nodes_from(range(self.n_faces))
            for (i0, face0), (i1, face1) in itertools.combinations(enumerate(self.faces), 2):
                intersection = np.intersect1d(face0, face1)
                if len(intersection) == 2:
                    self.face_graph.add_edge(i0, i1, intersection=intersection)
        return self.face_graph.copy()
    
    def define_decomposition_by_paths(self):
        """
        """
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
        """
        faces_vertex_permutations = self.faces[:, [list(perm) for perm in itertools.permutations(range(3), 2)]][..., None]
        chain_edges = np.array([chain[:-1], chain[1:]])
        surrounding_faces0 = np.argwhere((faces_vertex_permutations == chain_edges).all(axis=-2).any(axis=(-1, -2))).reshape(-1)
        dist = nx.multi_source_dijkstra_path_length(self.get_face_graph(), sources=set(surrounding_faces0))
        surrounding_faces = np.array([key for key, value in dist.items() if value <= level])
        return surrounding_faces
    
    def get_surrounding_disk_cuts(self, chain):
        """
        """
        pass

    def get_geodesic_homotopic_to_edge_chain(self, chain, max_level=2):
        """
        """
        pass

        

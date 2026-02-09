import itertools

import numpy as np
import scipy as sp
import networkx as nx


def get_increasing_graph(edge_graph, gradient_function):
    """
    """
    increasing_graph = nx.DiGraph()
    increasing_graph.add_nodes_from(edge_graph.nodes)
    for node in edge_graph.nodes():
        neighbors = list(edge_graph.neighbors(node))
        grad_vals = gradient_function(node, neighbors)
        if (grad_vals > 0).any():
            increasing_graph.add_edge(node, neighbors[grad_vals.argmax()])
    return increasing_graph

def get_chain_from(graph: nx.DiGraph, start):
    """
    """
    chain = [start]
    while graph.out_degree(chain[-1]) == 1:
        chain.append(next(iter(graph.successors(chain[-1]))))
    return chain

class MorseSmale:
    def __init__(self, faces, values, vertices=None):
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
    
    def gradient(self, index0, index1):
        """
        """
        val = self.values[index1] - self.values[index0]
        if self.vertices is not None:
            val /= np.linalg.norm(self.vertices[index1] - self.vertices[index0], axis=1)
        return val
    
    def get_edge_graph(self) -> nx.Graph:
        """
        """
        try:
            return self.edge_graph
        except AttributeError:
            self.edge_graph = nx.Graph()
            self.edge_graph.add_nodes_from(range(self.n_vertices))
            edges = np.unique(np.sort(np.concatenate(self.faces[:, [[0, 1], [0, 2], [1, 2]]]), axis=1), axis=0)
            self.edge_graph.add_edges_from(edges)
            return self.edge_graph
    
    def get_increasing_graph(self) -> nx.DiGraph:
        """
        """
        try:
            return self.increasing_graph
        except AttributeError:
            self.increasing_graph = get_increasing_graph(self.get_edge_graph(), self.gradient)
            return self.increasing_graph

    def get_decreasing_graph(self) -> nx.DiGraph:
        """
        """
        try:
            return self.decreasing_graph
        except AttributeError:
            self.decreasing_graph = get_increasing_graph(self.get_edge_graph(), lambda i0, i1: self.gradient(i1, i0))
            return self.decreasing_graph
    
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
            path = get_chain_from(self.get_increasing_graph(), next_node)
            path = np.append(saddle, path)
            yield path
        for saddle, next_node in self.iterate_saddles_and_decreasing_directions():
            path = get_chain_from(self.get_decreasing_graph(), next_node)
            path = np.append(saddle, path)
            yield path
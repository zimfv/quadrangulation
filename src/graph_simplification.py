import numpy as np
import networkx as nx

def concatenate_paths(path0, path1):
    """
    Concatenate two paths, allowing automatic reversal
    if endpoints match in opposite orientation.
    """

    if len(path0) == 0:
        return path1
    if len(path1) == 0:
        return path0

    if np.array_equal(path0[-1], path1[0]):
        return np.concatenate([path0[:-1], path1])

    if np.array_equal(path0[-1], path1[-1]):
        return np.concatenate([path0[:-1], path1[::-1]])

    if np.array_equal(path0[0], path1[-1]):
        return np.concatenate([path1[:-1], path0])

    if np.array_equal(path0[0], path1[0]):
        return np.concatenate([path1[::-1][:-1], path0])

    raise ValueError("Paths do not share endpoints.")


def make_cancellation(graph: nx.MultiGraph, values, node0, node1=None):
    """
    Makes the cancellation of birth-death pair in the paths graph.

    Pramaters:
    ----------
    graph: nx.MultiGraph
        Multigraph
        Nodes should have 'critical_type' attribute with values: 'max', 'min' or 'saddle'.
        This function should be sed in the mothod MorseSmale.get_paths_graph, and also applyable to the resulting graph.

    values: array

    node0: int
        The index of the canceling vertex.
        This expect to be local maxima or minima and the coresponding node in the graph expect to have degree 1 or 2.

    node1: int
        The index of the saddle, paired with canceling vertex

    """
    if (graph.degree(node0) not in [1, 2]) or (graph.nodes[node0].get('critical_type') not in ['min', 'max']):
        raise ValueError('node0 expect to be an index of local maxima or minima degree 1 or 2.')
    if node1 is None:
        neighbors = list(graph.neighbors(node0))
        neighbors_vals = np.array([values[node] for node in neighbors])
        node1 = neighbors[np.argmin(abs(neighbors_vals - values[node0]))]
    if graph.nodes[node1].get('critical_type') != 'saddle':
        raise ValueError('node1 expect to be an index of saddle')
    

    if graph.degree(node1) == 4:
        
        graph_after = graph.copy()
        graph_after.remove_nodes_from([node0, node1])

        if graph.degree(node0) == 2:
            #
            path1, path2 = [data['path'] for u, v, key, data in graph.edges(node1, keys=True, data=True) if graph.nodes[v].get('critical_type') == graph.nodes[node0].get('critical_type')]
            path3 = [data['path'] for u, v, key, data in graph.edges(node0, keys=True, data=True) if (graph.nodes[v].get('critical_type') == 'saddle') and not np.isin(node1, data['path'])][0]
            
            new_path = concatenate_paths(concatenate_paths(path1, path2), path3)

            new_edge_node0 = [node for node in graph.neighbors(node0) if node != node1][0]
            new_edge_node1 = [v for u, v, key, data in graph.edges(node1, keys=True, data=True) if graph.nodes[v].get('critical_type') == graph.nodes[node0].get('critical_type')]
            new_edge_node1 = [v for v in new_edge_node1 if v != node0][0]
            graph_after.add_edge(new_edge_node0, new_edge_node1, path=new_path)

        return graph_after
    
    else:
        graph_after = graph.copy()
        

    raise ValueError('Not yet solved case with a monkey saddle')


def simplify_graph(graph: nx.MultiGraph, values):
    """
    """
    graph_after = graph.copy()
    wrong_nodes = [node for node in graph_after.nodes() if graph_after.degree(node) in [1, 2]]
    while len(wrong_nodes) > 0:
        graph_after = make_cancellation(graph_after, values, wrong_nodes[0])
        wrong_nodes = [node for node in graph_after.nodes() if graph_after.degree(node) in [1, 2]]

    return graph_after


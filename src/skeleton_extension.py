import numpy as np
import scipy as sp
import networkx as nx
import igl

from src.skeleton_function import get_chains



def get_surrounding_chains(faces, center, defined_values=[]):
    """
    """
    surrounding_faces = faces[(faces == center).any(axis=1)]
    surrounding_edges = surrounding_faces[surrounding_faces != center].reshape(-1, 2)
    surrounding_nodes = np.unique(surrounding_edges)
    surrounding_graph = nx.Graph()
    surrounding_graph.add_nodes_from(surrounding_nodes)
    surrounding_graph.add_edges_from(surrounding_edges)
    surrounding_defined_values = np.intersect1d(defined_values, surrounding_nodes)

    artificial_nodes = np.max(list(surrounding_graph.nodes())) + np.arange(1, 3)
    for artificial_node in artificial_nodes:
        surrounding_graph.add_edges_from([(artificial_node, outer) for outer in surrounding_defined_values])
    

    surrounding_chains = [chain for chain in get_chains(surrounding_graph) if (len(chain) > 2) and not np.isin(chain, artificial_nodes).any()]
    return surrounding_chains
    

def have_boundary_neighbor(faces, boundary_indices, vertices_indices):
    """
    """
    adj = igl.adjacency_matrix(faces)[boundary_indices, :][:, vertices_indices]
    res = adj.toarray().sum(axis=0) > 0
    return res


def iterate_all_maximal_paths_in_dag(G: nx.DiGraph):
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("G must be a DAG")

    sources = [n for n in G.nodes if G.in_degree(n) == 0]

    def dfs(path):
        u = path[-1]
        succ = list(G.successors(u))
        if not succ:                 # sink => path is maximal
            yield path
            return
        for v in succ:
            yield from dfs(path + [v])

    for s in sources:
        yield from dfs([s])
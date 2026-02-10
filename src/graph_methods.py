import numpy as np
import networkx as nx


def get_chain_from(graph: nx.DiGraph, start):
    """
    """
    chain = [start]
    while graph.out_degree(chain[-1]) == 1:
        chain.append(next(iter(graph.successors(chain[-1]))))
    return chain

'''
def directed_from_sym_antisym(G, sym_func, antisym_func, weight_attr="weight", tie="skip", zero_tol=0.0, make_weight_nonneg=False):
    """
    Build a DiGraph from (usually undirected) graph G using:
      - direction from sign of antisym_func(u,v)
      - weight from sym_func(u,v)

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        If DiGraph, we still iterate over edges as given.
    sym_func : callable
        sym_func(u, v) -> weight value (symmetric in u,v).
        Can be vectorized, but we call it per-edge here.
    antisym_func : callable
        antisym_func(u, v) -> signed value (antisymmetric in u,v).
    tie : str
        What to do when antisym is (near) zero:
        - "skip": don't add an edge
        - "both": add both u->v and v->u with same weight
        - "arbitrary": choose u->v
    zero_tol : float
        Consider antisym == 0 when abs(antisym) <= zero_tol.
    make_weight_nonneg : bool
        If True, store abs(sym) as weight.

    Returns
    -------
    H : nx.DiGraph
    """
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))

    for u, v, edata in G.edges(data=True):
        anti = antisym_func(u, v)
        if abs(anti) <= zero_tol:
            if tie == "skip":
                continue
            w = sym_func(u, v)
            if make_weight_nonneg:
                w = abs(w)
            if tie == "both":
                H.add_edge(u, v, **{weight_attr: w})
                H.add_edge(v, u, **{weight_attr: w})
            elif tie == "arbitrary":
                H.add_edge(u, v, **{weight_attr: w})
            else:
                raise ValueError("tie must be 'skip', 'both', or 'arbitrary'")
            continue

        # choose direction by sign
        src, dst = (u, v) if anti > 0 else (v, u)

        w = sym_func(u, v)  # symmetric, so order doesn't matter
        if make_weight_nonneg:
            w = abs(w)

        H.add_edge(src, dst, **{weight_attr: w})
    return H
'''

def directed_from_sym_antisym(G, sym_func, antisym_func, weight_attr="weight", tie="skip", zero_tol=0.0, make_weight_nonneg=False):
    """
    Build a DiGraph from (usually undirected) graph G using:
      - direction from sign of antisym_func(u,v)
      - weight from sym_func(u,v)

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        If DiGraph, we still iterate over edges as given.
    sym_func : callable
        sym_func(u, v) -> weight value (symmetric in u,v).
        Can be vectorized, but we call it per-edge here.
    antisym_func : callable
        antisym_func(u, v) -> signed value (antisymmetric in u,v).
    tie : str
        What to do when antisym is (near) zero:
        - "skip": don't add an edge
        - "both": add both u->v and v->u with same weight
        - "arbitrary": choose u->v
    zero_tol : float
        Consider antisym == 0 when abs(antisym) <= zero_tol.
    make_weight_nonneg : bool
        If True, store abs(sym) as weight.

    Returns
    -------
    H : nx.DiGraph
    """
    edges = np.array(list(G.edges))
    weights = sym_func(edges[:, 0], edges[:, 1])
    if make_weight_nonneg:
        weights = abs(weights)
    grads = antisym_func(edges[:, 0], edges[:, 1])
    edges[grads < 0] = edges[grads < 0][:, [1, 0]]
    if tie == 'both':
        edges = np.concatenate([edges, edges[abs(grads) <= zero_tol]][:, [1, 0]])
        weights = np.concatenate([weights, weights[abs(grads) <= zero_tol]])
    H = nx.DiGraph()
    H.add_weighted_edges_from([(e0, e1, w) for (e0, e1), w in zip(edges, weights)], weight=weight_attr)
    return H



def get_steepest_increasing_graph(edge_graph, gradient_function, **kwargs):
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


def get_spaning_increasing_graph(edge_graph, gradient_function, distance_function):
    """
    """
    directed_edge_graph = directed_from_sym_antisym(edge_graph, sym_func=distance_function, antisym_func=gradient_function,
                                                    weight_attr="weight", tie="skip", zero_tol=0.0, make_weight_nonneg=False)
    #directed_edge_graph.add_weighted_edges_from([('super', node, 0) for node in directed_edge_graph.nodes() if directed_edge_graph.in_degree(node) == 0])
    directed_edge_graph.add_weighted_edges_from([(node, 'super', 0) for node in directed_edge_graph.nodes() if directed_edge_graph.out_degree(node) == 0])
    directed_edge_graph = directed_edge_graph.reverse()

    increasing_graph = nx.algorithms.tree.branchings.minimum_spanning_arborescence(directed_edge_graph, attr='weight', default=0)
    increasing_graph.remove_node('super')
    increasing_graph = increasing_graph.reverse()
    return increasing_graph

import numpy as np
import scipy as sp
import cvxpy as cp
import networkx as nx
import igl

from src.mesh_topology import get_skeleton_graph




def get_chains(graph: nx.Graph, similar_start_end_of_cycle=False):
    """
    Returns the list of all chains of the graph.

    A chain is a maximal path whose internal nodes have degree 2
    (endpoints have degree != 2), or (if a connected component is a
    pure cycle where all nodes have degree 2) the whole cycle as one chain.

    Returns
    -------
    list[list[Hashable]]
        Each chain is returned as an ordered list of nodes.
        For cycle-components, the chain is a cyclic order with the first node
        repeated at the end to make the cycle explicit.
    """
    G = graph

    # Degree in a MultiGraph can be >2 because of parallel edges; this function assumes simple graphs,
    # but we still use G.degree(v) which NetworkX defines for any graph.
    deg = dict(G.degree())

    visited_undirected_edges = set()  # store edges as (min(u,v), max(u,v))

    def mark_edge(u, v):
        a, b = (u, v) if u <= v else (v, u)
        visited_undirected_edges.add((a, b))

    def edge_marked(u, v):
        a, b = (u, v) if u <= v else (v, u)
        return (a, b) in visited_undirected_edges

    chains = []

    # 1) Build chains starting from "junctions" / endpoints: nodes with degree != 2
    for start in G.nodes:
        if deg[start] == 2:
            continue

        for nbr in G.neighbors(start):
            if edge_marked(start, nbr):
                continue

            chain = [start]
            prev = start
            cur = nbr
            mark_edge(prev, cur)

            # Walk forward while we're in degree-2 nodes (internal chain nodes)
            while True:
                chain.append(cur)

                if deg[cur] != 2:
                    # We hit a terminal/junction node: chain ends here.
                    break

                # Choose the next neighbor that isn't the previous node
                nbs = list(G.neighbors(cur))
                # In a simple graph with deg==2, there are exactly 2 neighbors.
                nxt = nbs[0] if nbs[0] != prev else nbs[1]

                if edge_marked(cur, nxt):
                    # We've already consumed this edge; stop to avoid duplicates.
                    break

                prev, cur = cur, nxt
                mark_edge(prev, cur)

            chains.append(chain)

    # 2) Remaining edges (if any) belong to components where all nodes have degree 2 => cycles.
    # We treat each such component as one chain (cycle order).
    for comp in nx.connected_components(G):
        comp_nodes = list(comp)
        if not comp_nodes:
            continue
        if any(deg[v] != 2 for v in comp_nodes):
            continue  # not a pure cycle component

        # If all edges in this component are already visited, skip.
        comp_edges = list(G.subgraph(comp_nodes).edges())
        if all(edge_marked(u, v) for (u, v) in comp_edges):
            continue

        # Build an explicit cyclic order by walking until we return to start.
        start = comp_nodes[0]
        nbrs = list(G.neighbors(start))
        if len(nbrs) != 2:
            # Shouldn't happen in a pure cycle, but be defensive.
            continue

        chain = [start]
        prev = start
        cur = nbrs[0]
        mark_edge(prev, cur)

        while True:
            chain.append(cur)
            if cur == start:
                break

            nbs = list(G.neighbors(cur))
            nxt = nbs[0] if nbs[0] != prev else nbs[1]

            # If we're about to close, allow it; otherwise avoid reusing visited edges.
            if edge_marked(cur, nxt) and nxt != start:
                break

            prev, cur = cur, nxt
            mark_edge(prev, cur)

            if cur == start:
                chain.append(start)
                break

        # Ensure it's a cycle representation: end equals start.
        if chain[-1] != start:
            chain.append(start)

        chains.append(chain)

    if not similar_start_end_of_cycle:
        for i, chain in enumerate(chains):
            if chain[0] == chain[-1]:
                chains[i] = chain[:-1]
    chains = [np.array(chain) for chain in chains]

    return chains


def get_integer_chain_representation(chains, vertices, p=3, omega=1.0, additional_lengths=[], print_status=True):
    """
    Get the best integer chain representation defining and solving a mixed-integer optimization problem.

    Parameters:
    -----------
    chains: list[np.array[int]]
        The chains of the boundary defined as indices of vertices
    
    vertices: np.array
        The vertices of the complex
    
    p: int
        Minimal number of edges between local minimas and local maximas

    omega: float
        The result is closer to chain lengths against uniform

    additional_lengths: array[int]
        Additional lengths to consider for the optimization

    Returns:
    --------
    n : np.array length len(chains)
        The integre representation of chain lengths
    """
    if isinstance(vertices, dict):
        chain_edge_lengths = [np.linalg.norm([vertices[v0] - vertices[v1] for v0, v1 in itertools.pairwise(chain)], axis=-1) for chain in chains]
    else:
        chain_edge_lengths = [np.linalg.norm(vertices[chain[1:]] - vertices[chain[:-1]], axis=-1) for chain in chains]
    chain_lengths_total = np.array([np.sum(edge_lengths) for edge_lengths in chain_edge_lengths])
    chain_lengths_max = np.array([np.max(edge_lengths) for edge_lengths in chain_edge_lengths])

    right_constrains = chain_lengths_total/(2*p*chain_lengths_max)

    # variables
    n = cp.Variable(len(chains) + len(additional_lengths), integer=True)
    s = cp.Variable()
    constraints = [n >= 1, n[:len(chains)] <= right_constrains]

    # objective
    ls = np.concatenate([chain_lengths_total, additional_lengths])
    objective = cp.Minimize(omega*cp.sum_squares(n - s*ls) + (1 - omega)*cp.sum_squares(n))

    # problem
    prob = cp.Problem(objective, constraints)

    # choose a mixed-integer solver you have installed
    #prob.solve(solver=cp.GUROBI, verbose=True)   # or SCIP / CPLEX / MOSEK if available
    prob.solve(solver=cp.SCIP, verbose=False)   # or SCIP / CPLEX / MOSEK if available
    
    if print_status:
        print("status:", prob.status)
        print("objective:", prob.value)
    if prob.status == 'optimal':
        print("n:", np.round(n.value).astype(int))
        print("s:", s.value)

        n = np.round(n.value).astype(int)
        n = n[:len(chains)]

        return n
    

    n = (ls / ls.min()).round().astype(int)[:len(chains)]

    return n


def get_skeleton_values(graph, vertices_pos, default_value=0, p=3, omega=1.0, additional_lengths=[]):
    """
    Defines the values on the skeleton graph.

    Parameters:
    -----------
    graph: nx.Graph
        Skeleton graph

    vertices_pos: dict | array
        The positionss of nodes/vertices in the skeleton graph

    default_value: float
        The default value for the function
    
    p: int
        Minimal number of edges between local minimas and local maximas

    omega: float
        The result is closer to chain lengths against uniform

    additional_lengths: array[int]
        Additional lengths to consider for the optimization

        
    Returns:
    --------
    vec_val: np.array(graph.number_of_nodes())
        The defined values in the nodes of the skeleton graph
    """
    if isinstance(vertices_pos, dict):
        d = len(next(graph.values().__iter__()))
        n = np.max(graph.nodes()).astype(int) + 1
        vertices = np.nan*np.zeros([n, d])
        for node, pos in vertices_pos.items():
            vertices[node] = pos
    else:
        vertices = np.array(vertices_pos)

    chains = get_chains(graph, similar_start_end_of_cycle=True)
    chain_lengths = np.array([np.linalg.norm(vertices[chain[1:]] - vertices[chain[:-1]], axis=-1).sum() for chain in chains])
    chain_lengths_int = get_integer_chain_representation(chains, vertices, p=p, omega=omega, additional_lengths=additional_lengths)

    node_list = np.array(list(graph.nodes))
    node_to_vertex = -1*np.ones(vertices.shape[0], dtype=int)
    node_to_vertex[node_list] = np.arange(len(node_list))

    vec_val = np.zeros(graph.number_of_nodes()) + default_value
    for i, (chain, length, length_int) in enumerate(zip(chains, chain_lengths, chain_lengths_int)):
        vec_t = np.append(0, np.linalg.norm(vertices[chain][1:] - vertices[chain][:-1], axis=1).cumsum()) / length
        vec_val[node_to_vertex[chain]] = np.cos(2*length_int*np.pi*vec_t)
        
    return vec_val


def detect_mins(faces, bnd_indices, bnd_vals):
    """
    """
    adj = igl.adjacency_matrix(faces)
    adj = adj[bnd_indices][:, bnd_indices].toarray().astype(bool)

    cond = (~adj | (bnd_vals > bnd_vals.reshape(-1, 1))).all(axis=1)
    bnd_mins = bnd_indices[cond]
    return bnd_mins


def detect_maxs(faces, bnd_indices, bnd_vals):
    """
    """
    return detect_mins(faces, bnd_indices, -bnd_vals)


def detect_cons(faces, bnd_indices):
    """
    """
    adj = igl.adjacency_matrix(faces)
    adj = adj[bnd_indices][:, bnd_indices].toarray()
    #bnd_cons = bnd_indices[adj.sum(axis=1) == 0]
    bnd_cons = bnd_indices[~adj.any(axis=1)]
    return bnd_cons


def detect_skeleton_parameters(vertices, faces, default_value=1, p=3, omega=1.0, additional_lengths=[]):
    """
    """
    skeleton_graph = get_skeleton_graph(faces)
    skeleton_indices = np.array(list(skeleton_graph.nodes()))
    if len(skeleton_indices) > 0:
        skeleton_values = get_skeleton_values(skeleton_graph, vertices, default_value=default_value, p=p, omega=omega, additional_lengths=additional_lengths)
        skeleton_mins = detect_mins(faces, skeleton_indices, skeleton_values)
        skeleton_maxs = detect_maxs(faces, skeleton_indices, skeleton_values)
        skeleton_cons = detect_cons(faces, skeleton_indices)
    else:
        skeleton_values, skeleton_mins, skeleton_maxs, skeleton_cons = np.zeros(shape=(4, 0), dtype=int)

    return skeleton_indices, skeleton_values, skeleton_mins, skeleton_maxs, skeleton_cons
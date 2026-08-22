import numpy as np
import scipy as sp
import networkx as nx
import cvxpy as cp
import igl

import itertools

from src import triangletools




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


def iterate_integral_lines_over_the_boundary(faces, bnd_indices, bnd_values, direction_from_min_to_max=True):
    """
    """
    bnd_faces = faces[np.isin(faces, bnd_indices).sum(axis=1) == 2]
    bnd_edges = bnd_faces[np.isin(bnd_faces, bnd_indices)].reshape(-1, 2)
    bnd_thirds = bnd_faces[~np.isin(bnd_faces, bnd_indices)]
    assert (np.sort(bnd_faces, axis=1) == np.sort(np.hstack([bnd_edges, bnd_thirds.reshape(-1, 1)]), axis=1)).all()

    assert np.isin(bnd_edges, bnd_indices).all()


    bnd_val_dict = {key: value for key, value in zip(bnd_indices, bnd_values)}
    bnd_edge_vals = np.vectorize(bnd_val_dict.get)(bnd_edges)
    if direction_from_min_to_max:
        bnd_edges[bnd_edge_vals[:, 1] < bnd_edge_vals[:, 0]] = bnd_edges[bnd_edge_vals[:, 1] < bnd_edge_vals[:, 0]][:, [1, 0]]
    else:
        bnd_edges[bnd_edge_vals[:, 1] > bnd_edge_vals[:, 0]] = bnd_edges[bnd_edge_vals[:, 1] > bnd_edge_vals[:, 0]][:, [1, 0]]

    bnd_graph = nx.DiGraph()
    for (e0, e1), third_vertex in zip(bnd_edges, bnd_thirds):
        bnd_graph.add_edge(e0, e1, third=third_vertex)
    
    for path in iterate_all_maximal_paths_in_dag(bnd_graph):
        yield np.array(path)


def iterate_edge_common_neighbors_over_path(faces, path):
    """
    """
    adj = igl.adjacency_matrix(faces).toarray().astype(bool)

    for e0, e1 in itertools.pairwise(path):
        common_neighbors = np.argwhere(adj[e0] & adj[e1]).ravel()
        yield (e0, e1), common_neighbors


def get_inequality_pairs(faces, boundary_indices, boundary_values, boundary_mins=None, boundary_maxs=None, boundary_cons=None, conus_strategy='saddle', 
                         remove_boundary=True):
    """
    """
    if boundary_mins is None:
        boundary_mins = detect_mins(faces, boundary_indices, boundary_values)
    if boundary_maxs is None:
        boundary_maxs = detect_maxs(faces, boundary_indices, boundary_values)
    if boundary_cons is None:
        boundary_cons = []

    adj = igl.adjacency_matrix(faces)
    adj_where =  np.argwhere(adj)

    inequalities_with_maxs =  adj_where[np.isin(adj_where[:, 0], boundary_maxs)]

    inequalities_on_paths = np.zeros(shape=[0, 2], dtype=int)
    for path in iterate_integral_lines_over_the_boundary(faces, boundary_indices, boundary_values, direction_from_min_to_max=True):
        for (e0, e1), common_neighbors in iterate_edge_common_neighbors_over_path(faces, path):
            #if e0 in boundary_mins:
            #    continue
            e0_neighbors = np.argwhere(adj[e0])[:, 1]
            e0_neighbors = e0_neighbors[~np.isin(e0_neighbors, boundary_indices)]
            neighbor_pairs = np.transpose([e0_neighbors, e0_neighbors])
            neighbor_pairs[:, 0] = e1
            inequalities_on_paths = np.vstack([inequalities_on_paths, neighbor_pairs, [[e1, e0]]])


    #defined_values = np.argwhere(~np.isnan(boundary_values)).ravel()
    defined_values = boundary_indices[~np.isnan(boundary_values)]
    inequalities_with_saddles = np.zeros(shape=[0, 2], dtype=int)
    for saddle in boundary_mins:
        for chain in get_surrounding_chains(faces, saddle, defined_values):
            v0, v1 = chain[[0, -1]]
            v2 = chain[len(chain)//2]
            chain_pairs = [[v0, saddle], 
                           [v1, saddle], 
                           [saddle, v2]]
            chain_pairs0 = np.transpose([chain[:len(chain)//2], chain[1 : len(chain)//2 + 1]])
            chain_pairs1 = np.transpose([chain[len(chain)//2 + 1:], chain[len(chain)//2:-1]])

            inequalities_with_saddles = np.vstack([inequalities_with_saddles, chain_pairs, chain_pairs0, chain_pairs1])

    
    inequalities_with_cons = np.zeros(shape=[0, 2], dtype=int)
    for conus in boundary_cons:
        conus_neighbors = np.argwhere(adj[conus])[:, 1]
        if conus_strategy.lower() == 'max':
            pairs = np.transpose([conus_neighbors, conus_neighbors])
            pairs[:, 0] = conus
        elif conus_strategy.lower() == 'min':
            pairs = np.transpose([conus_neighbors, conus_neighbors])
            pairs[:, 1] = conus
        elif conus_strategy.lower() == 'saddle':
            chain = get_surrounding_chains(faces, conus)[0]
            i0 = 0
            i1 = len(chain) // 4
            i2 = len(chain) // 2
            i3 = len(chain) - i1
            i4 = len(chain)
            chain = np.append(chain, chain[0])
            pairs = np.vstack([[(conus, chain[i0]), (conus, chain[i2]), (chain[i1], conus), (chain[i3], conus)], 
                               np.transpose([chain[i0 + 1: i1 + 1], chain[i0: i1]]), 
                               np.transpose([chain[i1: i2], chain[i1 + 1: i2 + 1]]), 
                               np.transpose([chain[i2 + 1: i3 + 1], chain[i2: i3]]), 
                               np.transpose([chain[i3: i4], chain[i3 + 1: i4 + 1]]), ])
            

        inequalities_with_cons = np.vstack([inequalities_with_cons, pairs])



    inequality_pairs = np.vstack([inequalities_with_maxs, 
                                  inequalities_with_saddles, 
                                  inequalities_on_paths, 
                                  inequalities_with_cons
                                  ])
    inequality_pairs = np.unique(inequality_pairs, axis=0)

    inequality_pairs = inequality_pairs[inequality_pairs[:, 0] != inequality_pairs[:, 1]]

    if remove_boundary:
        inequality_pairs = inequality_pairs[~np.isin(inequality_pairs, boundary_indices).all(axis=1)]


    inequality_pairs = np.unique(inequality_pairs, axis=0)

    return inequality_pairs



def solve_boundary_hermonics(faces, boundary_indices, boundary_values, vertices=None, 
                             boundary_mins=None, boundary_maxs=None, 
                             boundary_cons=None, conus_strategy='saddle', eps=None):
    """
    """
    if eps is None:
        eps = np.unique(boundary_values)
        eps = (eps[1:] - eps[:-1]).mean()

    inequality_pairs = get_inequality_pairs(faces, boundary_indices, boundary_values, 
                                            boundary_mins=boundary_mins, boundary_maxs=boundary_maxs, 
                                            boundary_cons=boundary_cons, conus_strategy=conus_strategy)
    second_boundary_indices = np.setdiff1d(np.unique(inequality_pairs), boundary_indices)
    working_vertices = np.concatenate([boundary_indices, second_boundary_indices])
    
    A = np.zeros((len(inequality_pairs), len(second_boundary_indices)))
    b = np.zeros(len(inequality_pairs))

    for i, (v0, v1) in enumerate(inequality_pairs):
        if v0 in second_boundary_indices:
            A[i, second_boundary_indices == v0] = -1
        if v1 in second_boundary_indices:
            A[i, second_boundary_indices == v1] = 1

        bv1 = boundary_values[boundary_indices == v1].sum()
        bv0 = boundary_values[boundary_indices == v0].sum()
        if bv0 != 0:
            bv0 -= eps
        if bv1 != 0:
            bv1 -= eps
        b[i] = bv0 - bv1

    adj = igl.adjacency_matrix(faces)[working_vertices][:, working_vertices].toarray()
    if vertices is None:
        D = np.eye(len(adj))*adj.sum(axis=1)
        L = D - adj
    else:
        dists = np.linalg.norm(vertices[working_vertices][:, None, :] - vertices[working_vertices][None, :, :], axis=-1)
        W = np.zeros_like(dists)
        W[adj > 0] = 1.0 / dists[adj > 0]
        D = np.diag(W.sum(axis=1))
        L = D - W

    x = cp.Variable(len(second_boundary_indices))
    v = cp.hstack([boundary_values, x])
    objective = cp.Minimize(cp.quad_form(v, L))
    constraints = [A @ x <= b]


    problem = cp.Problem(objective, constraints)
    problem.solve()

    #print("status:", problem.status)
    #print("optimal value:", problem.value)
    #print("x* =", x.value)

    second_boundary_values = x.value

    # check if there are edges with similar values
    '''
    edges = np.unique(np.sort(np.argwhere(adj), axis=1), axis=0)
    edges_vals = v.value[edges]
    edges_similar = edges[edges_vals[:, 0] == edges_vals[:, 1]]
    print(edges_similar)
    if edges_similar.size > 0:
        for e0, e1 in edges_similar:
            surrounding_nodes = np.unique(np.argwhere(adj[[e0, e1]])[:, 1])
            surrounding_vals = np.unique(v.value[surrounding_nodes])
            print(f'               surrounding_vals: {surrounding_vals}')
            surrounding_eps = 0.25*(surrounding_vals[1:] - surrounding_vals[:-1]).min()
            if e0 > len(boundary_indices):
                second_boundary_values[e0 - len(boundary_indices)] -= surrounding_eps
            if e1 > len(boundary_indices):
                second_boundary_values[e1 - len(boundary_indices)] += surrounding_eps
    '''
    return second_boundary_indices, second_boundary_values




def solve_second_boundary_hermonics(vertices, faces, boundary_indices, boundary_values, 
                                    boundary_mins=None, boundary_maxs=None, 
                                    boundary_cons=None, conus_strategy='saddle', eps=None, k=2):
    """
    """
    second_boundary_indices, second_boundary_values = solve_boundary_hermonics(faces, boundary_indices, boundary_values, 
                                                                               boundary_mins=boundary_mins, boundary_maxs=boundary_maxs, 
                                                                               boundary_cons=boundary_cons, conus_strategy=conus_strategy, eps=eps)

    faces_small = faces[~np.isin(faces, boundary_indices).any(axis=1)]
    vertices_small, faces_small, big2small, small2big = triangletools.compact_mesh(vertices, faces_small)

    second_boundary_indices_small = big2small[second_boundary_indices]
    bc = second_boundary_values[second_boundary_indices_small != -1].reshape(-1, 1)
    b = second_boundary_indices_small[second_boundary_indices_small != -1]

    u_small = igl.harmonic(vertices_small, faces_small, b, bc, k).ravel()

    u = np.nan*np.ones(len(vertices))
    u[small2big[np.arange(len(u_small))]] = u_small
    u[second_boundary_indices] = second_boundary_values
    u[boundary_indices] = boundary_values

    return u


def dirichlet_laplacian_eigenfunctions(vertices, faces, boundary_indices=None, k=10, which="LM"):
    """
    Compute k Dirichlet eigenfunctions of the Laplace–Beltrami operator:
        (-L_ff) u = lambda * M_ff u, with u(boundary)=0.

    Returns:
        evals: (k,) eigenvalues
        evecs_full: (n, k) eigenvectors padded with zeros on boundary
    """
    L = igl.cotmatrix(vertices, faces)
    M = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_VORONOI)

    if boundary_indices is None:
        boundary_vertices = np.ravel(igl.is_border_vertex(faces))
    else:
        boundary_vertices = np.isin(np.arange(len(vertices), boundary_indices))
    interrior_vertices = ~boundary_vertices 

    Lff = L[interrior_vertices][:, interrior_vertices]
    Mff = M[interrior_vertices][:, interrior_vertices]

    A = (-Lff).tocsc()
    B = Mff.tocsc()

    k_eff = min(k, A.shape[0] - 2) if A.shape[0] > 2 else 1

    # Shift-invert near 0 to get smallest eigenvalues robustly
    evals, evecs_f = sp.sparse.linalg.eigsh(A, k=k_eff, M=B, sigma=0.0, which=which)

    # Pad back to full vectors with zeros on boundary
    evecs = np.zeros((vertices.shape[0], evecs_f.shape[1]), dtype=float)
    evecs[interrior_vertices, :] = evecs_f

    # Sort
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def second_boundary_dirichlet_laplacian_eigenfunctions(vertices, faces, boundary_indices=None, k=10, which="LM"):
    """
    """
    if boundary_indices is None:
        boundary_indices = np.argwhere(igl.is_border_vertex(faces)).ravel()

    faces_small = faces[~np.isin(faces, boundary_indices).any(axis=1)]
    vertices_small, faces_small, big2small, small2big = triangletools.compact_mesh(vertices, faces_small)

    evals, evecs_small = dirichlet_laplacian_eigenfunctions(vertices_small, faces_small, k=k, which=which)
    evecs = np.zeros([vertices.shape[0], k], dtype=float)
    evecs[small2big[np.unique(faces_small)]] = evecs_small
    
    return evals, evecs


def second_boundary_dirichlet_laplacian_eigenfunction_plus_harmonic(vertices, faces, boundary_indices, boundary_values,  
                                                                    eval_harmonic_ratio = None, 
                                                                    boundary_mins=None, boundary_maxs=None, 
                                                                    boundary_cons=None, conus_strategy='saddle', eps=None, 
                                                                    harmonic_power=2, eigen_index=5, which='LM'):
    """
    """
    u = solve_second_boundary_hermonics(vertices, faces, boundary_indices, boundary_values, 
                                        boundary_mins=boundary_mins, boundary_maxs=boundary_maxs, 
                                        boundary_cons=boundary_cons, conus_strategy=conus_strategy, 
                                        eps=eps, k=harmonic_power)

    #if eigen_index == 0:
    #    eval = 0
    #else:
    #    evals, evecs = second_boundary_dirichlet_laplacian_eigenfunctions(vertices, faces, boundary_indices=boundary_indices, k=eigen_index, which=which)
    #    eval = evals[0]
    #
    #if eval_harmonic_ratio is not None:
    #    eval *= eval_harmonic_ratio*np.max(abs(u))/np.max(abs(eval))
    #
    #res = eval + u

    res = u
    return res
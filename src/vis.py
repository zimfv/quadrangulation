import numpy as np
import networkx as nx
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.ms import MorseSmale


def get_pv_mesh(vertices, faces):
    """
    Docstring for get_pv_mesh
    
    :param faces: Description
    """
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=faces.dtype), faces]).ravel()
    mesh = pv.PolyData(vertices, faces_pv)
    return mesh

def _pos_to_array(G, pos):
    """
    """
    nodes = list(G.nodes())
    if isinstance(pos, dict):
        P = np.asarray([pos[n] for n in nodes], dtype=float)
    else:
        P = np.asarray(pos, dtype=float)
        if P.shape != (len(nodes), 3):
            raise ValueError("pos must be dict {node:(x,y,z)} or array (n,3) matching G.nodes() order")
    return nodes, P


def _edges_polydata(nodes, P, edges):
    """
    """
    node_index = {n: i for i, n in enumerate(nodes)}
    lines = []
    for u, v in edges:
        if u in node_index and v in node_index:
            i, j = node_index[u], node_index[v]
            lines.extend([2, i, j])

    pd = pv.PolyData()
    pd.points = P
    pd.lines = np.array(lines, dtype=np.int64) if lines else np.empty((0,), dtype=np.int64)
    return pd, node_index


def add_graph_to_plotter(pl: pv.Plotter, G: nx.Graph, pos, offset=(0.0, 0.0, 0.0), 
                         node_size=10, edge_width=2, directed=None, arrow_scale=0.4, add_labels=False, 
                         node_color="tomato", edge_color="black", arrow_color="royalblue"):
    """
    """
    nodes, P = _pos_to_array(G, pos)
    P = P + np.asarray(offset, dtype=float)

    if directed is None:
        directed = G.is_directed()

    pts = pv.PolyData(P)
    edges_pd, node_index = _edges_polydata(nodes, P, G.edges())

    pl.add_mesh(edges_pd, color=edge_color, line_width=edge_width, render_lines_as_tubes=True)
    pl.add_mesh(pts, color=node_color, point_size=node_size, render_points_as_spheres=True)

    if add_labels:
        pl.add_point_labels(P, [str(n) for n in nodes], point_size=0, font_size=12)

    if directed and G.number_of_edges() > 0:
        starts, dirs = [], []
        for u, v in G.edges():
            a = P[node_index[u]]
            b = P[node_index[v]]
            d = b - a
            norm = np.linalg.norm(d)
            if norm == 0:
                continue
            d = d / norm

            # put arrow near the target node
            starts.append(b - arrow_scale * d)
            dirs.append(d)

        if len(starts) > 0:
            starts = np.asarray(starts, dtype=float)
            dirs = np.asarray(dirs, dtype=float)

            arrow_src = pv.PolyData(starts)
            arrow_src["vec"] = dirs
            arrow_src.set_active_vectors("vec")  # <-- IMPORTANT

            # per-arrow scaling (1 scalar per point)
            arrow_src["scale"] = np.full((starts.shape[0],), arrow_scale, dtype=float)

            arrows = arrow_src.glyph(
                orient="vec",          # <-- IMPORTANT (use this array)
                scale="scale",         # per-point scaling
                geom=pv.Arrow(),       # default arrow geometry is fine
            )

            pl.add_mesh(arrows, color=arrow_color)


def add_graph_to_plotter_by_components(pl: pv.Plotter, G: nx.Graph, pos, offset=(0.0, 0.0, 0.0), 
                                       node_size=10, edge_width=2, directed=None, arrow_scale=0.4, add_labels=False, 
                                       node_cmap="rainbow", edge_cmap="rainbow", arrow_cmap="rainbow", categotical_cmaps=False):
    """
    """
    node_cmap = plt.get_cmap(node_cmap)
    edge_cmap = plt.get_cmap(edge_cmap)
    arrow_cmap = plt.get_cmap(arrow_cmap)

    color_mltiplyer = 1 if categotical_cmaps else 1/nx.number_connected_components(G.to_undirected())
    for i, component in enumerate(nx.connected_components(G.to_undirected())):
        node_color = mcolors.to_hex(node_cmap(color_mltiplyer*i))
        edge_color = mcolors.to_hex(node_cmap(color_mltiplyer*i))
        arrow_color = mcolors.to_hex(node_cmap(color_mltiplyer*i))
        subG = G.subgraph(component)
        add_graph_to_plotter(pl=pl, G=subG, node_color=node_color, edge_color=edge_color, arrow_color=arrow_color, pos=pos, offset=offset, 
                             node_size=node_size, edge_width=edge_width, directed=directed, arrow_scale=arrow_scale, add_labels=add_labels)


def plot_segmentation_forests(ms: MorseSmale, window_size=(1000, 500), 
                              plot_complex=True, complex_color='white', complex_opacity=0.4, 
                              plot_critical_points=True, min_color="lime", saddle_color="pink", max_color='orangered', point_size=16, 
                              node_size=10, edge_width=2, directed=None, arrow_scale=0.4, add_labels=False,
                              node_cmap="tab10", edge_cmap="tab10", arrow_cmap="tab10", categotical_cmaps=True):
    """
    """
    if plot_complex:
        faces_pv = np.hstack([np.full((ms.faces.shape[0], 1), 3, dtype=ms.faces.dtype), ms.faces]).ravel()
        complex_mesh = pv.PolyData(ms.vertices, faces_pv)
    pos = {i: p for i, p in enumerate(ms.vertices)}
    
    pl = pv.Plotter(shape=(1, 2), window_size=window_size)

    pl.subplot(0, 0)
    pl.add_text('Increasing Forest', font_size=12)
    if plot_complex:
        pl.add_mesh(complex_mesh, color=complex_color, smooth_shading=False, show_edges=True, opacity=complex_opacity)
    add_graph_to_plotter_by_components(pl, ms.get_increasing_graph(), pos, 
                                       node_size=node_size, edge_width=edge_width, directed=directed, arrow_scale=arrow_scale, add_labels=add_labels,
                                       node_cmap=node_cmap, edge_cmap=edge_cmap, arrow_cmap=arrow_cmap, categotical_cmaps=categotical_cmaps)
    if plot_critical_points:
        pl.add_points(ms.vertices[ms.mins], color=min_color, point_size=point_size, render_points_as_spheres=True)
        pl.add_points(ms.vertices[ms.saddles], color=saddle_color, point_size=point_size, render_points_as_spheres=True)
        pl.add_points(ms.vertices[ms.maxs], color=max_color, point_size=point_size, render_points_as_spheres=True)
    
    pl.subplot(0, 1)
    pl.add_text('Decrasing Forest', font_size=12)
    if plot_complex:
        pl.add_mesh(complex_mesh, color=complex_color, smooth_shading=False, show_edges=True, opacity=complex_opacity)
    add_graph_to_plotter_by_components(pl, ms.get_decreasing_graph(), pos, 
                                       node_size=node_size, edge_width=edge_width, directed=directed, arrow_scale=arrow_scale, add_labels=add_labels,
                                       node_cmap=node_cmap, edge_cmap=edge_cmap, arrow_cmap=arrow_cmap, categotical_cmaps=categotical_cmaps)
    if plot_critical_points:
        pl.add_points(ms.vertices[ms.mins], color=min_color, point_size=point_size, render_points_as_spheres=True)
        pl.add_points(ms.vertices[ms.saddles], color=saddle_color, point_size=point_size, render_points_as_spheres=True)
        pl.add_points(ms.vertices[ms.maxs], color=max_color, point_size=point_size, render_points_as_spheres=True)
    
    pl.link_views()
    return pl



def add_critical_points_to_plotter(pl: pv.Plotter, ms: MorseSmale, min_color="lime", saddle_color="pink", max_color='orangered', point_size=12):
    """
    """
    pl.add_points(ms.vertices[ms.mins], color=min_color, point_size=point_size, render_points_as_spheres=True)
    pl.add_points(ms.vertices[ms.saddles], color=saddle_color, point_size=point_size, render_points_as_spheres=True)
    pl.add_points(ms.vertices[ms.maxs], color=max_color, point_size=point_size, render_points_as_spheres=True)


def add_paths_to_plotter(pl: pv.Plotter, ms: MorseSmale, path_color='white', linewidth=4, eps=0.0, path_cmap=None, categorical_cmap=None):
    """
    """
    faces_pv = np.hstack([np.full((ms.faces.shape[0], 1), 3, dtype=ms.faces.dtype), ms.faces]).ravel()
    mesh = pv.PolyData(ms.vertices, faces_pv)

    path_lines = []
    vertices_adds = np.zeros(ms.n_vertices)
    vertices_normals = mesh.point_normals

    for path in ms.iterate_paths():
        path_lines.append(ms.vertices[path] + eps*vertices_adds[path].reshape(-1, 1)*vertices_normals[path])
        vertices_adds[path[1:-1]] += 1

    if not (path_cmap is None):
        cmap = plt.get_cmap(path_cmap)
        if categorical_cmap is None:
            categorical_cmap = isinstance(cmap, mcolors.ListedColormap)
        path_colors = np.arange(len(path_lines))
        if not categorical_cmap:
            path_colors = path_colors/path_colors.max()
        path_colors = cmap(path_colors)
        path_colors = [mcolors.to_hex(i) for i in path_colors]
    elif type(path_color) is str:
        path_colors = [path_color for line in path_lines]
    else:
        path_colors = path_color
    for line, color in zip(path_lines, path_colors):
        pl.add_mesh(pv.lines_from_points(line), color=color, line_width=linewidth)


def add_complex_to_plotter(pl: pv.Plotter, ms: MorseSmale, opacity=1.0, smooth_shading=False, show_edges=True, 
                           with_values=True, data_title='values', color='white', value_cmap="viridis",
                           with_critical_points=True, min_color="lime", saddle_color="pink", max_color='orangered', point_size=12, 
                           with_paths=True, path_color='white', path_cmap=None, linewidth=4, eps=0.0):
    faces_pv = np.hstack([np.full((ms.faces.shape[0], 1), 3, dtype=ms.faces.dtype), ms.faces]).ravel()
    mesh = pv.PolyData(ms.vertices, faces_pv)
    if with_values:
        mesh.point_data[data_title] = ms.values
        pl.add_mesh(mesh, scalars=data_title, cmap=value_cmap, smooth_shading=smooth_shading, show_edges=show_edges, opacity=opacity)
    else:
        pl.add_mesh(mesh, color=color, smooth_shading=smooth_shading, show_edges=show_edges, opacity=opacity)
    if with_critical_points:
        add_critical_points_to_plotter(pl, ms, min_color=min_color, saddle_color=saddle_color, max_color=max_color, point_size=point_size)
    if with_paths:
        add_paths_to_plotter(pl, ms, path_color=path_color, path_cmap=path_cmap, linewidth=linewidth, eps=eps)



def plot_paths(ms: MorseSmale, color='white', value_cmap="viridis", 
               min_color="lime", saddle_color="pink", max_color='orangered', point_size=12, 
               path_color='white', path_cmap="rainbow", linewidth=4, 
               window_size=(1000, 500), eps=0.1, opacity0=1.0, opacity1=0.6):
    """
    """
    pl = pv.Plotter(shape=(1, 2), window_size=window_size)
    pl.subplot(0, 0)
    add_complex_to_plotter(pl, ms, opacity=opacity0, smooth_shading=False, show_edges=True, 
                           with_values=True, value_cmap=value_cmap, data_title='values', 
                           with_critical_points=True, min_color=min_color, saddle_color=saddle_color, max_color=max_color, point_size=point_size, 
                           with_paths=True, path_color=path_color, linewidth=linewidth, eps=0.0)
    
    pl.subplot(0, 1)
    add_complex_to_plotter(pl, ms, opacity=opacity1, smooth_shading=False, show_edges=True, 
                           with_values=False, color=color, 
                           with_critical_points=True, min_color=min_color, saddle_color=saddle_color, max_color=max_color, point_size=point_size, 
                           with_paths=True, path_cmap=path_cmap, linewidth=linewidth, eps=eps)

    pl.link_views()
    return pl

    

def plot_ms_comparition(mss: list[MorseSmale], titles=None, font_size=12, 
                        window_size=None, 
                        min_color="lime", saddle_color="pink", max_color='orangered', point_size=12,  
                        plot_values=True, opacity0=1.0, value_cmap='viridis', path_color='white', linewidth=4,
                        plot_paths=True, color='white', opacity1=0.4, path_cmap='rainbow', eps=0.1, 
                        plot_segmentation_forests=True, plot_complex_with_segmentation=True, opacity2=0.2, 
                        plot_critical_points_with_segmentation=True, graph_crytical_point_size=16, 
                        node_size=10, edge_width=2, directed=None, arrow_scale=0.4, add_labels=False,
                        node_cmap="tab10", edge_cmap="tab10", arrow_cmap="tab10", categotical_cmaps=True
                        ):
    n_cols = len(mss)
    n_rows = np.array([plot_values, plot_paths, plot_segmentation_forests, plot_segmentation_forests]).astype(int).sum()
    if window_size is None:
        window_size = (400*n_cols, 400*n_rows)
    pl = pv.Plotter(shape=(n_rows, n_cols), window_size=window_size)

    if titles is None:
        titles = np.arange(len(mss)).astype(str)
    
    i_row = 0

    if plot_values:
        for i_col, (ms, title) in enumerate(zip(mss, titles)):
            pl.subplot(i_row, i_col)
            pl.add_text(f'{title} - Vlaues', font_size=font_size)
            add_complex_to_plotter(pl, ms, opacity=opacity0, smooth_shading=False, show_edges=True, 
                                   with_values=True, value_cmap=value_cmap, data_title=f'{title} - Vlaues', 
                                   with_critical_points=True, min_color=min_color, saddle_color=saddle_color, max_color=max_color, point_size=point_size, 
                                   with_paths=True, path_color=path_color, linewidth=linewidth, eps=0.0)
        i_row += 1
    
    if plot_paths:
        for i_col, (ms, title) in enumerate(zip(mss, titles)):
            pl.subplot(i_row, i_col)
            pl.add_text(f'{title} - Paths', font_size=font_size)
            add_complex_to_plotter(pl, ms, opacity=opacity1, smooth_shading=False, show_edges=True, 
                                with_values=False, color=color, 
                                with_critical_points=True, min_color=min_color, saddle_color=saddle_color, max_color=max_color, point_size=point_size, 
                                with_paths=True, path_cmap=path_cmap, linewidth=linewidth, eps=eps)
        i_row += 1

    if plot_segmentation_forests:
        for i_col, (ms, title) in enumerate(zip(mss, titles)):
            pl.subplot(i_row, i_col)
            pl.add_text(f'{title} - Increasing Forest', font_size=font_size)
            if plot_complex_with_segmentation:
                faces_pv = np.hstack([np.full((ms.faces.shape[0], 1), 3, dtype=ms.faces.dtype), ms.faces]).ravel()
                complex_mesh = pv.PolyData(ms.vertices, faces_pv)
                pl.add_mesh(complex_mesh, color=color, smooth_shading=False, show_edges=True, opacity=opacity2)
            pos = {i: p for i, p in enumerate(ms.vertices)}
            add_graph_to_plotter_by_components(pl, ms.get_increasing_graph(), pos, 
                                               node_size=node_size, edge_width=edge_width, directed=directed, arrow_scale=arrow_scale, add_labels=add_labels, 
                                               node_cmap=node_cmap, edge_cmap=edge_cmap, arrow_cmap=arrow_cmap, categotical_cmaps=categotical_cmaps)
            if plot_critical_points_with_segmentation:
                add_critical_points_to_plotter(pl, ms, min_color=min_color, saddle_color=saddle_color, max_color=max_color, point_size=graph_crytical_point_size)
        
        i_row += 1
        for i_col, (ms, title) in enumerate(zip(mss, titles)):
            pl.subplot(i_row, i_col)
            pl.add_text(f'{title} - Decreasing Forest', font_size=font_size)
            if plot_complex_with_segmentation:
                faces_pv = np.hstack([np.full((ms.faces.shape[0], 1), 3, dtype=ms.faces.dtype), ms.faces]).ravel()
                complex_mesh = pv.PolyData(ms.vertices, faces_pv)
                pl.add_mesh(complex_mesh, color=color, smooth_shading=False, show_edges=True, opacity=opacity2)
            pos = {i: p for i, p in enumerate(ms.vertices)}
            add_graph_to_plotter_by_components(pl, ms.get_decreasing_graph(), pos, 
                                               node_size=node_size, edge_width=edge_width, directed=directed, arrow_scale=arrow_scale, add_labels=add_labels, 
                                               node_cmap=node_cmap, edge_cmap=edge_cmap, arrow_cmap=arrow_cmap, categotical_cmaps=categotical_cmaps)
            if plot_critical_points_with_segmentation:
                add_critical_points_to_plotter(pl, ms, min_color=min_color, saddle_color=saddle_color, max_color=max_color, point_size=graph_crytical_point_size)
        
    pl.link_views()
    return pl

        
            
            

    
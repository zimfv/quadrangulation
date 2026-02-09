import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.ms import MorseSmale


def plot_paths(ms: MorseSmale, 
               min_color="lime", saddle_color="pink", max_color='orangered', point_size=12, 
               value_cmap="viridis", path_cmap="rainbow", line_width=4, 
               window_size=(1000, 500), eps=0.05, opacity0=1.0, opacity1=0.6):
    """
    """
    path_lines = [ms.vertices[chain] for chain in ms.iterate_paths()]
    path_lines_shifted = []
    path_adds = np.zeros(ms.n_vertices)
    for path in ms.iterate_paths():
        path_adds[ms.mins + ms.maxs + ms.saddles] = 0
        path_lines_shifted.append(ms.vertices[path] + eps*path_adds[path].reshape(-1, 1))
        path_adds[path] += 1
    path_colors = [mcolors.to_hex(c) for c in plt.get_cmap(path_cmap)(np.linspace(0, 1, len(path_lines)))]
    np.random.shuffle(path_colors)

    faces_pv = np.hstack([np.full((ms.faces.shape[0], 1), 3, dtype=ms.faces.dtype), ms.faces]).ravel()
    mesh = pv.PolyData(ms.vertices, faces_pv)
    mesh.point_data["values"] = ms.values
    
    pl = pv.Plotter(shape=(1, 2), window_size=window_size)

    pl.subplot(0, 0)
    pl.add_mesh(mesh, scalars="values", cmap=value_cmap, smooth_shading=False, show_edges=True, opacity=opacity0)
    pl.add_points(ms.vertices[ms.mins], color=min_color, point_size=point_size, render_points_as_spheres=True)
    pl.add_points(ms.vertices[ms.saddles], color=saddle_color, point_size=point_size, render_points_as_spheres=True)
    pl.add_points(ms.vertices[ms.maxs], color=max_color, point_size=point_size, render_points_as_spheres=True)
    for line in path_lines:
        pl.add_mesh(pv.lines_from_points(line), color="white", line_width=line_width)

    
    pl.subplot(0, 1)
    pl.add_mesh(mesh, color='white', smooth_shading=False, show_edges=True, opacity=opacity1)
    pl.add_points(ms.vertices[ms.mins], color=min_color, point_size=point_size, render_points_as_spheres=True)
    pl.add_points(ms.vertices[ms.saddles], color=saddle_color, point_size=point_size, render_points_as_spheres=True)
    pl.add_points(ms.vertices[ms.maxs], color=max_color, point_size=point_size, render_points_as_spheres=True)
    for line, color in zip(path_lines_shifted, path_colors):
        pl.add_mesh(pv.lines_from_points(line), color=color, line_width=line_width)

    pl.link_views()
    return pl
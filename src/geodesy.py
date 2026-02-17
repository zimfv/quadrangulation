import numpy as np

import triangletools
from pygeodesic import geodesic





def compute_geodesic(vertices, faces, source_vid, target_vid, with_distance=False):
    """
    """
    faces = np.unique(np.sort(faces, axis=1), axis=0)


    geo_distance, geopath = 0, vertices[source_vid]


    V, F, old2new, new2old = triangletools.compact_mesh(vertices, faces)
    source_vid_i = old2new[source_vid]
    target_vid_i = old2new[target_vid]
    geo = geodesic.PyGeodesicAlgorithmExact(V, F)
    geo_distance_i, geopath_i = geo.geodesicDistance(source_vid, target_vid)

    geo_distance += geo_distance_i
    geopath = np.concatenate([geopath, geopath_i[1:]])
    if with_distance:
        return geopath, geo_distance
    return geopath
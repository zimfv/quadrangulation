import numpy as np



class MorseSmale:
    def __init__(self, faces, values, vertices=None):
        r"""
        Initialize the 2-dimensional simplicilial complex with N vertices and M 2-faces to quadrangulate

        Parameters:
        -----------
        faces: array shape (M, 3)
            The list of 2-faces of the simplicial complex

        values: array shape (N, )
            The filtration values of the vertices
        
        vertices: array shape (N, d) or None
            If the complex has an embeding in d-dimensional eucledean space, we can define the cords of the vertices
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


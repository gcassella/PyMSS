import PyMSS.primitives as primitives
import numpy as np

class Cuboid(primitives.Polyhedron):
    def __init__(self, J, M, scale_factor=[1,1,1], origin=[0,0,0]):
        vertices = [[1/2,1/2,1/2],\
                    [1/2,-1/2,1/2],\
                    [-1/2,-1/2,1/2],\
                    [-1/2,1/2,1/2],\
                    [1/2,1/2,-1/2],\
                    [1/2,-1/2,-1/2],\
                    [-1/2,-1/2,-1/2],\
                    [-1/2,1/2,-1/2]]

        winding = [[0,1,2,3],\
                   [0,4,5,1],\
                   [4,7,6,5],\
                   [7,3,2,6],\
                   [0,3,7,4],\
                   [5,6,2,1]]

        super().__init__(vertices, winding, J, M, scale_factor, origin)

        self._width = np.linalg.norm(np.subtract(self._vertices[3], self._vertices[0]))
        self._height = np.linalg.norm(np.subtract(self._vertices[1], self._vertices[0]))
        self._length = np.linalg.norm(np.subtract(self._vertices[4], self._vertices[0]))

    @primitives.Polyhedron.volume.getter
    def volume(self):
        return self._width*self._height*self._length

class Tetrahedron(primitives.Polyhedron):
    def __init__(self, J, M, scale_factor=[1,1,1], origin=[0,0,0]):
        vertices = [[1,0,-1/np.sqrt(2)],\
                    [-1,0,-1/np.sqrt(2)],\
                    [0,1,1/np.sqrt(2)],\
                    [0,-1,1/np.sqrt(2)]]

        winding = [[0,2,1],\
                   [0,3,2],\
                   [0,1,3],\
                   [1,2,3]]

        super().__init__(vertices, winding, J, M, scale_factor, origin)

    @primitives.Polyhedron.volume.getter
    def volume(self):
        # Use Heron's formula to find the area of one face

        detJ = np.absolute(np.linalg.det(self._simplex_jacobian_matrix))

        return 1/6 * detJ

    @property
    def _simplex_jacobian_matrix(self):
        # Gives the Jacobian matrix of the affine transformation
        # of this tetrahedron to the standard 3-simplex

        xx = self._vertices[1][0] - self._vertices[0][0]
        xy = self._vertices[2][0] - self._vertices[0][0]
        xz = self._vertices[3][0] - self._vertices[0][0]

        yx = self._vertices[1][1] - self._vertices[0][1]
        yy = self._vertices[2][1] - self._vertices[0][1]
        yz = self._vertices[3][1] - self._vertices[0][1]

        zx = self._vertices[1][2] - self._vertices[0][2]
        zy = self._vertices[2][2] - self._vertices[0][2]
        zz = self._vertices[3][2] - self._vertices[0][2]

        jacobian_matrix = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])

        return jacobian_matrix

    def _simplex_transform(self, r):
        # Transforms unit simplex coordinates u,v,w into
        # real space coordinates

        return np.add(self._vertices[0], np.dot(self._simplex_jacobian_matrix, r))

    def integrate(self, func, order):
        # Integrates function func over the tetrahedron
        # using Guass-Legendre Quadrature of order order

        # Jacobian of the n-simplex transformation
        detJ = np.absolute(np.linalg.det(self._simplex_jacobian_matrix))

        def transfunc(r):

            c = ((1-r[0])**2*(1-r[1]) / 64)
            x = (1+r[0])/2
            y = ((1-r[0])*(1+r[1]))/4
            z = ((1-r[0])*(1-r[1])*(1+r[2]))/8

            point = self._simplex_transform([x,y,z])
            integrand = np.multiply(func(point), detJ)

            return np.multiply(c, integrand)

        points, weights = np.polynomial.legendre.leggauss(order)

        accumulator = 0

        for i, u in enumerate(points):
            for j, v in enumerate(points):
                for k, w in enumerate(points):
                    accumulator = np.add(accumulator,
                                weights[i]*weights[j]*weights[k]*transfunc([u, v, w]))

        return accumulator

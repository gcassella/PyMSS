import sys
import numpy as np

def epsilon_metric(r1: list, r2: list):
    euclidean_metric2 = (r2[0] - r1[0])**2 + (r2[1] - r1[1])**2 + (r2[2] - r1[2])**2

    return np.sqrt(euclidean_metric2 + 1e-15)

def vector_norm(vector: list):
    magnitude = np.linalg.norm(vector)
    return [vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude]

class Edge:
    def __init__(self, vertices: list):
        if not len(vertices) == 2:
            return None
        
        self.vertices = vertices

    def edge_altitude(self, r: list):
        a = epsilon_metric(r, self.vertices[1])
        b = epsilon_metric(r, self.vertices[0])
        c = epsilon_metric(self.vertices[0], self.vertices[1])

        d = (a+b+c) / (a+b-c)

        return np.log(d)

    def vector_representation(self):
        return np.subtract(self.vertices[1], self.vertices[0])

class Face:
    def __init__(self, vertices: list, J: list, M=[0,0,1]):
        if not len(vertices) >= 3:
            return None

        self.M = M
        self.J = J

        self.vertices = vertices

        self.edges = []

        for k in range(len(vertices)-1):
            self.edges.append(Edge([vertices[k],vertices[k+1]]))
        self.edges.append(Edge([vertices[-1],vertices[0]]))

        # Calculate surface norm from two arbitrary edges

        vec_edge_a = self.edges[0].vector_representation()
        vec_edge_b = self.edges[1].vector_representation()

        self.norm = vector_norm(np.cross(vec_edge_a, vec_edge_b))

    def _sum_edge_altitude(self, r: list):
        accumulator = 0

        for edge in self.edges:
            vec_rep = edge.vector_representation()
            normed_vec_rep = vec_rep / np.linalg.norm(vec_rep)

            trip_prod = np.dot(np.cross(self.norm, np.subtract(edge.vertices[0] ,r)), normed_vec_rep)
            accumulator += edge.edge_altitude(r)*trip_prod

        return accumulator

    def _sum_edge_altitude_grad(self, r: list):
        accumulator = [0, 0, 0]

        for edge in self.edges:
            vec_rep = edge.vector_representation()
            normed_vec_rep = vec_rep / np.linalg.norm(vec_rep)

            edge_altitude_grad = np.cross(self.norm, normed_vec_rep) * edge.edge_altitude(r)
            accumulator = np.add(accumulator, edge_altitude_grad)

        return accumulator

    def _calculate_solid_angle(self, r: list):
        accumulator = 0

        for i in range(0,len(self.vertices),2):
            triangle_vertices = []

            triangle_vertices.append(self.vertices[i])
            triangle_vertices.append(self.vertices[(i+1)%len(self.vertices)])
            triangle_vertices.append(self.vertices[(i-1)%len(self.vertices)])

            a = np.subtract(triangle_vertices[0], r)
            b = np.subtract(triangle_vertices[1], r)
            c = np.subtract(triangle_vertices[2], r)
            a_mag = epsilon_metric(r, triangle_vertices[0])
            b_mag = epsilon_metric(r, triangle_vertices[1])
            c_mag = epsilon_metric(r, triangle_vertices[2])

            D = a_mag*b_mag*c_mag + c_mag*np.dot(a, b) + b_mag*np.dot(a, c) + c_mag*np.dot(b, c)

            argument = np.dot(a, np.cross(b, c))

            accumulator += 2*np.arctan2(argument, D)

        return accumulator

    def _get_face_altitude(self, r: list):
        height = np.dot(np.subtract(self.vertices[0], r), self.norm)
        return self._sum_edge_altitude(r) - self._calculate_solid_angle(r)*height

    def _get_face_altitude_grad(self, r: list):
        return np.add(self._sum_edge_altitude_grad(r), np.multiply(self._calculate_solid_angle(r),self.norm))

    def b_flux(self, r: list):
        from_curr = np.cross(self.J, self.norm) * self._get_face_altitude(r) * 1.26e-6 / (4*np.pi)
        from_magn = - np.cross(np.cross(self.M, self.norm), self._get_face_altitude_grad(r)) * 1.26e-6 / (4*np.pi)
        return np.add(from_curr, from_magn)

    def h_flux(self, r: list):
        return np.cross(self.J, self.norm) * self._get_face_altitude(r) * 1.26e-6 / (4*np.pi)
    
    def m_flux(self, r: list):
        return -np.cross(np.cross(self.M, self.norm), self._get_face_altitude_grad(r)) * 1.26e-6 / (4*np.pi)

class Polyhedron:
    def __init__(self, vertices, winding, J, M, scale_factor=[1,1,1], origin=[0,0,0]):
        self._N = len(winding)

        # Hang on to these for the sake of future proofing if we want to make copies
        self._vertices = vertices
        self._vertices = np.multiply(self._vertices, scale_factor)
        self._vertices = np.add(self._vertices, origin)

        self._winding = winding
        self._J = J
        self._M = M
        self._origin = origin

        self._faces = []

        for wind in winding:
            face_vertices = []

            for index in wind:
                face_vertices.append(self._vertices[index])

            self._faces.append(Face(face_vertices,J,M))

    @property
    def volume(self):
        # Calculate the volume of arbitrary polyhedron
        volume = 0

        return volume

    @property
    def origin(self):
        return self._origin

    @property
    def M(self):
        return self._M

    def unit_copy(self, axis):
        if axis == 0:
            newM = [1,0,0]
        elif axis == 1:
            newM = [0,1,0]
        elif axis == 2:
            newM = [0,0,1]
        else:
            newM = [0,0,0]

        return Polyhedron(self._vertices, self._winding, self._J, newM)

    def copy(self, newM):
        return Polyhedron(self._vertices, self._winding, self._J, newM)

    def field(self, r, field = "B"):
        acc = [0, 0, 0]

        for face in self._faces:
            if field == "B":
                acc = np.add(acc, face.b_flux(r))
            elif field == "H":
                acc = np.add(acc, face.h_flux(r))
            elif field == "M":
                acc = np.add(acc, face.m_flux(r))
            else:
                print('oops')

        return acc

    def constitutive(self, field):
        # Default constitutive law is just some
        # susceptibility multiplying the field
        # 
        # this is easily reimplemented for child objs

        susc = 10
        return np.multiply(susc, field)

    def integrate(self, func, order):
        # Default integration behavior is to just return the function value
        # at the center of the polyhedron. Should be reimplemented to be Gauss-Legendre
        # quadrature for the geometry of a child Polyhedron for non generics

        return func(self.origin)
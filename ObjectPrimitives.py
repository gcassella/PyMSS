import sys
import numpy as np
import matplotlib.pyplot as plt

def epsilon_metric(r1: list, r2: list):
    euclidean_metric2 = (r2[0] - r1[0])**2 + (r2[1] - r1[1])**2 + (r2[2] - r1[2])**2

    return np.sqrt(euclidean_metric2 + 1e-12)

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
    def __init__(self, vertices, winding, J, M):
        self._N = len(winding)

        self.J = J
        self.M = M

        self.faces = []

        for wind in winding:
            face_vertices = []
            for index in wind:
                face_vertices.append(vertices[index])

            self.faces.append(Face(face_vertices,J,M))

    def field(self, r, field = "B"):
        acc = [0, 0, 0]

        for face in self.faces:
            if field == "B":
                acc = np.add(acc, face.b_flux(r))
            elif field == "H":
                acc = np.add(acc, face.h_flux(r))
            elif field == "M":
                acc = np.add(acc, face.m_flux(r))
            else:
                print('oops')

        return acc


if __name__ == '__main__':
    v = [[1/3*np.sqrt(3),0,0],[-1/6*np.sqrt(3),1/2,0],[-1/6*np.sqrt(3),-1/2,0],[0,0,1/3*np.sqrt(6)]]

    poly = Polyhedron(v, [[0,2,1],[0,3,2],[0,1,3],[1,2,3]],[0,0,10],[10,10,10])

    """
    face1 = Face([v[0],v[2],v[1]],[0,0,0],[10,10,10])
    face2 = Face([v[0],v[3],v[2]],[0,0,0],[10,10,10])
    face3 = Face([v[0],v[1],v[3]],[0,0,0],[10,10,10])
    face4 = Face([v[1],v[2],v[3]],[0,0,0],[10,10,10])
    """

    bfield = []
    hfield = []
    mfield = []

    Z = np.linspace(0,1,num=1000)

    for z in Z:
        
        r = [0,0, z] 
        bfield.append(poly.field(r, "B"))
        hfield.append(poly.field(r, "H"))
        mfield.append(poly.field(r, "M"))
    

    bfield = np.array(bfield)
    hfield = np.array(hfield)
    mfield = np.array(mfield)

    plt.plot(Z,np.linalg.norm(bfield, axis=1), 'k-')
    plt.plot(Z,np.linalg.norm(hfield, axis=1), 'b-')
    plt.plot(Z,np.linalg.norm(mfield, axis=1), 'r-')
    plt.show()


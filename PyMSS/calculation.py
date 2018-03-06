import numpy as np
import PyMSS.polyhedra

def construct_average_interaction_matrix(sourcelist, order=5):
    N = len(sourcelist)

    interaction_matrix = np.zeros((3*N,3*N))

    k=0
    h=0

    for source_k in sourcelist:
        h=0
        for source_h in sourcelist:
            sub_matrix = np.zeros((3,3))

            unit_source_i = source_h.unit_copy(0)
            unit_source_j = source_h.unit_copy(1)
            unit_source_k = source_h.unit_copy(2)

            sub_matrix[:,0] = source_k.integrate(lambda r: unit_source_i.field(r), order) / source_k.volume
            sub_matrix[:,1] = source_k.integrate(lambda r: unit_source_j.field(r), order) / source_k.volume
            sub_matrix[:,2] = source_k.integrate(lambda r: unit_source_k.field(r), order) / source_k.volume

            interaction_matrix[k*3:(k+1)*3,h*3:(h+1)*3] = sub_matrix

            h+=1
        k+=1

    return interaction_matrix

def construct_average_field_tensor(sourcelist, order=5):
    N = len(sourcelist)

    field_matrix = np.zeros((N,3))

    k=0

    for source_k in sourcelist:
        acc = [0,0,0]

        for source_h in sourcelist:
            average_field = source_k.integrate(lambda r: source_h.field(r, field="H"), order) / source_k.volume

            acc = np.add(acc, average_field)
        
        acc = np.divide(acc, N)
        field_matrix[k,:] = acc

        k+=1

    return field_matrix

def construct_point_interaction_matrix(sourcelist):
    N = len(sourcelist)

    interaction_matrix = np.zeros((3*N,3*N))

    k=0
    h=0

    for source_k in sourcelist:
        h=0
        for source_h in sourcelist:
            sub_matrix = np.zeros((3,3))

            unit_source_i = source_h.unit_copy(0)
            unit_source_j = source_h.unit_copy(1)
            unit_source_k = source_h.unit_copy(2)

            sub_matrix[:,0] = unit_source_i.field(source_k.origin)
            sub_matrix[:,1] = unit_source_j.field(source_k.origin)
            sub_matrix[:,2] = unit_source_k.field(source_k.origin)

            interaction_matrix[k*3:(k+1)*3,h*3:(h+1)*3] = sub_matrix

            h+=1
        k+=1

    return interaction_matrix

def construct_point_field_tensor(sourcelist):
    N = len(sourcelist)

    field_matrix = np.zeros((N,3))

    k=0

    for source_k in sourcelist:
        acc = [0,0,0]

        for source_h in sourcelist:
            acc = np.add(acc, source_h.field(source_k.origin, field="H"))
        
        acc = np.divide(acc, N)
        field_matrix[k,:] = acc

        k+=1

    return field_matrix


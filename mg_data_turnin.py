''' 
    Data generation for Neural Network of Au=b Multigrid Solver
    Randomized u and compute solution vectors b from consistent Laplacian A
'''

import numpy 
from math import *

def Laplacian(n, stencil=[-1, 2, -1], periodic=True):
    A = stencil[1] * numpy.eye(n) + stencil[2] * numpy.eye(n, k=1) + stencil[0] * numpy.eye(n, k=-1)
    if periodic:
        A[0,-1] = stencil[0]
        A[-1,0] = stencil[2]
    return A

def gen_data(gridsize, n): #input is number of training/testing samples desired, matrix size of A is gridsize x gridsize
    dataset = []
    solset = []
    A = Laplacian(gridsize)
    for _ in range(n):
       u = numpy.random.rand(gridsize)
        u = u - numpy.mean(u) 

        b = A @ u

        dataset.append(b)
        solset.append(u)

    return numpy.array(dataset), numpy.array(solset) ]

#print (gen_data(6,10)) #shape is (n, gridsize, 1)

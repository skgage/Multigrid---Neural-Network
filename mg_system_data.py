''' 
    Test problems for Neural Network of Ax=b or Linear Multigrid Solver
    Can either randomize A or x and compute solution vectors b

    12/2 Possibly it makes sense to leave A = [[2,-1],[-1,2]] so that it learns it just
    needs to inverse and multiply by b essentially to compute u
    Leaving A this way is the Laplacian
    If A were to stay the same each time, this certainly presents a difficult since each input is requiring a 
    different output

    Ex: A = [[1 2],[3 4]] x = [[1],[1]] then A*x = [[3],[7]] = b

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
        #k = random.randint(0,1)
       # k = 0 #for now 12/2
        #if (k == 0): #let A be [[2,-1,0],[-1,2,-1],...,[0,-1,2]]
            #will need an efficient way to do this fast
            #for now will let gridsize be 2x2 for getting started
            #u = numpy.random.random((gridsize,1))
           # u = numpy.random.uniform(low=-10, high=10, size=(gridsize,1)) #does range matter? possibily the larger the range, the easier for network?
        u = numpy.random.rand(gridsize)
        u = u - numpy.mean(u)   
       # u_test = [1]*gridsize #FOR TEST OF AI=A FEB12 707PM
        #u = u_test      
        #print ('u = ', u)
        b = A @ u
        #print ('b = ', b)
        dataset.append(b)
        solset.append(u)
    dataset = numpy.reshape(numpy.array(dataset),[n,1,gridsize,1])
    solset = numpy.reshape(numpy.array(solset),[n,1,gridsize,1])
        # if (k == 1): #randomly generate tridiagonal matrix? maybe should leave as 2,-1, so on and just random x vector
        #     continue
    return dataset, solset #returns input matrix of [A b] and solution array u]

def AI_data(gridsize,n):
     A = Laplacian(gridsize)
     #print (A)
     I = numpy.identity(gridsize)
     if (n == 1):
        r = numpy.random.randint(0,gridsize)
       #print ('r = ', r)
        A = A[:,r]
        I = I[:,r]
     else:
        A = A[:,:n]
        I = I[:,:n]
    #print ('Aprime = ', A)
     A = numpy.reshape(numpy.array(A),[n,1,gridsize,1])
     I = numpy.reshape(numpy.array(I),[n,1,gridsize,1])
     return A, I

#print (gen_data(6,6)) #shape is (n, gridsize, 1)
#print (AI_data(6,1))

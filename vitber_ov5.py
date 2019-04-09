from scipy import sparse as spa
import numpy as np
from numpy import linalg as LA
from scipy import linalg as scla

def get_A(n):
	A = np.zeros([n,n])
	A[0,0] = 2
	A[1,0] = -1
	A[0,1] = -1
	A[n,n] = 2
	for i in range(n-2):
		A[i+1,i+1] = 2
		A[i, i+1] = -1
		A[i+1, i] = -1
	return A

#def sparse_get_A(n):
#    diag = [sp.full(n, 2.0), sp.full(n-1, -1.0), sp.full(n-1, -1.0)]
#    A = spar.diags(diag, offsets=(0,1,-1))
#    return A

 def SOR(A, b, n, om, f, tol, maxiter):
 	u0 = np.zeros(n)
 	L, D = np.diag(A.diagonal), np.tril(A, k=-1)
 	U = A - L - D
 	M = D + om*L
 	N = A - M
 	u = u0
 	omLDinv = LA.inverse(M)
 	for i in range(maxiter):
 		um1 = u
 		u = omLDinv @ ((1-om)*2*u-om*2*u+om*B)
 		if(LA.norm(u-um1)<tol):
 			break
 	return u


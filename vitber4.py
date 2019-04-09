import numpy as np
from numpy import linalg as LA
from scipy import linalg as scla

def forward_subs(LU,P,b):
    n, m = LU.shape
    Pb = b[P]
    c = np.zeros(n)
    c[0] = Pb[0]
    for k in range(1,n):
        c[k] = Pb[k] - LU[P[k],0:k] @ c[0:k]
        
    return c

def backward_subs(LU,P,c):
    n,m = LU.shape
    x = np.zeros(n)
    x[n-1] = c[n-1]/LU[P[n-1],n-1]
    for k in range(n-1,0,-1):
        x[k-1] = (c[k-1]-LU[P[k-1],k:] @ x[k:])/LU[P[k-1],k-1]
        
    return x

def mylu(A):
	n = len(A)

	P = np.arange(n)
	for k in range(n):
		pivot = k + np.argmax(abs(A[k:,k]))
		
		P[[k, pivot]] = P[[pivot, k]]
		
		A[P[k:],k] /= A[P[k],k]

		col = np.pad(A[P][k+1:,k],(k+1,0),'constant')
		row = A[P][k,:]
		destroyer = np.outer(col, row)
		A[P] -= destroyer
	return P

def getAb():
    A=np.array([[0.3050, 0.5399, 0.9831, 0.4039, 0.1962],
                [0.2563, -0.1986, 0.7903, 0.6807, 0.5544],
                [0.7746, 0.6253, -0.1458, 0.1704,  0.5167],
                [0.4406, 0.9256, 0.4361, -0.2254, 0.7784],
                [0.4568, 0.2108, 0.6006, 0.3677, -0.8922]])
    b=np.array([0.9876,-1.231,0.0987,-0.5544,0.7712])
    return A,b
A, b = getAb()
p_vec = mylu(A)

c = forward_subs(LU, p_vec, b)
print("Forward:")

print(c)

print("Backward:")
x = backward_subs(LU,p_vec,c)
print(x)

print("Ax")
print(A @ x)
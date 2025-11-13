import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

students_index = [1, 9, 8, 0, 3, 5]
N = 1200 + 10*students_index[-2] + students_index[-1]

def equations(index, N):
    e = int(index[3])
    f = int(index[2])
    a1 = 3
    a2 = a3 = -1
    b = np.array([math.sin(n * (f + 1)) for n in range(1, N + 1)])
    A = np.zeros((N, N))
    
    for i in range(N):
        A[i][i] = a1

        if i - 1 >= 0:
            A[i][i - 1] = a2
        if i + 1 < N:
            A[i][i + 1] = a2
        if i - 2 >= 0:
            A[i][i - 2] = a3
        if i + 2 < N:
            A[i][i + 2] = a3
    
    return A, b

def LU_factorization(A):
    n = len(A)
    U = copy.copy(A)
    L = np.eye(n)

    for i in range(2, n+1):
        for j in range(1, i):
            L[i-1, j-1] = U[i-1, j-1] / U[j-1, j-1]
            U[i-1, :] = U[i-1, :] - L[i-1, j-1] * U[j-1, :]
    
    return L, U

def LU_solve(L, U, b):
    L, U = LU_factorization(A)

    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)

    return x

A, b = equations(students_index, N)
start = time.time()
L, U = LU_factorization(A)
x = LU_solve(L, U, b)
end = time.time()

lu_time = end - start

residuum = A @ x - b
norm_r = np.linalg.norm(residuum)

print(f"Norma residuum dla metody faktoryzacji LU: {norm_r}")
print(f"Czas wykonania metody LU: {lu_time:.6f} sekund")
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

def equations(index, N):
    e = int(index[3])
    f = int(index[2])
    a1 = 5 + e
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

def jacobi_method(A, b, tol = 1e-9, max_iterations = 100):
    N = len(b)
    x = np.zeros(N)

    L = np.tril(A, -1)
    U = np.triu(A, 1)
    D = np.diag(np.diag(A))

    M = np.linalg.solve(D, -(L + U))
    w = np.linalg.solve(D, b)

    residuum = []

    for i in range(max_iterations):
        x_new = M @ x + w

        r = A @ x_new - b
        norm_r = np.linalg.norm(r)
        residuum.append(norm_r)

        if norm_r < tol:
            return x_new, i + 1, residuum
        
        x = x_new
    
    return x, max_iterations, residuum

def gauss_seidel_method(A, b, tol = 1e-9, max_iterations = 100):
    N = len(A)
    x = np.zeros(N)

    L = np.tril(A, -1)
    U = np.triu(A, 1)
    D = np.diag(np.diag(A))

    T = D + L
    residuum = []

    for i in range(max_iterations):
        rhs = b - U @ x
        x_new = np.linalg.solve(T, rhs)

        r = A @ x_new - b
        norm_r = np.linalg.norm(r)
        residuum.append(norm_r)

        if norm_r < tol:
            return x_new, i + 1, residuum

        x = x_new

    return x, max_iterations, residuum

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

students_index = [1, 9, 8, 0, 3, 5]
N_numbers = [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 2750, 3000, 3250]
times_jacobi = []
times_gauss = []
times_lu = []

for N in N_numbers:
    print(f"Obliczenia dla N = {N}")
    A, b = equations(students_index, N)

    start = time.time()
    solution_jacobi, _, _ = jacobi_method(A, b)
    end = time.time()
    jacobi_time = end - start
    print(f"Czas metody jacobiego: {jacobi_time}")
    times_jacobi.append(jacobi_time)

    start = time.time()
    solution_gauss, _, _ = gauss_seidel_method(A, b)
    end = time.time()
    gauss_time = end - start
    print(f"Czas metody gaussa: {gauss_time}")
    times_gauss.append(gauss_time)

    start = time.time()
    L, U = LU_factorization(A)
    x = LU_solve(L, U, b)
    end = time.time()
    lu_time = end - start
    print(f"Czas metody LU: {lu_time}")
    times_lu.append(lu_time)

plt.plot(N_numbers, times_jacobi, label="Jacobi")
plt.plot(N_numbers, times_gauss, label="Gauss-Seidel")
plt.plot(N_numbers, times_lu, label="LU")
plt.xlabel("Liczba niewiadomych (N)")
plt.ylabel("Czas wykonania (s)")
plt.title("Czas działania metod – skala liniowa")
plt.legend()
plt.grid(True)
plt.savefig("Wykres_E_lin.png")
plt.show()

plt.plot(N_numbers, times_jacobi, label="Jacobi")
plt.plot(N_numbers, times_gauss, label="Gauss-Seidel")
plt.plot(N_numbers, times_lu, label="LU")
plt.xlabel("Liczba niewiadomych (N)")
plt.ylabel("Czas wykonania (s)")
plt.yscale("log")
plt.title("Czas działania metod – skala logarytmiczna")
plt.legend()
plt.grid(True)
plt.savefig("Wykres_E_log.png")
plt.show()
import math
import numpy as np
import matplotlib.pyplot as plt

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

A, b = equations(students_index, N)
x_jacobi, iter_jacobi, res_jacobi = jacobi_method(A, b)
print(f"\nRozwiazanie metoda jacobiego: {x_jacobi}")
print(f"Metoda Jacobiego zakończona po {iter_jacobi} iteracjach.")
x_gauss, iter_gauss, res_gauss = gauss_seidel_method(A, b)
print(f"\nRozwiazanie metoda gaussa: {x_gauss}")
print(f"Metoda Gaussa zakończona po {iter_gauss} iteracjach.")

plt.plot(range(len(res_jacobi)), res_jacobi, label="Jacobi method")
plt.plot(range(len(res_gauss)), res_gauss, label="Gauss-Seidel method")
plt.yscale('log')
plt.xlabel("Iteracje")
plt.ylabel("Norma residuum")
plt.title("Zmiana normy residuum w kolejnych iteracjach")
plt.legend()
plt.grid()
plt.savefig('Wykres_C.png')
plt.show()
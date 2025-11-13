import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy

kanion_kolorado = pd.read_csv("2018_paths/WielkiKanionKolorado.csv", sep=',', skiprows=1, header=None)
x_kanion = kanion_kolorado[0].values.tolist()
y_kanion = kanion_kolorado[1].values.tolist()
data_kanion = (x_kanion, y_kanion)
#print(data_kanion)

chelm = pd.read_csv("2018_paths/chelm.txt", sep=" ", header=None)
x_chelm = chelm[0].values.tolist()
y_chelm = chelm[1].values.tolist()
data_chelm = (x_chelm, y_chelm)
#print(data_chelm)

def LU_factorization(A):
    n = len(A)
    U = copy.deepcopy(A)
    L = np.eye(n)

    for i in range(2, n+1):
        for j in range(1, i):
            L[i-1, j-1] = U[i-1, j-1] / U[j-1, j-1]
            U[i-1, :] = U[i-1, :] - L[i-1, j-1] * U[j-1, :]
    
    return L, U

def LU_solve(A, b):
    L, U = LU_factorization(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x

def profile_wysokościowe_wykres(data, text):
    plt.figure(figsize=(12, 6))
    plt.plot(data[0], data[1])
    plt.xlabel('Dystans (m)')
    plt.ylabel('Wzniesienie (m)')
    plt.title(f"Profil wysokościowy: {text}")
    filename = f"wykresy/{text}_podstawowe_dane_profil_wysokościowy.png"
    plt.savefig(filename, dpi=300)
    plt.close()

def skaluj_lista(x_list, a, b):
    return [(x - a) / (b - a) for x in x_list]

def transformuj_lista(x_scaled_list, a, b):
    return [a + x * (b - a) for x in x_scaled_list]

def wybierz_wezly(x_list, y_list, n, metoda):
    if metoda == "chebyshev":
        a = x_list[0]
        b = x_list[-1]
        x_chebyshev = [(a + b)/2 + (b - a)/2 * np.cos(np.pi * (2 * i + 1) / (2 * n)) for i in range(n)]
        y_chebyshev = [np.interp(xc, x_list, y_list) for xc in x_chebyshev]  # interpolacja liniowa danych
        return x_chebyshev, y_chebyshev
    elif metoda == "równomiernie":
        indeksy = [int(i) for i in np.linspace(0, len(x_list) - 1, n)]
        x_wezly = [x_list[i] for i in indeksy]
        y_wezly = [y_list[i] for i in indeksy]
        return x_wezly, y_wezly

def interpolacja_lagrange_punkt(x_nodes, y_nodes, x):
    n = len(x_nodes)
    total = 0.0
    for i in range(n):
        xi, yi = x_nodes[i], y_nodes[i]
        
        term = yi
        for j in range(n):
            if j != i:
                xj = x_nodes[j]
                term *= (x - xj) / (xi - xj)
        total += term
    return total

def interpolacja_lagrange(x_nodes, y_nodes, x_values):
    return [interpolacja_lagrange_punkt(x_nodes, y_nodes, x) for x in x_values]

def interpolacja_lagrange_dla_danych(dane, n_wezlow):
    if dane == data_kanion:
        x_data, y_data = data_kanion
        tytul = "Wielki Kanion Kolorado"
        profile_wysokościowe_wykres(data_kanion, tytul)

    elif dane == data_chelm:
        x_data, y_data = data_chelm
        tytul = "Chełm"
        profile_wysokościowe_wykres(data_chelm, tytul)
    else:
        raise ValueError("Nieznane dane")

    x_nodes, y_nodes = wybierz_wezly(x_data, y_data, n_wezlow, "równomiernie")
    a, b = min(x_nodes), max(x_nodes)
    x_nodes_scaled = skaluj_lista(x_nodes, a, b)
    x_dense_scaled = np.linspace(0, 1, 1000)
    x_dense = transformuj_lista(x_dense_scaled, a, b)
    y_interpolowane = interpolacja_lagrange(x_nodes_scaled, y_nodes, x_dense_scaled)

    plt.figure(figsize=(12, 6))
    plt.plot(x_data, y_data, label='Dane oryginalne')
    plt.scatter(x_nodes, y_nodes, color='red', label='Węzły interpolacji')
    plt.plot(x_dense, y_interpolowane, color='green', label=f'Interpolacja Lagrangea ({n_wezlow} punktów)')
    plt.xlabel('Dystans (m)')
    plt.ylabel('Wzniesienie (m)')
    plt.title(f'{tytul} – Interpolacja Lagrangea ({n_wezlow} węzłów)')
    plt.legend()
    plt.yscale('symlog', linthresh=100)
    filename = f"wykresy/{tytul}_{n_wezlow}_wezlow_lagrange.png"
    plt.savefig(filename, dpi=300)
    plt.show()

def interpolacja_spline_punkt(x_nodes, y_nodes, M, x):
    n = len(x_nodes)
    
    for i in range(n - 1):
        if x_nodes[i] <= x <= x_nodes[i + 1]:
            h = x_nodes[i + 1] - x_nodes[i]
            A = (x_nodes[i + 1] - x) / h
            B = (x - x_nodes[i]) / h
            S = (A * y_nodes[i] + B * y_nodes[i + 1] +
                 ((A**3 - A) * M[i] + (B**3 - B) * M[i + 1]) * h**2 / 6)
            return S

def interpolacja_spline(x_nodes, y_nodes, x_values):
    n = len(x_nodes)
    h = [x_nodes[i+1] - x_nodes[i] for i in range(n - 1)]
    A = np.zeros((n, n))
    b = np.zeros(n)

    A[0, 0] = 1
    A[-1, -1] = 1

    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 6 * ((y_nodes[i + 1] - y_nodes[i]) / h[i] - (y_nodes[i] - y_nodes[i - 1]) / h[i - 1])

    M = LU_solve(A, b)

    return [interpolacja_spline_punkt(x_nodes, y_nodes, M, x) for x in x_values]

def interpolacja_spline_dla_danych(dane, n_wezlow):
    if dane == data_kanion:
        x_data, y_data = data_kanion
        tytul = "Wielki Kanion Kolorado"
    elif dane == data_chelm:
        x_data, y_data = data_chelm
        tytul = "Chełm"
    else:
        raise ValueError("Nieznane dane")
    
    x_nodes, y_nodes = wybierz_wezly(x_data, y_data, n_wezlow, "równomiernie")
    x_dense = np.linspace(min(x_nodes), max(x_nodes), 1000)
    y_interpolowane = interpolacja_spline(x_nodes, y_nodes, x_dense)

    plt.figure(figsize=(12, 6))
    plt.plot(x_data, y_data, label='Dane oryginalne')
    plt.scatter(x_nodes, y_nodes, color='red', label='Węzły interpolacji')
    plt.plot(x_dense, y_interpolowane, color='blue', label=f'Splajny trzeciego stopnia ({n_wezlow} punktów)')
    plt.xlabel('Dystans (m)')
    plt.ylabel('Wzniesienie (m)')
    plt.title(f'{tytul} – Interpolacja splajnami ({n_wezlow} węzłów)')
    plt.legend()
    plt.yscale('symlog', linthresh=100)
    filename = f"wykresy/{tytul}_{n_wezlow}_wezlow_spline.png"
    plt.savefig(filename, dpi=300)
    plt.show()

def analiza_dodatkowa_wezly_chebysheva(dane, n_wezlow):
    if dane == data_kanion:
        x_data, y_data = data_kanion
        tytul = "Wielki Kanion Kolorado"
    elif dane == data_chelm:
        x_data, y_data = data_chelm
        tytul = "Chełm"
    else:
        raise ValueError("Nieznane dane")

    x_nodes, y_nodes = wybierz_wezly(x_data, y_data, n_wezlow, "chebyshev")
    a, b = min(x_nodes), max(x_nodes)
    x_nodes_scaled = skaluj_lista(x_nodes, a, b)
    x_dense_scaled = np.linspace(0, 1, 1000)
    x_dense = transformuj_lista(x_dense_scaled, a, b)
    y_interpolowane = interpolacja_lagrange(x_nodes_scaled, y_nodes, x_dense_scaled)

    plt.figure(figsize=(12, 6))
    plt.plot(x_data, y_data, label='Dane oryginalne')
    plt.scatter(x_nodes, y_nodes, color='red', label='Węzły interpolacji')
    plt.plot(x_dense, y_interpolowane, color='magenta', label=f'Interpolacja Lagrangea z węzłami Chebysheva ({n_wezlow} punktów)')
    plt.xlabel('Dystans (m)')
    plt.ylabel('Wzniesienie (m)')
    plt.title(f'{tytul} – Interpolacja Lagrangea z węzłami Chebysheva ({n_wezlow} węzłów)')
    plt.legend()
    plt.yscale('symlog', linthresh=100)
    filename = f"wykresy/{tytul}_{n_wezlow}_chebyshev_lagrange.png"
    plt.savefig(filename, dpi=300)
    plt.show()

for n_wezlow in [8, 16, 32, 64, 128]:
    interpolacja_lagrange_dla_danych(data_kanion, n_wezlow)
    interpolacja_lagrange_dla_danych(data_chelm, n_wezlow)
    interpolacja_spline_dla_danych(data_kanion, n_wezlow)
    interpolacja_spline_dla_danych(data_chelm, n_wezlow)
    analiza_dodatkowa_wezly_chebysheva(data_kanion, n_wezlow)
    analiza_dodatkowa_wezly_chebysheva(data_chelm, n_wezlow)
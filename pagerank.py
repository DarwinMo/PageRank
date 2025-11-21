# @author : Mohamed Abdessalame

import numpy as np

def google_matrix(A, a):
    n = A.shape[0]
    S = np.ones((n, n)) / n
    return (1 - a) * A + a * S



def vecteur_de_score(A, a):
    G = google_matrix(A, a)
    vp, vecp = np.linalg.eig(G)
    indice_rho = np.argmax(vp)
    x = vecp[:, indice_rho]
    x = x / np.sum(x)
    return np.real(x)



def score_dynamique(A, a, iter_max=300, eps=1e-3):
    n = len(A)
    G = google_matrix(A, a)
    z = np.ones((n, 1)) / n
    for i in range(iter_max + 1):
        z1 = np.dot(G,z)
        if np.all(np.abs(z1 - z) < eps):
            break
        z = z1
    z = z / np.sum(z)
    return z




if __name__ == "__main__":
    A1 = np.array([
        [0, 0, 1, 1 / 2],
        [1 / 3, 0, 0, 0],
        [1 / 3, 1 / 2, 0, 1 / 2],
        [1 / 3, 1 / 2, 0, 0]
    ], dtype=float)

    A2 = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0]
    ], dtype=float)
    print("A1:\n")
    print("Score  A1 :\n", vecteur_de_score(A1, 0.15), "\n")
    print("Score (par les puissance) A1 :\n", score_dynamique(A1, 0.15, 300), "\n")
    print("A2: \n")
    print("Score  A2 :\n", vecteur_de_score(A2, 0.15), "\n")
    print("Score (par les puissance) A2 :\n", score_dynamique(A2, 0.15, 300), "\n")

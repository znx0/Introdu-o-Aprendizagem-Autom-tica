import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
os.makedirs("figures", exist_ok=True)

# NAO MODIFICAR A SEED
np.random.seed(555)


# -------------------------------------------------------------------
# Inicializacao dos centroides
# -------------------------------------------------------------------

def kminit(X, K, flag):
    """
    Inicializa K centroides usando inicializacao aleatoria ou k-means++.

    Input:
        X: ndarray com dimensoes (N, d)
        K: numero de clusters
        flag: "random" ou "km++"

    Output:
        C: ndarray com dimensoes (K, d)
    """

    N = X.shape[0]

    if not 1 <= K <= N:
        raise ValueError("K tem de estar entre 1 e N.")

    if flag not in ("random", "km++"):
        raise ValueError("flag tem de ser 'random' ou 'km++'.")

    # Inicializacao do primeiro centroide
    primeiro_idx = np.random.randint(N)
    indices = [primeiro_idx]
    C = [X[primeiro_idx].copy()]

    if flag == "random":
        # Escolher os restantes centroides sem repetir indices
        restantes = np.delete(np.arange(N), primeiro_idx)
        escolhidos = np.random.choice(
            restantes, size=K - 1, replace=False
        )
        for idx in escolhidos:
            C.append(X[idx].copy())

    elif flag == "km++":
        for _ in range(K - 1):
            C_arr = np.array(C)

            # Distancia quadratica de cada ponto ao centroide mais proximo
            dists = np.sum(
                (X[:, np.newaxis, :] - C_arr[np.newaxis, :, :]) ** 2,
                axis=2
            )
            min_dists = np.min(dists, axis=1)

            # Evitar escolher novamente um indice ja selecionado
            min_dists[indices] = 0.0
            soma = np.sum(min_dists)

            if soma > 0:
                probs = min_dists / soma
                idx = np.random.choice(N, p=probs)
            else:
                # Se todas as distancias forem zero, escolher
                # um indice ainda nao selecionado
                restantes = np.setdiff1d(
                    np.arange(N), indices
                )
                idx = np.random.choice(restantes)

            indices.append(idx)
            C.append(X[idx].copy())

    return np.array(C)


# -------------------------------------------------------------------
# Calculo da funcao de custo
# -------------------------------------------------------------------

def kmeans_custo(X, C, s):
    """
    Calcula as distancias quadradas aos centroides atribuidos
    e o custo total J.

    Input:
        X: ndarray (N, d)
        C: ndarray (K, d)
        s: ndarray (N,) com os indices dos clusters

    Output:
        dists_sq: distancias quadradas (N,)
        J: soma das distancias quadradas
    """

    centros_atribuidos = C[s]

    dists_sq = np.sum(
        (X - centros_atribuidos) ** 2,
        axis=1
    )

    J = np.sum(dists_sq)

    return dists_sq, J


# -------------------------------------------------------------------
# Algoritmo k-means
# -------------------------------------------------------------------

def kmeans(X, K, flag, max_iter=300, tol=1e-6):
    """
    Aplica o algoritmo k-means.

    Output:
        C: centroides finais (K, d)
        z: atribuicoes finais (N,)
    """

    N, d = X.shape

    # Inicializacao
    C = kminit(X, K, flag)

    J_anterior = None

    for _ in range(max_iter):

        # Passo 1: atribuicao dos pontos aos centroides mais proximos
        dists = np.sum(
            (X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2,
            axis=2
        )
        z = np.argmin(dists, axis=1)

        # Passo 2: atualizacao dos centroides
        C_novo = np.zeros((K, d))

        for k in range(K):
            pontos_k = X[z == k]

            if len(pontos_k) > 0:
                C_novo[k] = pontos_k.mean(axis=0)
            else:
                # Reinicializar clusters vazios num ponto de X
                C_novo[k] = X[np.random.randint(N)]

        # Atualizar centroides
        C = C_novo

        # Recalcular atribuicoes para os centroides atualizados
        dists = np.sum(
            (X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2,
            axis=2
        )
        z = np.argmin(dists, axis=1)

        # Calcular custo consistente com os centroides e atribuicoes
        _, J_atual = kmeans_custo(X, C, z)

        # Criterio de paragem
        if J_anterior is not None:
            if abs(J_anterior - J_atual) < tol:
                break

        J_anterior = J_atual

    # Atribuicao final
    dists = np.sum(
        (X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2,
        axis=2
    )
    z = np.argmin(dists, axis=1)

    return C, z


# -------------------------------------------------------------------
# Execucao principal
# -------------------------------------------------------------------

if __name__ == "__main__":

    filename = (
        sys.argv[1] if len(sys.argv) > 1 else "Xtrain.pkl"
    )

    # Ler os dados
    df_train = pd.read_pickle(filename)

    # Concatenar todas as poses de todas as sequencias
    X = np.concatenate(
        df_train["Skeleton_Sequence"].to_numpy()
    )

    print("Shape de X:", X.shape)

    # Testar diferentes valores de K
    K_valores = range(1, 11)
    custos = []

    for K in K_valores:
        C, z = kmeans(X, K, flag="random")
        _, J = kmeans_custo(X, C, z)

        custos.append(J)

        print(f"K={K} -> custo J={J:.2f}")

    # Grafico do custo em funcao de K

    plt.plot(list(K_valores), custos, marker="o")
    plt.xlabel("Numero de centroides (K)")
    plt.ylabel("Custo J")
    plt.title(
        "Custo do k-means em funcao de K "
        "(inicializacao random)"
    )
    plt.xticks(list(K_valores))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/PartI2a.jpg", dpi=300)
    plt.show()

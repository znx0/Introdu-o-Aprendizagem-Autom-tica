import numpy as np
import panda as pd
import sys

# NAO MODIFICAR A SEED
np.random.seed(555)

def kminit(X, K, flag):
    """
    Inicialização - pode ser aleatória ou recorrendo ao k-means++

    Input:
    X: um numpy ndarray com dimensões (N,d), em que cada linha corresponde
       a uma amostra
    K: um inteiro que corresponde ao número de clusters a estimar
    flag: indicador sobre o tipo de inicialização ("random" ou "km++")

    Output:
    C: um numpy ndarray com dimensões (K,d), onde cada linha é a inicialização
       de um dos centróides
    """
  
    # A inicialização do primeiro centróide é fixa
    N = X.shape[0]
    C = []
    C.append(X[np.random.randint(N)])

    if flag == 'random':
        # escolher K-1 centróides, aleatoriamente, a partir de X
        cluster = np.random.choice(N, size=K - 1, replace=False)
        for idx in cluster:
            C.append(X[idx])

    elif flag == 'km++':
        for _ in range(K - 1):
            C_arr = np.array(C)
            # distância² de cada ponto ao centróide mais próximo já escolhido
            dists = np.sum((X[:, np.newaxis, :] - C_arr[np.newaxis, :, :]) ** 2, axis=2)
            min_dists = np.min(dists, axis=1)
            probs = min_dists / np.sum(min_dists)
            idx = np.random.choice(N, p=probs)
            C.append(X[idx])

    return np.array(C)  # matriz de centróides/clusters

#------------------------------------------------------------------------------------#

def kmeans_custo(X, C, s):
    """
    Calcula a função de custo do k-means

    Input:
    X: um numpy ndarray com dimensões (N,d), em que cada linha corresponde
       a uma amostra
    C: um numpy ndarray com dimensões (K,d), onde cada linha é a inicialização
       de um dos centróides
    s: um numpy ndarray de dimensão (N,) onde cada entrada é um inteiro que
       representa o índice do cluster para o ponto xi

    Returns squared Euclidean distances from each point to the center of
    its assigned cluster (dists_sq), e o custo total J (soma dessas distâncias)
    """
  
    centros_atribuidos = C[s]

    # distância euclidiana ao quadrado entre cada ponto e o seu centróide
    dists_sq = np.sum((X - centros_atribuidos) ** 2, axis=1)
    # soma dos custos
    J = np.sum(dists_sq)

    return dists_sq, J
    
#------------------------------------------------------------------------------------#

def kmeans(X, K, flag, max_iter=300, tol=1e-6):
    """
    Aplica o algoritmo k-means à matriz de dados X.

    X: um numpy ndarray com dimensões (N,d), em que cada linha corresponde
       a uma amostra
    K: um inteiro que corresponde ao número de clusters a estimar
    flag: indicador sobre o tipo de inicialização ("random" ou "km++")

    Devolve um tuplo contendo (C, z):
    C: um numpy ndarray com dimensões (K,d), onde cada linha é o centróide final
    z: um numpy ndarray de dimensão (N,) onde cada entrada é um inteiro
       que representa o índice do cluster para o ponto x(i)
    """
  
    N, d = X.shape

    # Inicialização dos centróides
    C = kminit(X, K, flag)

    J_anterior = None

    for _ in range(max_iter):

        # Passo de atribuição: associa cada ponto ao centróide mais próximo
        dists = np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2)
        z = np.argmin(dists, axis=1)

        # Passo de atualização: recalcula cada centróide como a média dos pontos do cluster
        C_novo = np.zeros((K, d))
        for k in range(K):
            pontos_k = X[z == k]
            if len(pontos_k) > 0:
                C_novo[k] = pontos_k.mean(axis=0)
            else:
                # cluster vazio: reinicializa com um ponto aleatório
                C_novo[k] = X[np.random.randint(N)]

        C = C_novo

        # Critério de paragem: variação do custo J desprezável
        _, J_atual = kmeans_custo(X, C, z)
        if J_anterior is not None and abs(J_anterior - J_atual) < tol:
            break
        J_anterior = J_atual

    # Atribuição final com os centróides convergidos
    dists = np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2)
    z = np.argmin(dists, axis=1)

    return C, z

#------------------------------------------------------------------------------------#

# Garante que o bloco só corre quando corre o ficheiro diretamente
if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "Xtrain.pkl"
# Forma flexivel de não ter o nome do fichieor fixo no código
    df_train = pd.read_pickle(filename)
    X = np.concatenate(df_train['Skeleton_Sequence'].to_numpy())
    print("Shape de X:", X.shape)  # deve dar (N, 66)
    
# Para testes/debug podemos meter K=valor fixo     
K_valores = range(1, 11)  # testa K de 1 a 10, temos que alterar consoante o enunciado
    custos = []

    for K in K_valores:
        C, z = kmeans(X, K, flag="random")
        _, J = kmeans_custo(X, C, z)
        custos.append(J)
        print(f"K={K} -> custo J={J:.2f}")

    # --- Gráfico ---
    import matplotlib.pyplot as plt

    plt.plot(list(K_valores), custos, marker='o')
    plt.xlabel("Número de centróides (K)")
    plt.ylabel("Custo J")
    plt.title("Custo do k-means em função de K (inicialização random)")
    plt.savefig("PartI2a.jpg")  # nome pedido no enunciado
    plt.show()

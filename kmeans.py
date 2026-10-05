def kmeans(X, K, flag, max_iter=300, tol=1e-6):

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

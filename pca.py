import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kmeans import kmeans, kmeans_custo

# --- R5.a ---
df_train = pd.read_pickle("Xtrain.pkl")
X = np.concatenate(df_train["Skeleton_Sequence"].to_numpy())   # (N, 66)

# Normalização (centrar + escalar), crítica para o PCA
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# PCA sem definir n_components nem threshold
pca = PCA()
pca.fit(X_norm)

# Variância explicada por cada componente e variância acumulada
var_exp = pca.explained_variance_ratio_          # shape (66,)
var_cum = np.cumsum(var_exp)                     # shape (66,)

# Primeiro número de componentes cuja variância acumulada chega a 95%
n_95 = int(np.argmax(var_cum >= 0.95)) + 1
print("Componentes para 95% da variância:", n_95)
print("Variância acumulada com esse número:", f"{var_cum[n_95 - 1]:.4%}")
print(f"Número total de componentes: {pca.n_components_}")
print(f"Variância explicada pelas primeiras {n_95 - 1}: {var_cum[n_95 - 2]:.4%}")
print(f"Variância explicada pelas primeiras {n_95}: "
      f"{var_cum[n_95 - 1]:.4%}")

# Figura de suporte
plt.figure()
plt.plot(range(1, len(var_cum) + 1), var_cum, marker="o", markersize=3)
plt.axhline(0.95, linestyle="--", color="red", label="95%")
plt.axvline(n_95, linestyle="--", color="gray", label=f"{n_95} componentes")
plt.xlabel("Número de componentes principais")
plt.ylabel("Variância explicada acumulada")
plt.title("PCA: variância explicada acumulada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("PartII5a.jpg", dpi=300)

# --- R5.b ---

# Nomes das 66 caracteristicas originais
features = []
for j in range(33):
    features.append(f"Marcador {j} - u")
    features.append(f"Marcador {j} - v")

# Analisar as 8 componentes que explicam pelo menos 95%
for i in range(n_95):
    pesos = pca.components_[i]

    # Ordenar por valor absoluto do peso, do maior para o menor
    indices = np.argsort(np.abs(pesos))[::-1][:10]

    print(f"\nComponente principal {i + 1}")
    print(f"Variancia explicada: "
          f"{pca.explained_variance_ratio_[i]:.4%}")
    print("Top 10 caracteristicas:")

    for idx in indices:
        print(f"{features[idx]}: peso = {pesos[idx]:.4f}")

# Figura de apoio: pesos de cada feature nas 2 primeiras componentes
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for c, ax in enumerate(axs):
    ax.bar(range(66), pca.components_[c])
    ax.set_ylabel(f"Peso em PC{c + 1}")
    ax.grid(True, axis="y")
axs[-1].set_xticks(range(66))
axs[-1].set_xticklabels(
    [f"{i // 2}{'u' if i % 2 == 0 else 'v'}" for i in range(66)],
    rotation=90, fontsize=6
)
axs[-1].set_xlabel("Característica original (marcador + coordenada)")
plt.tight_layout()
plt.savefig("PartII5b.jpg", dpi=300)

# --- R6: PCA implementado de raiz (apenas numpy) ---

def pca_fit(X):
    """
    Ajusta o PCA a X (N, d).
    Devolve: media (d,), valores_proprios (d,), W (d, d)
    onde as colunas de W são as componentes principais, ordenadas
    da que explica mais variância para a que explica menos.
    """
    media = X.mean(axis=0)
    Xc = X - media
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)        # matriz de covariância (d, d)

    valores, vetores = np.linalg.eigh(cov)       # eigh: para matrizes simétricas
    ordem = np.argsort(valores)[::-1]            # eigh devolve ordem crescente
    return media, valores[ordem], vetores[:, ordem]


def pca_project(X, media, W, r):
    """Projeta X (N, d) nas primeiras r componentes -> Z (N, r)."""
    return (X - media) @ W[:, :r]


def pca_reconstruct(Z, media, W):
    """Reconstrói X_hat (N, d) a partir de Z (N, r)."""
    r = Z.shape[1]
    return Z @ W[:, :r].T + media

# --- R6.a ---
media, valores, W = pca_fit(X_norm)
var_ratio = valores / valores.sum()

Z2 = pca_project(X_norm, media, W, r=2)         # (N, 2)
print("Shape de Z2:", Z2.shape)
print("Variância explicada pelas 2 componentes:", var_ratio[:2].sum())

plt.figure(figsize=(7, 6))
plt.scatter(Z2[:, 0], Z2[:, 1], s=5, alpha=0.4)
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.title("Dados projetados em 2 componentes principais")
plt.grid(True)
plt.tight_layout()
plt.savefig("PartII6a.jpg", dpi=300)

# --- R6.b ---
# Reconstrução com r = 2
Z2 = pca_project(X_norm, media, W, r=2)
X_norm_rec = pca_reconstruct(Z2, media, W)          # espaço normalizado
X_rec = scaler.inverse_transform(X_norm_rec)        # espaço original

# MSE global (média sobre todas as poses e todas as características)
mse_orig = np.mean((X - X_rec) ** 2)
mse_norm = np.mean((X_norm - X_norm_rec) ** 2)
print("MSE (espaço original):", mse_orig)
print("MSE (espaço normalizado):", mse_norm)
print("Verificação: 1 - variância explicada =", 1 - var_ratio[:2].sum())

# Erro por característica (66,) e por marcador (33,), somando u e v
erro_feat = np.mean((X - X_rec) ** 2, axis=0)
erro_marc = erro_feat[0::2] + erro_feat[1::2]

# Marcadores com maior erro
ordem = np.argsort(erro_marc)[::-1]
print("\nMarcadores com maior erro de reconstrução:")
for j in ordem[:10]:
    print(f"Marcador {j}: {erro_marc[j]:.6f}")

# Figura
plt.figure(figsize=(10, 4))
plt.bar(range(33), erro_marc)
plt.xlabel("Marcador")
plt.ylabel("Erro quadrático médio (u² + v²)")
plt.title("Erro de reconstrução por marcador (r = 2)")
plt.xticks(range(33))
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("PartII6b.jpg", dpi=300)

# --- R6.c ---
def avaliar_reconstrucao(r):
    """Projeta, reconstrói e mede o erro para r componentes."""
    Z = pca_project(X_norm, media, W, r)
    X_norm_rec = pca_reconstruct(Z, media, W)
    X_rec = scaler.inverse_transform(X_norm_rec)

    var_expl = var_ratio[:r].sum()
    mse_orig = np.mean((X - X_rec) ** 2)
    mse_norm = np.mean((X_norm - X_norm_rec) ** 2)

    erro_feat = np.mean((X - X_rec) ** 2, axis=0)
    erro_marc = erro_feat[0::2] + erro_feat[1::2]
    return Z, var_expl, mse_orig, mse_norm, erro_marc

r_95 = n_95   # 8, obtido na R5.a
Z8, var8, mse_orig8, mse_norm8, erro_marc8 = avaliar_reconstrucao(r_95)
_, var2, mse_orig2, mse_norm2, erro_marc2 = avaliar_reconstrucao(2)

print("Shape de Z (r = 8):", Z8.shape)
print(f"\n{'':22s}{'r = 2':>12s}{'r = ' + str(r_95):>12s}")
print(f"{'Variância explicada':22s}{var2:12.4%}{var8:12.4%}")
print(f"{'MSE (original)':22s}{mse_orig2:12.6f}{mse_orig8:12.6f}")
print(f"{'MSE (normalizado)':22s}{mse_norm2:12.6f}{mse_norm8:12.6f}")
print("Verificação: 1 - variância explicada (r = 8) =", 1 - var8)

ordem8 = np.argsort(erro_marc8)[::-1]
print("\nMarcadores com maior erro (r = 8):")
for j in ordem8[:10]:
    print(f"Marcador {j}: {erro_marc8[j]:.6f}   (r = 2: {erro_marc2[j]:.6f})")

# Figura: erro por marcador, r = 2 vs r = 8, com o mesmo eixo
x = np.arange(33)
plt.figure(figsize=(10, 4))
plt.bar(x - 0.2, erro_marc2, width=0.4, label="r = 2")
plt.bar(x + 0.2, erro_marc8, width=0.4, label=f"r = {r_95}")
plt.xlabel("Marcador")
plt.ylabel("Erro quadrático médio (u² + v²)")
plt.title("Erro de reconstrução por marcador")
plt.xticks(range(33))
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("PartII6c.jpg", dpi=300)
plt.show()
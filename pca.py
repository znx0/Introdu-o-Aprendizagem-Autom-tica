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
plt.show()

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
plt.show()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmeans import kmeans, kmeans_custo
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from kmeans import kmeans

os.makedirs("figures", exist_ok=True)

# -------------------------------------------------------------------
# Dados
# -------------------------------------------------------------------
df = pd.read_pickle("Xtrain.pkl")
seqs = [np.asarray(s) for s in df["Skeleton_Sequence"]]

comprimentos = np.array([s.shape[0] for s in seqs])
print("Colunas:", df.columns.tolist())
print("Número de sequências:", len(seqs))
print("min / mediana / max:", comprimentos.min(), np.median(comprimentos), comprimentos.max())
print("Total de poses:", comprimentos.sum())          # deve dar 8374

todas = np.concatenate(seqs).reshape(-1, 33, 2)
print("u: min/max =", todas[:, :, 0].min().round(3), todas[:, :, 0].max().round(3))
print("v: min/max =", todas[:, :, 1].min().round(3), todas[:, :, 1].max().round(3))


# -------------------------------------------------------------------
# Descritores de forma do corpo
# -------------------------------------------------------------------
PARES = [(11, 12), (23, 24), (11, 13), (13, 15), (12, 14),
         (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)]


def tronco_seq(seq, ar=1.0):
    """Comprimento do tronco (ponto médio dos ombros -> ponto médio das ancas), (T,)."""
    P = seq.reshape(len(seq), 33, 2).astype(float).copy()
    P[:, :, 0] *= ar
    return np.linalg.norm(
        (P[:, 11] + P[:, 12]) / 2 - (P[:, 23] + P[:, 24]) / 2, axis=1
    )


def segmentos_brutos(seq, ar=1.0):
    """Comprimento de cada segmento do corpo, sem normalizar: (T, 10)."""
    P = seq.reshape(len(seq), 33, 2).astype(float).copy()
    P[:, :, 0] *= ar
    return np.stack(
        [np.linalg.norm(P[:, a] - P[:, b], axis=1) for a, b in PARES], axis=1
    )


def segmentos_corpo(seq, ar=1.0):
    """Segmentos normalizados pelo tronco instantâneo: (T, 10)."""
    return segmentos_brutos(seq, ar) / tronco_seq(seq, ar)[:, None]


def descritor_mediana(seq, ar=1.0):
    return np.median(segmentos_corpo(seq, ar), axis=0)


def descritor_q(seq, ar=1.0, q=90):
    # numa projeção 2D os segmentos só encurtam, por isso um percentil
    # alto aproxima melhor o comprimento real do que a mediana
    return np.percentile(segmentos_corpo(seq, ar), q, axis=0)


def descritor_robusto(seq, ar=1.0, q=90):
    # normaliza pelo percentil do tronco da própria sequência (momentos em
    # que o paciente está mais direito), em vez de pelo tronco instantâneo
    tr = np.percentile(tronco_seq(seq, ar), q)
    return np.percentile(segmentos_brutos(seq, ar), q, axis=0) / tr


# -------------------------------------------------------------------
# Variação entre sequências vs. dentro de cada sequência
# -------------------------------------------------------------------
F = np.array([descritor_mediana(s) for s in seqs])                 # (71, 10)
F_std_dentro = np.array([segmentos_corpo(s).std(axis=0) for s in seqs])

print("\nVariação ENTRE sequências (desvio padrão):", F.std(axis=0).round(3))
print("Variação DENTRO de uma sequência (média):", F_std_dentro.mean(axis=0).round(3))


# -------------------------------------------------------------------
# Teste de fiabilidade (split-half)
# -------------------------------------------------------------------
def fiabilidade_metades(seqs, f, nome):
    """
    Compara a 1.ª metade de cada sequência com a 2.ª metade de todas as
    sequências. Se o descritor identifica o paciente, a metade da própria
    sequência deve ficar entre as mais próximas.
    """
    A = np.array([f(s[:len(s) // 2]) for s in seqs])
    B = np.array([f(s[len(s) // 2:]) for s in seqs])
    sc = StandardScaler().fit(np.vstack([A, B]))
    A, B = sc.transform(A), sc.transform(B)

    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)      # (71, 71)
    propria = np.diag(d)
    posicao = (d < propria[:, None]).sum(axis=1)   # 0 = a própria metade é a mais próxima

    print(f"\n[{nome}]")
    print("  Mediana da posição:", np.median(posicao), f"(ao acaso seria ~{len(seqs) // 2})")
    print("  Fração em que a própria metade é a mais próxima:", round(np.mean(posicao == 0), 3))


fiabilidade_metades(seqs, descritor_mediana, "mediana, ar = 1")
fiabilidade_metades(seqs, descritor_q, "percentil 90, ar = 1")
fiabilidade_metades(seqs, descritor_robusto, "robusto (tronco p90), ar = 1")

AR = 16 / 9
fiabilidade_metades(seqs, lambda s: descritor_mediana(s, AR), f"mediana, ar = {AR:.2f}")
fiabilidade_metades(seqs, lambda s: descritor_q(s, AR), f"percentil 90, ar = {AR:.2f}")
fiabilidade_metades(seqs, lambda s: descritor_robusto(s, AR), f"robusto, ar = {AR:.2f}")


# -------------------------------------------------------------------
# Matrizes de descritores normalizadas
# -------------------------------------------------------------------
F_norm = StandardScaler().fit_transform(F)
F90 = np.array([descritor_q(s) for s in seqs])
F90_norm = StandardScaler().fit_transform(F90)
F_rob = np.array([descritor_robusto(s) for s in seqs])
F_rob_norm = StandardScaler().fit_transform(F_rob)


def sil(Fn, z):
    return silhouette_score(Fn, z) if len(np.unique(z)) > 1 else float("nan")


# -------------------------------------------------------------------
# Silhouette por número de grupos (k-means do grupo, km++)
# -------------------------------------------------------------------
print("\nSilhouette por número de grupos:")
print("   K | mediana |   p90   | robusto")
for K in range(2, 16):
    _, z1 = kmeans(F_norm, K, "km++")
    _, z2 = kmeans(F90_norm, K, "km++")
    _, z3 = kmeans(F_rob_norm, K, "km++")
    print(f"  {K:2d} | {sil(F_norm, z1):7.3f} | {sil(F90_norm, z2):7.3f} | {sil(F_rob_norm, z3):7.3f}")


# -------------------------------------------------------------------
# Dendrogramas (mediana e robusto)
# -------------------------------------------------------------------
Z = linkage(F_norm, method="ward")
plt.figure(figsize=(12, 4))
dendrogram(Z, color_threshold=0)
plt.xlabel("Sequência")
plt.ylabel("Distância (Ward)")
plt.title("Dendrograma das sequências (descritor: mediana)")
plt.tight_layout()
plt.savefig("figures/PartIV10.jpg", dpi=300)

Z_rob = linkage(F_rob_norm, method="ward")
plt.figure(figsize=(12, 4))
dendrogram(Z_rob, color_threshold=0)
plt.xlabel("Sequência")
plt.ylabel("Distância (Ward)")
plt.title("Dendrograma das sequências (descritor robusto)")
plt.tight_layout()
plt.savefig("figures/PartIV10_robusto.jpg", dpi=300)

for nome, ZZ in (("mediana", Z), ("robusto", Z_rob)):
    for K in (2, 3, 4):
        grupos = fcluster(ZZ, t=K, criterion="maxclust")
        print(f"\nDendrograma ({nome}) cortado em {K} grupos -> tamanhos:",
              np.bincount(grupos)[1:])


# -------------------------------------------------------------------
# Estabilidade: os grupos dependem do descritor? (ARI)
# -------------------------------------------------------------------
print("\nEstabilidade (ARI entre descritores):")
for K in (2, 3, 4):
    _, z_med = kmeans(F_norm, K, "km++")
    _, z_p90 = kmeans(F90_norm, K, "km++")
    _, z_rob = kmeans(F_rob_norm, K, "km++")
    print(f"  K = {K}: ARI(mediana, p90) = {adjusted_rand_score(z_med, z_p90):.3f}"
          f" | ARI(mediana, robusto) = {adjusted_rand_score(z_med, z_rob):.3f}"
          f" | ARI(p90, robusto) = {adjusted_rand_score(z_p90, z_rob):.3f}")
    print(f"        tamanhos: mediana {np.bincount(z_med)} | p90 {np.bincount(z_p90)}"
          f" | robusto {np.bincount(z_rob)}")


# -------------------------------------------------------------------
# Diagnósticos: o que está a separar os grupos?
# -------------------------------------------------------------------
t_med = np.array([np.median(tronco_seq(s)) for s in seqs])
t_min = np.array([tronco_seq(s).min() for s in seqs])
print("\nTronco mediano por sequência (min/mediana/max):",
      t_med.min().round(3), np.median(t_med).round(3), t_med.max().round(3))
print("Piores razões tronco mínimo / tronco mediano:", np.sort(t_min / t_med)[:5].round(2))

ext = np.abs(F90_norm).max(axis=1)
print("Sequências com valores mais extremos no p90:", np.argsort(ext)[::-1][:8])
print("  valores:", np.sort(ext)[::-1][:8].round(1))

for nome, Fn in (("p90", F90_norm), ("robusto", F_rob_norm)):
    _, z = kmeans(Fn, 2, "km++")
    for g in range(2):
        idx = np.where(z == g)[0]
        print(f"Grupo {nome} {g}: n = {len(idx)}, comprimento mediano = {np.median(comprimentos[idx])},"
              f" tronco mediano = {np.median(t_med[idx]):.3f}")

# Cruzamento com a Q9 (quando existir o agrupamento por exercício):
# print(pd.crosstab(grupo_exercicio, z_rob))

# k-means com vários arranques: fica com o de menor custo J
def kmeans_melhor(X, K, flag="km++", n=20):
    melhor = None
    for _ in range(n):
        C, z = kmeans(X, K, flag)
        J = kmeans_custo(X, C, z)[1]
        if melhor is None or J < melhor[0]:
            melhor = (J, C, z)
    return melhor[1], melhor[2]

# Estabilidade entre metades: agrupa os descritores calculados só com a 1.ª
# metade e só com a 2.ª metade de cada sequência (Ward, determinístico).
# Se os grupos refletirem algo estável de cada sequência, devem coincidir.
FA = StandardScaler().fit_transform(np.array([descritor_robusto(s[:len(s) // 2]) for s in seqs]))
FB = StandardScaler().fit_transform(np.array([descritor_robusto(s[len(s) // 2:]) for s in seqs]))
ZA, ZB = linkage(FA, method="ward"), linkage(FB, method="ward")

print("\nEstabilidade entre metades (ARI, Ward):")
for K in (2, 3, 4):
    gA = fcluster(ZA, t=K, criterion="maxclust")
    gB = fcluster(ZB, t=K, criterion="maxclust")
    g = fcluster(Z_rob, t=K, criterion="maxclust")
    print(f"  K = {K}: ARI(A, B) = {adjusted_rand_score(gA, gB):.3f}"
          f" | ARI(A, completo) = {adjusted_rand_score(gA, g):.3f}"
          f" | ARI(B, completo) = {adjusted_rand_score(gB, g):.3f}")

# Os 3 grupos do robusto estão a separar o comprimento das sequências?
g3 = fcluster(Z_rob, t=3, criterion="maxclust")
for k in np.unique(g3):
    idx = np.where(g3 == k)[0]
    print(f"Grupo {k}: n = {len(idx)}, comprimento mediano = {np.median(comprimentos[idx])},"
          f" intervalo = {comprimentos[idx].min()}-{comprimentos[idx].max()}")

# ARI robusto vs mediana com 20 arranques por K
print("\nARI com vários arranques:")
for K in (2, 3, 4):
    _, za = kmeans_melhor(F_rob_norm, K)
    _, zb = kmeans_melhor(F_norm, K)
    print(f"  K = {K}: ARI(robusto, mediana) = {adjusted_rand_score(za, zb):.3f}"
          f" | tamanhos: {np.bincount(za)} e {np.bincount(zb)}")

# 1) O descritor robusto depende do comprimento da sequência?
logL = np.log(comprimentos)
print("\nSpearman entre cada descritor robusto e log(comprimento):")
for j, (a, b) in enumerate(PARES):
    rho, _ = spearmanr(F_rob[:, j], logL)
    print(f"  segmento {a:2d}-{b:2d}: rho = {rho:+.2f}")

# 2) Descritor em janelas do mesmo tamanho (30 poses = comprimento mínimo)
def janela_central(seq, n=30):
    ini = (len(seq) - n) // 2
    return seq[ini:ini + n]

F_jan = StandardScaler().fit_transform(
    np.array([descritor_robusto(janela_central(s)) for s in seqs])
)
Z_jan = linkage(F_jan, method="ward")

print("\nJanelas de 30 poses (comprimento igual para todas):")
for K in (2, 3):
    g_jan = fcluster(Z_jan, t=K, criterion="maxclust")
    g = fcluster(Z_rob, t=K, criterion="maxclust")
    print(f"  K = {K}: ARI(janela, completo) = {adjusted_rand_score(g_jan, g):.3f}"
          f" | tamanhos: {np.bincount(g_jan)[1:]}")

# 3) Que sequência fica isolada em K = 4?
g4 = fcluster(Z_rob, t=4, criterion="maxclust")
print("\nSequências isoladas em K = 4:", np.where(np.bincount(g4)[g4] == 1)[0])

plt.show()   # última linha do ficheiro
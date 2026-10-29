import numpy as np
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def visualize_sequences(sequence, title="Skeleton Sequence Animation"):
    """
    Visualizes a skeleton sequence of shape (num_frames, 66) as a video with drawn edges.
    Each frame consists of 33 joints with (u, v) coordinates.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.invert_yaxis() # MediaPipe coordinates have y going downwards
    ax.set_title(title)

    # Reshape sequence to (num_frames, 33, 2)
    num_frames = sequence.shape[0]
    print(f"Total Frames to Animate: {num_frames}")
    coords = sequence.reshape(num_frames, 33, 2)

    # Plot joints
    scatter = ax.scatter([], [], c='blue', s=30, zorder=3)

    # Define connections (bones) for MediaPipe Pose (33 keypoints)
    mp_connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ]

    # Initialize lines for each connection
    lines = [ax.plot([], [], color='red', linewidth=2, zorder=2)[0] for _ in mp_connections]

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        for line in lines:
            line.set_data([], [])
        return [scatter] + lines

    def update(frame):
        frame_coords = coords[frame]
        scatter.set_offsets(frame_coords)

        # Update line data
        for line, (i, j) in zip(lines, mp_connections):
            # Check if points are valid/present
            x = [frame_coords[i, 0], frame_coords[j, 0]]
            y = [frame_coords[i, 1], frame_coords[j, 1]]
            line.set_data(x, y)

        return [scatter] + lines

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=num_frames, interval=100, blit=True)
    plt.close(fig)
    return HTML(anim.to_html5_video())
    
def calcular_angulo(p1, p2, p3):
    """
    Calcula o ângulo em graus entre três pontos: p1 (início), p2 (vértice) e p3 (fim).
    Os pontos são vetores de coordenadas 2D ou 3D.
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cosseno_angulo = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    angulo = np.arccos(np.clip(cosseno_angulo, -1.0, 1.0))
    return np.degrees(angulo)

def obter_nomes_atributos():
    """
    Gera nomes explícitos para os atributos correspondentes aos vetores extraídos em português.
    """
    rotulos_pontos_chave = [
        "nariz", "olho_esquerdo_interno", "olho_esquerdo", "olho_esquerdo_externo",
        "olho_direito_interno", "olho_direito", "olho_direito_externo",
        "orelha_esquerda", "orelha_direita", "boca_esquerda", "boca_direita",
        "ombro_esquerdo", "ombro_direito", "cotovelo_esquerdo", "cotovelo_direito",
        "punho_esquerdo", "punho_direito", "mindinho_esquerdo", "mindinho_direito",
        "indicador_esquerdo", "indicador_direito", "polegar_esquerdo", "polegar_direito",
        "quadril_esquerdo", "quadril_direito", "joelho_esquerdo", "joelho_direito",
        "tornozelo_esquerdo", "tornozelo_direito", "calcanhar_esquerdo", "calcanhar_direito",
        "pe_esquerdo_dedo", "pe_direito_dedo"
    ]

    nomes = []
    # Atributos estatísticos de velocidade
    for pc in rotulos_pontos_chave:
        nomes.append(f"{pc}_media_vel_x")
        nomes.append(f"{pc}_media_vel_y")
    for pc in rotulos_pontos_chave:
        nomes.append(f"{pc}_desvio_padrao_vel_x")
        nomes.append(f"{pc}_desvio_padrao_vel_y")

    # Atributos estatísticos de aceleração
    for pc in rotulos_pontos_chave:
        nomes.append(f"{pc}_media_acel_x")
        nomes.append(f"{pc}_media_acel_y")
    for pc in rotulos_pontos_chave:
        nomes.append(f"{pc}_desvio_padrao_acel_x")
        nomes.append(f"{pc}_desvio_padrao_acel_y")

    # Atributos de amplitude de movimento (ROM) articular
    nomes.extend([
        "amplitude_movimento_cotovelo_esquerdo",
        "amplitude_movimento_cotovelo_direito",
        "amplitude_movimento_joelho_esquerdo",
        "amplitude_movimento_joelho_direito"
    ])
    return nomes

def extrair_atributos(sequencia):
    """
    Extrai a amplitude de movimento (ROM) das articulações bem como atributos estatísticos
    baseados em velocidade e aceleração a partir de uma sequência de coordenadas de articulações
    de formato (num_frames, 66).
    """
    # 1. Atributos estatísticos de movimento padrão
    if len(sequencia) <= 1:
        return np.zeros(66 * 4 + 4)  # Espaço reservado

    coords = sequencia.reshape(sequencia.shape[0], 33, 2)

    velocidades = np.diff(coords, axis=0)
    aceleracoes = np.diff(velocidades, axis=0) if len(velocidades) > 1 else np.zeros_like(velocidades)

    media_vel = np.mean(velocidades, axis=0).flatten()
    desvio_padrao_vel = np.std(velocidades, axis=0).flatten()
    media_acel = np.mean(aceleracoes, axis=0).flatten()
    desvio_padrao_acel = np.std(aceleracoes, axis=0).flatten()

    atributos_movimento = np.concatenate([media_vel, desvio_padrao_vel, media_acel, desvio_padrao_acel])

    # 2. Ângulos Articulares & Amplitude de Movimento (ROM)
    trios_articulacoes = [
        (11, 13, 15), # Cotovelo Esquerdo
        (12, 14, 16), # Cotovelo Direito
        (23, 25, 27), # Joelho Esquerdo
        (24, 26, 28)  # Joelho Direito
    ]

    atributos_rom = []
    for inicio, vertice, fim in trios_articulacoes:
        angulos = []
        for frame in range(coords.shape[0]):
            p1 = coords[frame, inicio]
            p2 = coords[frame, vertice]
            p3 = coords[frame, fim]
            angulos.append(calcular_angulo(p1, p2, p3))

        # A Amplitude de Movimento é a diferença entre o maior e o menor ângulo atingido
        rom = np.max(angulos) - np.min(angulos)
        atributos_rom.append(rom)

    # Combina a dinâmica do movimento e os limites físicos de ROM articular
    return np.concatenate([atributos_movimento, np.array(atributos_rom)])

def imprimir_arvore(no, profundidade=0, nomes_atributos=None, idx_to_class=None):
    """
    Exibe recursivamente a estrutura da Árvore de Decisão customizada em português.
    Esta versão é auto-contida e adequada para distribuição a alunos.

    Parâmetros:
    ----------
    no : Node
        O nó atual da árvore (raiz ou sub-árvore a ser impressa).
    profundidade : int, opcional
        Nível atual de profundidade (usado para controlar o recuo do texto). Padrão é 0.
    nomes_atributos : list of str, opcional
        Lista ordenada contendo os nomes legíveis de cada atributo. Padrão é None.
    idx_to_class : dict, opcional
        Dicionário para mapear índices de classes numéricas em nomes legíveis (ex: {0: 'E1'}). Padrão é None.
    """
    recuo = "  " * profundidade
    if no.is_leaf_node():
        if idx_to_class is not None and no.value in idx_to_class:
            nome_classe = idx_to_class[no.value]
        else:
            nome_classe = f"Classe {no.value}"
        print(f"{recuo}└── Folha: {nome_classe}")
        return

    # Determina a descrição do atributo
    if nomes_atributos is not None and no.feature < len(nomes_atributos):
        desc_atributo = nomes_atributos[no.feature]
    else:
        desc_atributo = f"Atributo {no.feature}"

    print(f"{recuo}├── [{desc_atributo} <= {no.limiar:.6f}]")

    # Imprime o Ramo Esquerdo (Verdadeiro)
    print(f"{recuo}  [Verdadeiro]")
    imprimir_arvore(no.esquerda, profundidade + 2, nomes_atributos, idx_to_class)

    # Imprime o Ramo Direito (Falso)
    print(f"{recuo}  [Falso]")
    imprimir_arvore(no.direita, profundidade + 2, nomes_atributos, idx_to_class)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registrar hooks para capturar ativações e gradientes
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None, actual_len=None):
        # Passagem direta (Forward)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()

        # Passagem inversa (Backward) com foco na classe de interesse
        loss = output[0, class_idx]
        loss.backward()

        # Cálculo dos pesos dos canais usando a média global dos gradientes
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Ponderação dos mapas de ativação pelos pesos correspondentes
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Média dos mapas ponderados (gerando o mapa de calor 2D)
        heatmap = torch.mean(self.activations, dim=1).squeeze()

        # Aplicar ReLU para manter apenas recursos que contribuem positivamente para a classe
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)

        # Normalizar o mapa de calor
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        # Se o comprimento real for fornecido, recorte o heatmap para remover o preenchimento (padding)
        if actual_len is not None:
            # A resolução temporal do heatmap pode ser reduzida dependendo do pooling da rede.
            # A arquitetura possui MaxPool2d duas vezes, o que reduz as dimensões por um fator de 4.
            pool_factor = 4
            reduced_len = int(np.ceil(actual_len / pool_factor))
            heatmap = heatmap[:, :reduced_len]

        return heatmap, class_idx

def plot_gradcam(heatmap, input_image, predicted_class_name, actual_len=None):
    """
    Faz o gráfico da representação original da entrada (coordenadas x tempo)
    """
    # Recortar imagem original se o comprimento real estiver definido
    if actual_len is not None:
        input_image = input_image[:, :actual_len]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Mostrar imagem original de entrada da CNN (sem preenchimento)
    im0 = axes[0].imshow(input_image, aspect='auto', cmap='viridis')
    axes[0].set_ylabel("Coordenadas (66)")
    axes[0].set_xlabel("Tempo (Frames)")
    fig.colorbar(im0, ax=axes[0])

    # Mostrar o Grad-CAM reconstruído (sem preenchimento)
    im1 = axes[1].imshow(heatmap, aspect='auto', cmap='jet', alpha=0.8)
    axes[1].set_ylabel("Mapa de Saliência")
    axes[1].set_xlabel("Tempo")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()
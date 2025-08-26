import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# --- NOVO: Fixar a semente aleatória para resultados reproduzíveis ---
# Isso garante que a mesma imagem seja gerada toda vez que o código rodar.
np.random.seed(42)

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')
ax.set_axis_off()

layer_specs = [
    ('Entrada\n(1247 Features.)', 10, 0),
    ('Densa 1\n(260 neurônios)', 8, 0.3),
    ('Densa 2\n(160 neurônios)', 7, 0.25),
    ('Densa 3\n(64 neurônios)', 5, 0.1),
    ('Densa 4\n(24 neurônios)', 4, 0),
    ('Saída\n(3 classes)', 3, 0)
]
n_layers = len(layer_specs)
layer_spacing = 3.0
neuron_radius = 0.2


# --- ETAPA 1: Determinar quais neurônios serão "desligados" pelo dropout ---
dropped_neurons = {}
for i, (_, n_neurons, dropout_rate) in enumerate(layer_specs):
    if dropout_rate > 0:
        # Calcula o número de neurônios a serem desligados
        n_to_drop = int(n_neurons * dropout_rate)
        # Escolhe aleatoriamente os índices dos neurônios a serem desligados
        dropped_indices = np.random.choice(range(n_neurons), n_to_drop, replace=False)
        dropped_neurons[i] = dropped_indices


# --- ETAPA 2: Desenhar os neurônios, marcando visualmente os que foram "desligados" ---
neuron_positions = []
for i, (name, n_neurons, dropout_rate) in enumerate(layer_specs):
    x = i * layer_spacing
    layer_pos = []
    y_start = - (n_neurons - 1) / 2.0
    
    # Pega a lista de neurônios desligados para esta camada (ou uma lista vazia se não houver dropout)
    dropped_indices_in_layer = dropped_neurons.get(i, [])

    for j in range(n_neurons):
        y = y_start + j * 0.7
        
        color = 'royalblue'
        
        # Se o neurônio atual está na lista de desligados, muda sua cor
        if j in dropped_indices_in_layer:
            color = '#d3d3d3' # Cinza claro para neurônio inativo
        
        # Lógica para colorir a saída (permanece a mesma)
        if i == n_layers - 1 and j == 1: 
            color = 'gold'

        neuron = Circle((x, y), neuron_radius, facecolor=color, edgecolor='black', zorder=4)
        ax.add_patch(neuron)
        layer_pos.append((x, y))
        
    neuron_positions.append(layer_pos)
    ax.text(x, y + 0.8, name, ha='center', fontsize=10)
    if dropout_rate > 0:
        ax.text(x, y_start - 0.8, f'Dropout ({dropout_rate})', ha='center', fontsize=9, style='italic', color='red')


# --- ETAPA 3: Desenhar as conexões, pulando as dos neurônios "desligados" ---
for i in range(n_layers - 1):
    # Pega a lista de neurônios desligados na camada de ORIGEM (i)
    dropped_indices_in_layer = dropped_neurons.get(i, [])
    
    # 'enumerate' nos dá o índice 'j' de cada neurônio de origem
    for j, (x1, y1) in enumerate(neuron_positions[i]):
        
        # Se o neurônio de origem (j) foi desligado, pula para o próximo com 'continue'
        if j in dropped_indices_in_layer:
            continue # Não desenha NENHUMA conexão saindo deste neurônio
        
        for x2, y2 in neuron_positions[i + 1]:
            ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.2, zorder=1)
            
# A seta destacada pode ser removida ou mantida para indicar o fluxo principal
# Vamos mantê-la para mostrar o caminho do resultado final.
start_node = neuron_positions[0][len(neuron_positions[0]) // 2]
end_node = neuron_positions[-1][1] 
arrow = FancyArrowPatch(start_node, end_node,
                        connectionstyle="arc3,rad=0.1",
                        color="gold",
                        arrowstyle="-|>",
                        mutation_scale=20,
                        linewidth=2.5,
                        zorder=5)
ax.add_patch(arrow)

ax.autoscale_view()
plt.title("Ilustração da Arquitetura com Efeito de Dropout", fontsize=16)
plt.savefig("arquitetura_com_dropout.png", dpi=300, bbox_inches='tight')

print("\n=======================================================")
print("  Ilustração com Dropout salva como 'arquitetura_com_dropout.png'")
print("=======================================================")
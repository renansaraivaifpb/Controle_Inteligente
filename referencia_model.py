import numpy as np
import matplotlib.pyplot as plt

def plot_receptive_fields():
    """
    Gera um gráfico ilustrando como diferentes arquiteturas
    (MLP, CNN, RNN) processam um sinal de série temporal.
    """
    # Cria um sinal de exemplo (e.g., dados de um acelerômetro)
    time = np.linspace(0, 1, 100)
    signal = np.sin(15 * time) * np.exp(-time * 2) + np.random.randn(100) * 0.1

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Visão Comparativa das Arquiteturas Neurais sobre um Sinal Temporal', fontsize=18)

    # 1. MLP (Perceptron de Múltiplas Camadas)
    ax1 = axes[0]
    ax1.plot(time, signal, color='gray', label='Sinal de Entrada')
    # MLP processa todas as features de uma vez, sem noção de tempo
    ax1.fill_between(time, -1.5, 1.5, color='skyblue', alpha=0.3)
    ax1.text(0.5, 0.5, 'Toda a janela de tempo\né processada como um\nvetor de características\n(sem ordem temporal)', 
             ha='center', va='center', fontsize=12, transform=ax1.transAxes)
    ax1.set_title('MLP: Visão Global e Desestruturada', fontsize=14)
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. CNN (Rede Neural Convolucional)
    ax2 = axes[1]
    ax2.plot(time, signal, color='gray')
    # CNN aplica filtros locais para encontrar padrões
    for i in range(5):
        start_idx = 15 + i * 18
        kernel_size = 10
        ax2.fill_between(time[start_idx:start_idx+kernel_size], -1.5, 1.5, color='lightgreen', alpha=0.7)
    ax2.text(0.5, 0.5, 'Filtros (kernels) deslizam sobre o sinal\npara detectar padrões locais\n(picos, vales, etc.)', 
             ha='center', va='center', fontsize=12, transform=ax2.transAxes)
    ax2.set_title('1D-CNN: Visão Baseada em Padrões Locais', fontsize=14)
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. RNN (Rede Neural Recorrente)
    ax3 = axes[2]
    ax3.plot(time, signal, color='gray')
    # RNN processa a sequência passo a passo, mantendo uma memória
    ax3.arrow(0.1, 0.9, 0.6, 0, head_width=0.1, head_length=0.03, fc='salmon', ec='darkred', 
              transform=ax3.transAxes, length_includes_head=True)
    ax3.text(0.5, 0.5, 'Processamento sequencial passo a passo,\nonde a informação do passado (`h_t-1`)\ninfluencia o presente (`h_t`)', 
             ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    ax3.set_title('RNN: Visão Sequencial com Memória', fontsize=14)
    ax3.set_xlabel('Tempo (s)')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("comparativo_arquiteturas.png", dpi=300)
    plt.show()

# Gerar o plot
plot_receptive_fields()
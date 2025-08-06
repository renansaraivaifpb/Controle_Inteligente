# features/frequency_domain.py

import numpy as np
from scipy.fft import rfft, rfftfreq

def calculate_spectral_features(window_axis, sampling_frequency):
    """
    Calcula features espectrais (Potência e Frequência Dominante) para uma janela de um único eixo.
    Retorna um dicionário de features.
    """
    N = len(window_axis)
    if N < 2:  # Precisa de pelo menos 2 pontos para FFT
        return {'spectral_power': 0, 'dominant_freq': 0}

    # Usa rfft por ser mais eficiente para sinais reais
    fft_values = rfft(window_axis.to_numpy())
    fft_abs = np.abs(fft_values)
    freqs = rfftfreq(N, 1 / sampling_frequency)
    
    # Potência Espectral Total
    # A soma dos quadrados das magnitudes da FFT. Representa a energia total do sinal.
    power = np.sum(fft_abs**2) / N

    # Frequência Dominante
    # Ignora o componente DC (índice 0) para achar a frequência principal do movimento
    dominant_freq_index = np.argmax(fft_abs[1:]) + 1
    dominant_freq = freqs[dominant_freq_index]

    return {
        'spectral_power': power,
        'dominant_freq': dominant_freq
    }
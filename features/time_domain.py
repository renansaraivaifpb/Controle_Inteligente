# features/time_domain.py

import numpy as np
from scipy.stats import skew, kurtosis

def calculate_rms(window_axis):
    """Calcula o Root Mean Square para uma janela de um único eixo."""
    if window_axis.empty:
        return 0
    return np.sqrt(np.mean(window_axis**2))

def calculate_skewness(window_axis):
    """Calcula a Skewness (assimetria) para uma janela de um único eixo."""
    if window_axis.empty:
        return 0
    return skew(window_axis)

def calculate_kurtosis(window_axis):
    """Calcula a Curtose para uma janela de um único eixo."""
    if window_axis.empty:
        return 0
    return kurtosis(window_axis)

def calculate_sma(window_df):
    """C10: Calcula a Signal Magnitude Area para uma janela (todos os eixos)."""
    if window_df.empty:
        return 0
    # A integral se torna uma soma no tempo discreto. A divisão por N é a média.
    return np.mean(np.abs(window_df['accX'])) + np.mean(np.abs(window_df['accY'])) + np.mean(np.abs(window_df['accZ']))

def calculate_sma_horizontal(window_df):
    """C11: Calcula a Signal Magnitude Area no plano horizontal (X, Z) para uma janela."""
    if window_df.empty:
        return 0
    return np.mean(np.abs(window_df['accX'])) + np.mean(np.abs(window_df['accZ']))
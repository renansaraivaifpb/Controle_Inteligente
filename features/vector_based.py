# features/vector_based.py

import numpy as np

def calculate_svm(window_df):
    """C1: Calcula a média do Sum Vector Magnitude (SVM) para uma janela."""
    if window_df.empty:
        return 0
    # A fórmula C1[k] é a magnitude instantânea. Para a janela, pegamos a média.
    svm = np.sqrt(window_df['accX']**2 + window_df['accY']**2 + window_df['accZ']**2)
    return np.mean(svm)

def calculate_svm_horizontal(window_df):
    """C2: Calcula a média do SVM no plano horizontal (X, Z) para uma janela."""
    if window_df.empty:
        return 0
    svm_h = np.sqrt(window_df['accX']**2 + window_df['accZ']**2)
    return np.mean(svm_h)

def calculate_angle_vertical(window_df):
    """C4: Calcula o ângulo médio com a vertical para uma janela."""
    if window_df.empty:
        return 0
    # Nota: A fórmula usa -a_y, sugerindo que o eixo Y pode estar invertido ou representar a vertical.
    # Implementando conforme a fórmula dada.
    angle = np.arctan2(np.sqrt(window_df['accX']**2 + window_df['accZ']**2), -window_df['accY'])
    return np.mean(angle)

def calculate_std_magnitude(window_df):
    """C9: Calcula a magnitude do desvio padrão para uma janela."""
    if window_df.empty:
        return 0
    sigma_x_sq = window_df['accX'].std()**2
    sigma_y_sq = window_df['accY'].std()**2
    sigma_z_sq = window_df['accZ'].std()**2
    return np.sqrt(sigma_x_sq + sigma_y_sq + sigma_z_sq)

def calculate_std_magnitude_horizontal(window_df):
    """C8: Calcula a magnitude do desvio padrão no plano horizontal (X, Z) para uma janela."""
    if window_df.empty:
        return 0
    sigma_x_sq = window_df['accX'].std()**2
    sigma_z_sq = window_df['accZ'].std()**2
    return np.sqrt(sigma_x_sq + sigma_z_sq)
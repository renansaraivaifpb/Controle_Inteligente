# main.py

import pandas as pd
import numpy as np
import os

# Importa as funções dos nossos módulos de features
from features.time_domain import (calculate_rms, calculate_skewness, calculate_kurtosis,
                                  calculate_sma, calculate_sma_horizontal)
from features.vector_based import (calculate_svm, calculate_svm_horizontal, calculate_angle_vertical,
                                   calculate_std_magnitude, calculate_std_magnitude_horizontal)
from features.frequency_domain import calculate_spectral_features

# --- PARÂMETROS DE CONFIGURAÇÃO ---
SAMPLING_FREQUENCY = 50  # Em Hz (ajuste conforme seus dados)
WINDOW_SECONDS = 2       # Tamanho da janela em segundos
WINDOW_SAMPLES = int(SAMPLING_FREQUENCY * WINDOW_SECONDS)


# Para este exemplo, vamos criar dados fictícios se o arquivo não existir
DATA_FILE = 'data/aceleracoes_dos_eixos.csv'
LABEL_COLUMN = 'label'

# --- CARREGAMENTO DE DADOS ---
print(f"Carregando dados de '{DATA_FILE}'...")
df = pd.read_csv(DATA_FILE)

# ADICIONE ESTA LINHA DE VERIFICAÇÃO!
print(f"!!! CONFIRMAÇÃO: O arquivo '{DATA_FILE}' foi carregado e possui {len(df)} linhas. !!!")

print("Dados brutos carregados:")
print(df.head())


if not os.path.exists(DATA_FILE):
    print(f"Arquivo '{DATA_FILE}' não encontrado. Criando dados de exemplo...")
    os.makedirs('data', exist_ok=True)
    timestamps = np.linspace(0, 30, 30 * SAMPLING_FREQUENCY)
    walking_x = np.sin(2 * np.pi * 2 * timestamps[:10*SAMPLING_FREQUENCY]) + np.random.normal(0, 0.1, 10*SAMPLING_FREQUENCY)
    falling_x = np.zeros(10*SAMPLING_FREQUENCY); falling_x[50:60] = np.random.normal(15, 2, 10)
    sitting_x = np.zeros(10*SAMPLING_FREQUENCY)
    data_dict = {
        'accX': np.concatenate([walking_x, falling_x, sitting_x]),
        'accY': np.concatenate([walking_x*0.5, falling_x*0.2, sitting_x*0.1]),
        'accZ': np.concatenate([np.random.normal(9.8, 0.2, 10*SAMPLING_FREQUENCY), falling_x*0.5, np.random.normal(9.8, 0.1, 10*SAMPLING_FREQUENCY)]),
        'label': ['andando'] * len(walking_x) + ['caindo'] * len(falling_x) + ['sentando'] * len(sitting_x)
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(DATA_FILE, index=False)
else:
    print(f"Carregando dados de '{DATA_FILE}'...")
    df = pd.read_csv(DATA_FILE)

print("Dados brutos carregados:")
print(df.head())

# --- PROCESSAMENTO EM JANELAS E EXTRAÇÃO DE FEATURES ---

processed_features = []
print(f"\nIniciando processamento com janelas de {WINDOW_SAMPLES} amostras...")

# Itera sobre os dados em "saltos" do tamanho da janela (sem sobreposição)
for i in range(0, len(df) - WINDOW_SAMPLES, WINDOW_SAMPLES):
    window = df.iloc[i : i + WINDOW_SAMPLES].copy() # .copy() para evitar SettingWithCopyWarning
    
    # O rótulo da janela é a atividade mais frequente nela
    label = window['label'].mode()[0]
    
    # Dicionário para armazenar todas as features desta janela
    feature_row = {}
    
    # --- 1. Features de Domínio do Vetor ---
    feature_row['SVM'] = calculate_svm(window)
    feature_row['SVM_horizontal'] = calculate_svm_horizontal(window)
    feature_row['Angle_vertical'] = calculate_angle_vertical(window)
    feature_row['STD_magnitude'] = calculate_std_magnitude(window)
    feature_row['STD_magnitude_horizontal'] = calculate_std_magnitude_horizontal(window)

    # --- 2. Features de Domínio do Tempo ---
    feature_row['SMA'] = calculate_sma(window)
    feature_row['SMA_horizontal'] = calculate_sma_horizontal(window)
    
    # Features calculadas por eixo
    for axis in ['accX', 'accY', 'accZ']:
        window_axis = window[axis]
        feature_row[f'{axis}_rms'] = calculate_rms(window_axis)
        feature_row[f'{axis}_skew'] = calculate_skewness(window_axis)
        feature_row[f'{axis}_kurtosis'] = calculate_kurtosis(window_axis)
        
        # --- 3. Features de Domínio da Frequência ---
        spectral_feats = calculate_spectral_features(window_axis, SAMPLING_FREQUENCY)
        # Adiciona as features espectrais ao nosso dicionário de linha
        feature_row[f'{axis}_spectral_power'] = spectral_feats['spectral_power']
        feature_row[f'{axis}_dominant_freq'] = spectral_feats['dominant_freq']

    # Adiciona o rótulo da janela
    feature_row['label'] = label
    
    processed_features.append(feature_row)

# Converte a lista de dicionários em um DataFrame final
features_df = pd.DataFrame(processed_features)

print("\n--- Processamento concluído! ---")
print(f"Foram geradas {len(features_df)} linhas de features.")
print("\nExemplo do DataFrame de Features gerado:")
print(features_df.head())

# Salva o resultado em um novo arquivo CSV
output_file = 'features_extraidas.csv'
features_df.to_csv(output_file, index=False)
print(f"\nDataFrame de features salvo em '{output_file}'")

print(features_df.head(15))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1A. DADOS NUMÉRICOS (para as cores do heatmap) ---
# Primeiro, definimos os dados como números puros.
data_numeric = {
    'ANDANDO': [92.3, 8.1, 23.6],
    'CAINDO': [2.6, 88.8, 1.8],
    'SENTANDO': [5.2, 3.1, 74.5],
}

# Rótulos para as linhas (Valores Reais)
index_labels = ['ANDANDO', 'CAINDO', 'SENTANDO']

# Criar o DataFrame numérico
df_conf_matrix_numeric = pd.DataFrame(data_numeric, index=index_labels)

# --- 1B. DADOS EM TEXTO (para as anotações nas células) ---
# Agora, criamos um segundo DataFrame para as anotações,
# formatando cada número do DataFrame anterior como uma string com '%'.
df_conf_matrix_annot = df_conf_matrix_numeric.applymap(lambda x: f'{x:.1f}%')


# Dados do F1 Score (sem alteração)
f1_scores = {
    'ANDANDO': [0.87],
    'CAINDO': [0.93],
    'SENTANDO': [0.48]
}
df_f1 = pd.DataFrame(f1_scores, index=['F1 SCORE'])


# --- 2. Criação da Visualização ---
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Confusion Matrix', fontsize=16, y=1.02, x=0.45)
cmap_custom = sns.light_palette("seagreen", as_cmap=True)


# --- 3. Plotando o Heatmap da Matriz de Confusão ---
# ATENÇÃO ÀS MUDANÇAS AQUI:
sns.heatmap(df_conf_matrix_numeric,          # 1. Usamos os dados NUMÉRICOS para as cores
            annot=df_conf_matrix_annot,       # 2. Usamos o DataFrame de TEXTO para as anotações
            fmt='',                           # 3. REMOVEMOS o 'fmt' pois a anotação já está formatada
            cmap=cmap_custom,
            linewidths=1,
            linecolor='white',
            cbar=False,
            ax=ax,
            annot_kws={"size": 12, "weight": "bold"})

# Customização dos rótulos e eixos (sem alteração)
ax.set_xlabel('Valores Previstos', fontsize=12, labelpad=10)
ax.set_ylabel('Valores Reais', fontsize=12, labelpad=10)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.tick_params(axis='x', labelsize=11, rotation=0)
ax.tick_params(axis='y', labelsize=11, rotation=0)


# --- 4. Adicionando a Linha do F1 Score (sem alteração) ---
for i, col in enumerate(df_f1.columns):
    f1_value = df_f1[col].values[0]
    if isinstance(f1_value, (int, float)):
        text_to_display = f'{f1_value:.2f}'
    else:
        text_to_display = ''
    ax.text(i + 0.5, 3.5, text_to_display, ha='center', va='center', fontsize=12, weight='bold')

ax.text(-0.5, 3.5, 'F1 SCORE', ha='center', va='center', fontsize=12, weight='bold')


# --- 5. Ajustes Finais e Salvamento ---
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('matriz_confusao_modelo_percent.png', dpi=300, bbox_inches='tight')
print("Imagem 'matriz_confusao_modelo_percent.png' salva com sucesso!")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import os

# --- PASSO 0: CONFIGURAÇÃO ---
# Define o nome da pasta onde os dados processados serão salvos
OUTPUT_DIR = 'output_data'
# Cria o diretório se ele não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Os arquivos de saída serão salvos no diretório: '{OUTPUT_DIR}/'")


# --- PASSO 1: CARREGAR O DATASET DE FEATURES ---
print("\nCarregando o dataset de features...")
try:
    features_df = pd.read_csv('features_extraidas.csv')
    print("Dataset carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: O arquivo 'features_extraidas.csv' não foi encontrado.")
    print("Por favor, execute o script 'main.py' primeiro para gerar as features.")
    exit()


# --- PASSO 2: LIMPEZA FINAL DOS DADOS ---
if features_df.isnull().sum().sum() > 0:
    features_df.fillna(0, inplace=True)
    print("Valores NaN (nulos) foram substituídos por 0.")


# --- PASSO 3: PREPARAR E DIVIDIR OS DADOS ---
X = features_df.drop('label', axis=1)
y = features_df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nDados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")


# --- PASSO 4: SALVAR OS CONJUNTOS DE TREINO E TESTE ---
# Recombina as features (X) e os rótulos (y) para salvar os arquivos completos
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Salva em arquivos CSV dentro da pasta de saída
train_path = os.path.join(OUTPUT_DIR, 'treino_dataset.csv')
test_path = os.path.join(OUTPUT_DIR, 'teste_dataset.csv')
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Conjuntos de treino e teste salvos em '{train_path}' e '{test_path}'.")


# --- PASSO 5: TREINAR O MODELO ---
print("\nTreinando o modelo RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Modelo treinado com sucesso!")


# --- PASSO 6: REALIZAR E AVALIAR AS PREDIÇÕES ---
print("\nRealizando previsões no conjunto de teste...")
predictions = model.predict(X_test)

# Acurácia Geral
accuracy = accuracy_score(y_test, predictions)
print(f"\nAcurácia Geral do Modelo: {accuracy * 100:.2f}%")

# Relatório de Classificação Detalhado
print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, predictions))


# --- PASSO 7: ANÁLISE DETALHADA DAS PREDIÇÕES E ERROS ---
# Cria um novo DataFrame para comparar os resultados
evaluation_df = X_test.copy()
evaluation_df['label_verdadeiro'] = y_test
evaluation_df['label_predito'] = predictions
evaluation_df['acertou?'] = (evaluation_df['label_verdadeiro'] == evaluation_df['label_predito'])

# Salva o dataframe de avaliação detalhada
evaluation_path = os.path.join(OUTPUT_DIR, 'predicoes_detalhadas_teste.csv')
evaluation_df.to_csv(evaluation_path, index=False)
print(f"\nAnálise detalhada das predições salva em '{evaluation_path}'.")

# Mostra apenas as linhas onde o modelo errou a previsão
erros_df = evaluation_df[evaluation_df['acertou?'] == False]

print("\n--- ANÁLISE DOS ERROS DE PREDIÇÃO ---")
if erros_df.empty:
    print("O modelo acertou todas as previsões no conjunto de teste! Excelente!")
else:
    print(f"O modelo errou {len(erros_df)} de {len(X_test)} previsões.")
    # Mostra apenas as colunas relevantes para a análise do erro
    print(erros_df[['label_verdadeiro', 'label_predito']])


# --- PASSO 8: VISUALIZAR A MATRIZ DE CONFUSÃO ---
print("\nGerando a Matriz de Confusão...")
fig, ax = plt.subplots(figsize=(10, 7))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap='Blues')
ax.set_title('Matriz de Confusão')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_curve
)

# =============================================
# 1. CONFIGURAÇÕES DE CAMINHOS
# =============================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dados"
MODELS_DIR = BASE_DIR / "modelos"

ARQUIVO_TESTE_SEM_LABEL = "dataset_teste_sem_label.csv"
ARQUIVO_TESTE_COM_LABEL = "dataset_teste_com_label.csv"

MODELO_RF = "rf_model.pkl"
MODELO_MLP = "mlp_model.pkl"
SCALER = "standard_scaler.pkl"
ENCODER = "label_encoder.pkl"

# =============================================
# 2. CARREGAR MODELOS E PRÉ-PROCESSAMENTO
# =============================================
try:
    rf = joblib.load(MODELS_DIR / MODELO_RF)
    mlp = joblib.load(MODELS_DIR / MODELO_MLP)
    scaler = joblib.load(MODELS_DIR / SCALER)
    le = joblib.load(MODELS_DIR / ENCODER)
except FileNotFoundError as e:
    print(f"[ERRO] Arquivo não encontrado: {e}")
    exit()

# =============================================
# 3. CARREGAR DATASETS DE TESTE
# =============================================
try:
    df_sem_label = pd.read_csv(DATA_DIR / ARQUIVO_TESTE_SEM_LABEL)
    df_com_label = pd.read_csv(DATA_DIR / ARQUIVO_TESTE_COM_LABEL)
except FileNotFoundError as e:
    print(f"[ERRO] Dataset não encontrado: {e}")
    exit()

if len(df_sem_label) != len(df_com_label):
    print("[ERRO] Os arquivos com e sem label devem ter o mesmo número de linhas.")
    exit()

if "Label" not in df_com_label.columns:
    print("[ERRO] O arquivo com rótulo deve conter a coluna 'Label'.")
    exit()

# =============================================
# 4. PRÉ-PROCESSAMENTO
# =============================================
X = df_sem_label.copy()
y = df_com_label["Label"]
y_encoded = le.transform(y)

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)
X_scaled = scaler.transform(X)

# =============================================
# 5. PREDIÇÕES
# =============================================
y_pred_rf = rf.predict(X)
y_proba_rf = rf.predict_proba(X)

y_pred_mlp = mlp.predict(X_scaled)
y_proba_mlp = mlp.predict_proba(X_scaled)

# =============================================
# 6. FUNÇÃO DE AVALIAÇÃO
# =============================================
def avaliar_modelo(nome, y_true, y_pred, y_proba):
    print(f"\n Avaliação do modelo: {nome}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall:", recall_score(y_true, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_true, y_pred, average="weighted"))
    print("ROC AUC:", roc_auc_score(y_true, y_proba, multi_class='ovr'))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=le.classes_))

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Matriz de Confusão - {nome}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

    # Curva ROC (se binário)
    if y_proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label=f'{nome} (AUC = {roc_auc_score(y_true, y_proba[:, 1]):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {nome}')
        plt.legend()
        plt.show()

    # Exibir matriz como tabela
    print("\nMatriz de confusão (valores brutos):")
    for i, row in enumerate(cm):
        print(f"{le.classes_[i]} ({i}): {row}")

# =============================================
# 7. AVALIAR MODELOS
# =============================================
avaliar_modelo("Random Forest", y_encoded, y_pred_rf, y_proba_rf)
avaliar_modelo("MLP", y_encoded, y_pred_mlp, y_proba_mlp)

# =============================================
# 8. SALVAR RESULTADOS EM CSV
# =============================================
df_resultado = df_sem_label.copy()
df_resultado["Label"] = y
df_resultado["Pred_RF"] = le.inverse_transform(y_pred_rf)
df_resultado["Pred_MLP"] = le.inverse_transform(y_pred_mlp)
df_resultado["Conf_RF"] = np.max(y_proba_rf, axis=1)
df_resultado["Conf_MLP"] = np.max(y_proba_mlp, axis=1)

saida_path = DATA_DIR / f"avaliacao_resultados.csv"
df_resultado.to_csv(saida_path, index=False)
print(f"\n Arquivo com predições e rótulos salvo em: {saida_path}")
print(" Teste concluído com sucesso.")

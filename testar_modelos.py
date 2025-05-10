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

# Arquivos
ARQUIVO_TESTE = "dataset_filtrado_teste.csv"
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
# 3. CARREGAR NOVO DATASET DE TESTE
# =============================================
try:
    df = pd.read_csv(DATA_DIR / ARQUIVO_TESTE)
except FileNotFoundError:
    print(f"[ERRO] Dataset de teste '{ARQUIVO_TESTE}' não encontrado em {DATA_DIR.resolve()}")
    exit()

# Separar X e y
if "Label" in df.columns:
    X = df.drop("Label", axis=1)
    y = df["Label"]
    y_encoded = le.transform(y)
else:
    X = df.copy()
    y_encoded = None
    print("[AVISO] Dataset não contém a coluna 'Label'. Avaliação completa será limitada.")

# =============================================
# 4. PRÉ-PROCESSAMENTO
# =============================================
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
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d",
                cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
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

# =============================================
# 7. AVALIAÇÃO
# =============================================
if y_encoded is not None:
    avaliar_modelo("Random Forest", y_encoded, y_pred_rf, y_proba_rf)
    avaliar_modelo("MLP", y_encoded, y_pred_mlp, y_proba_mlp)
else:
    print("[INFO] Rótulos não disponíveis. Exibindo contagem das classes previstas:")
    print("Random Forest:", pd.Series(y_pred_rf).value_counts())
    print("MLP:", pd.Series(y_pred_mlp).value_counts())

    # Salvar predições
    df["Pred_RF"] = le.inverse_transform(y_pred_rf)
    df["Pred_MLP"] = le.inverse_transform(y_pred_mlp)
    df["Conf_RF"] = np.max(y_proba_rf, axis=1)
    df["Conf_MLP"] = np.max(y_proba_mlp, axis=1)

    output_path = DATA_DIR / f"predicoes_{ARQUIVO_TESTE}"
    df.to_csv(output_path, index=False)
    print(f"[OK] Arquivo salvo com predições: {output_path}")

print("\n Teste com nova base concluído.")

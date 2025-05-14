import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

# =============================================
# 1. CONFIGURAÇÃO DE CAMINHOS E NOMES DE ARQUIVOS
# =============================================
DATA_DIR = Path("dados")
MODELS_DIR = Path("modelos")

ARQUIVO_TESTE_SEM_LABEL = "dataset_sem_label.csv"
ARQUIVO_TESTE_COM_LABEL = "dataset_com_label.csv"

MODELO_RF = "rf_model.pkl"
MODELO_MLP = "mlp_model.pkl"
SCALER = "standard_scaler.pkl"
ENCODER = "label_encoder.pkl"

# =============================================
# 2. CARREGAMENTO DOS MODELOS E PRÉ-PROCESSADORES
# =============================================
rf = joblib.load(MODELS_DIR / MODELO_RF)
mlp = joblib.load(MODELS_DIR / MODELO_MLP)
scaler = joblib.load(MODELS_DIR / SCALER)
le = joblib.load(MODELS_DIR / ENCODER)

# =============================================
# 3. CARREGAMENTO DOS DADOS DE TESTE
# =============================================
X = pd.read_csv(DATA_DIR / ARQUIVO_TESTE_SEM_LABEL)
y = pd.read_csv(DATA_DIR / ARQUIVO_TESTE_COM_LABEL)["Label"]
y_encoded = le.transform(y)

# =============================================
# 4. PRÉ-PROCESSAMENTO DOS DADOS DE ENTRADA
# =============================================
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)
X_scaled = scaler.transform(X)

# =============================================
# 5. PREDIÇÃO DOS MODELOS
# =============================================
y_pred_rf = rf.predict(X)
y_proba_rf = rf.predict_proba(X)

y_pred_mlp = mlp.predict(X_scaled)
y_proba_mlp = mlp.predict_proba(X_scaled)

# =============================================
# 6. ANÁLISE DETALHADA DE ERROS
# =============================================
def analisar_erros(y_true, y_pred, labels):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n🔍 Falsos Positivos (previsto ataque, mas era BENIGNO):")
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt == "BENIGN" and yp != "BENIGN":
            print(f"➡ Linha {i}: Previsto = {yp} | Real = {yt}")

    print("\n🔍 Falsos Negativos (previsto BENIGNO, mas era ataque):")
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt != "BENIGN" and yp == "BENIGN":
            print(f"➡ Linha {i}: Previsto = {yp} | Real = {yt}")

# =============================================
# 7. FUNÇÃO DE AVALIAÇÃO DOS MODELOS
# =============================================
def avaliar_modelo(nome, y_true, y_pred, y_proba):
    print(f"\n📊 Avaliação - {nome}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall:", recall_score(y_true, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_true, y_pred, average="weighted"))
    print("ROC AUC:", roc_auc_score(y_true, y_proba, multi_class='ovr'))

    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Matriz de Confusão - {nome}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

    # Curva ROC - multi-classe
    y_true_bin = label_binarize(y_true, classes=range(len(le.classes_)))
    n_classes = y_proba.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    plt.figure()
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{le.classes_[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falso Positivo (FPR)')
    plt.ylabel('Verdadeiro Positivo (TPR)')
    plt.title(f'Curva ROC Multi-classe - {nome}')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # Erros principais
    benign_index = np.where(le.classes_ == "BENIGN")[0][0]
    false_positives = cm[:, benign_index].sum() - cm[benign_index, benign_index]
    false_negatives = cm[benign_index].sum() - cm[benign_index, benign_index]

    print(f"\n❌ Falsos Positivos (BENIGNO como ATAQUE): {false_positives}")
    print(f"❌ Falsos Negativos (ATAQUE como BENIGNO): {false_negatives}")

    # Erros linha a linha
    print("\n📌 Erros detalhados:")
    analisar_erros(le.inverse_transform(y_true), le.inverse_transform(y_pred), le.classes_)

# =============================================
# 8. AVALIAR AMBOS OS MODELOS
# =============================================
avaliar_modelo("Random Forest", y_encoded, y_pred_rf, y_proba_rf)
avaliar_modelo("MLP", y_encoded, y_pred_mlp, y_proba_mlp)

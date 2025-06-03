import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
import time

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
IMPUTER = "imputer.pkl"
SELECTOR = "feature_selector.pkl"

# =============================================
# 2. CARREGAMENTO DOS MODELOS E PRÉ-PROCESSADORES
# =============================================
rf = joblib.load(MODELS_DIR / MODELO_RF)
mlp = joblib.load(MODELS_DIR / MODELO_MLP)
scaler = joblib.load(MODELS_DIR / SCALER)
le = joblib.load(MODELS_DIR / ENCODER)
imputer = joblib.load(MODELS_DIR / IMPUTER)
selector = joblib.load(MODELS_DIR / SELECTOR)

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
X_imputed = imputer.transform(X)
X_selected = selector.transform(X_imputed)
X_scaled = scaler.transform(X_selected)

# =============================================
# 5. PREDIÇÃO DOS MODELOS
# =============================================
start_rf = time.time()
y_pred_rf = rf.predict(X_selected)
y_proba_rf = rf.predict_proba(X_selected)
rf_time = time.time() - start_rf

start_mlp = time.time()
y_pred_mlp = mlp.predict(X_scaled)
y_proba_mlp = mlp.predict_proba(X_scaled)
mlp_time = time.time() - start_mlp

print(f"\n⏱️ Tempo de classificação - Random Forest: {rf_time:.4f} segundos")
print(f"⏱️ Tempo de classificação - MLP: {mlp_time:.4f} segundos")

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
# 7. AVALIAÇÃO DOS MODELOS
# =============================================
def avaliar_modelo(nome, y_true, y_pred, y_proba):
    classes = le.classes_

    print(f"\n📊 Avaliação - {nome}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision (weighted):", precision_score(y_true, y_pred, average="weighted", zero_division=0))
    print("Recall (weighted):", recall_score(y_true, y_pred, average="weighted", zero_division=0))
    print("F1 Score (weighted):", f1_score(y_true, y_pred, average="weighted", zero_division=0))

    print("\n📝 Classification Report:\n", classification_report(y_true, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_true, y_pred)

    # ROC AUC por classe
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = y_proba.shape[1]
    roc_auc = dict()
    for i in range(n_classes):
        try:
            roc_auc[i] = auc(*roc_curve(y_true_bin[:, i], y_proba[:, i])[:2])
        except ValueError:
            roc_auc[i] = float('nan')

    # Análise detalhada de erros
    benign_index = np.where(classes == "BENIGN")[0][0]
    false_positives = cm[:, benign_index].sum() - cm[benign_index, benign_index]
    false_negatives = cm[benign_index].sum() - cm[benign_index, benign_index]

    print(f"\n❌ Falsos Positivos (BENIGNO classificado como ATAQUE): {false_positives}")
    print(f"❌ Falsos Negativos (ATAQUE classificado como BENIGNO): {false_negatives}")

    # Métricas por classe
    metrics_table = []
    for i, classe in enumerate(classes):
        metrics_table.append([
            classe,
            f"{precision_score(y_true, y_pred, labels=[i], average=None)[0]:.4f}",
            f"{recall_score(y_true, y_pred, labels=[i], average=None)[0]:.4f}",
            f"{f1_score(y_true, y_pred, labels=[i], average=None)[0]:.4f}",
            f"{roc_auc[i]:.4f}"
        ])

    print("\n📌 Métricas por Classe:")
    print(tabulate(metrics_table,
                   headers=["Classe", "Precision", "Recall", "F1-Score", "ROC AUC"],
                   tablefmt="grid",
                   stralign="center",
                   numalign="center"))

    print("\n📌 Erros detalhados:")
    analisar_erros(le.inverse_transform(y_true), le.inverse_transform(y_pred), le.classes_)

# =============================================
# 8. AVALIAR AMBOS OS MODELOS
# =============================================
avaliar_modelo("Random Forest", y_encoded, y_pred_rf, y_proba_rf)
avaliar_modelo("MLP", y_encoded, y_pred_mlp, y_proba_mlp)

# =============================================
# 9. TABELA COMPARATIVA DE MÉTRICAS POR CLASSE
# =============================================
def gerar_tabela_comparativa(y_true, y_pred_dict, y_proba_dict, label_encoder):
    resultados = []
    classes = label_encoder.classes_
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))

    for nome_modelo, y_pred in y_pred_dict.items():
        y_proba = y_proba_dict[nome_modelo]

        precisao, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(classes)), zero_division=0
        )

        roc_auc_scores = []
        for i in range(len(classes)):
            try:
                roc_auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
            except ValueError:
                roc_auc = float('nan')
            roc_auc_scores.append(roc_auc)

        for idx, classe in enumerate(classes):
            resultados.append({
                "Modelo": nome_modelo,
                "Classe": classe,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precisao[idx],
                "Recall": recall[idx],
                "F1-Score": f1[idx],
                "ROC AUC": roc_auc_scores[idx]
            })

    df_resultados = pd.DataFrame(resultados)
    return df_resultados

# Executa a comparação
y_pred_dict = {
    "Random Forest": y_pred_rf,
    "MLP": y_pred_mlp
}

y_proba_dict = {
    "Random Forest": y_proba_rf,
    "MLP": y_proba_mlp
}

tabela_metricas = gerar_tabela_comparativa(y_encoded, y_pred_dict, y_proba_dict, le)

print("\n📋 Tabela Comparativa de Métricas por Classe:\n")
print(tabela_metricas.to_string(index=False))

# Arredondar e exibir com tabulate
tabela_metricas_formatada = tabela_metricas.copy()
tabela_metricas_formatada[["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]] = \
    tabela_metricas_formatada[["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]].round(6)

print("\n📋 Tabela Comparativa de Métricas por Classe:\n")
print(tabulate(tabela_metricas_formatada, headers='keys', tablefmt='grid', showindex=False))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)

# Leitura dos datasets separados
df_train = pd.read_csv("dados/dataset_filtrado.csv")
df_test = pd.read_csv("dados/dataset_filtrado_teste.csv")

# Separar X e y
X_train, y_train = df_train.drop("Label", axis=1), df_train["Label"]
X_test, y_test = df_test.drop("Label", axis=1), df_test["Label"]

# Remover inf e NaN
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.dropna(inplace=True)
X_test.dropna(inplace=True)
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

# Codificação de rótulos (fit no treino, transform no teste)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Normalização (fit no treino, transform no teste)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento - MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
mlp.fit(X_train_scaled, y_train_enc)
y_pred_mlp = mlp.predict(X_test_scaled)
y_proba_mlp = mlp.predict_proba(X_test_scaled)

# Treinamento - Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_enc)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)

# Função de avaliação
def avaliar_modelo(nome, y_true, y_pred, y_proba):
    print(f"\nAvaliação do modelo: {nome}")
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

    # Curva ROC (binária)
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

# Avaliação
avaliar_modelo("MLP", y_test_enc, y_pred_mlp, y_proba_mlp)
avaliar_modelo("Random Forest", y_test_enc, y_pred_rf, y_proba_rf)

# Cria diretório 'modelos'
Path("modelos").mkdir(exist_ok=True)

# Salva modelos e pré-processadores
joblib.dump(mlp, Path("modelos/mlp_model.pkl"))
joblib.dump(rf, Path("modelos/rf_model.pkl"))
joblib.dump(scaler, Path("modelos/standard_scaler.pkl"))
joblib.dump(le, Path("modelos/label_encoder.pkl"))

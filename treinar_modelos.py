import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)

from pathlib import Path

# 1. Leitura dos dados
caminho_dataset = Path("dados") / "dataset_filtrado.csv"
df = pd.read_csv(caminho_dataset)

# 2. Separação X e y
X = df.drop("Label", axis=1)
y = df["Label"]

# 2.5 Limpeza de dados (ADICIONE AQUI)
print("Valores infinitos antes da limpeza:", np.isinf(X).sum().sum())
X.replace([np.inf, -np.inf], np.nan, inplace=True)
print("Valores NaN após substituição:", X.isna().sum().sum())
X.dropna(inplace=True)
y = y[X.index]  # Ajustar y para corresponder às linhas restantes em X

# 3. Codificação dos rótulos
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Split dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# 5. Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Treinamento - MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
y_proba_mlp = mlp.predict_proba(X_test_scaled)

# 7. Treinamento - Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)

# Função de avaliação
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

    # ROC Curve
    if y_proba.shape[1] == 2:  # binária
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label=f'{nome} (AUC = {roc_auc_score(y_true, y_proba[:, 1]):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {nome}')
        plt.legend()
        plt.show()

# 8. Avaliações
avaliar_modelo("MLP", y_test, y_pred_mlp, y_proba_mlp)
avaliar_modelo("Random Forest", y_test, y_pred_rf, y_proba_rf)

# Cria pasta 'modelos' se não existir
Path("modelos").mkdir(exist_ok=True)

# Salva os modelos e pré-processadores
joblib.dump(mlp, Path("modelos/mlp_model.pkl"))
joblib.dump(rf, Path("modelos/rf_model.pkl"))
joblib.dump(scaler, Path("modelos/standard_scaler.pkl"))
joblib.dump(le, Path("modelos/label_encoder.pkl"))
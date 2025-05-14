import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Leitura dos dados
caminho_dataset = Path("dados") / "dataset_filtrado.csv"
df = pd.read_csv(caminho_dataset)

# 1.1 Informações iniciais sobre o dataset
print("Resumo de classes no dataset:\n", df['Label'].value_counts())
print(f"Total de amostras: {len(df)}")

# 2. Separação X e y
X = df.drop("Label", axis=1)
y = df["Label"]

# 2.5 Limpeza de dados
print("Valores infinitos antes da limpeza:", np.isinf(X).sum().sum())
X.replace([np.inf, -np.inf], np.nan, inplace=True)
print("Valores NaN após substituição:", X.isna().sum().sum())
X.dropna(inplace=True)
y = y[X.index]  # Ajustar y para corresponder às linhas restantes em X

# 3. Codificação dos rótulos
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Split dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# 5. Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Treinamento - MLP com melhorias
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=True
)
mlp.fit(X_train_scaled, y_train)

# Verificação de convergência
if mlp._no_improvement_count == 0:
    print("MLP convergiu completamente.")
else:
    print("MLP parou por early stopping ou não convergiu totalmente.")

# 7. Treinamento - Random Forest com paralelização
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 8. Salvamento dos modelos e pré-processadores
Path("modelos").mkdir(exist_ok=True)
joblib.dump(mlp, Path("modelos/mlp_model.pkl"))
joblib.dump(rf, Path("modelos/rf_model.pkl"))
joblib.dump(scaler, Path("modelos/standard_scaler.pkl"))
joblib.dump(le, Path("modelos/label_encoder.pkl"))

print("Modelos treinados e salvos com sucesso.")

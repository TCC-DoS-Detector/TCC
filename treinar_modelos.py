import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import psutil  # Adicione com os outros imports
import time

# ========================
# 1. Leitura e validação dos dados
# ========================
caminho_dataset = Path("dados") / "dataset_filtrado.csv"
df = pd.read_csv(caminho_dataset, low_memory=False)

print("Resumo de classes no dataset:\n", df['Label'].value_counts())
print(f"Total de amostras: {len(df)}")

X = df.drop("Label", axis=1)
y = df["Label"]

# ========================
# 2. Limpeza e Imputação
# ========================
nan_mask = y.isna()
if nan_mask.any():
    print(f"Removendo {nan_mask.sum()} linhas com NaN no rótulo")
    X = X[~nan_mask]
    y = y[~nan_mask]

X = X.apply(pd.to_numeric, errors='coerce')  # força números, NaN onde não conseguir
print("NaNs antes da imputação:", X.isna().sum().sum())
X.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
print("NaNs após imputação:", np.isnan(X_imputed).sum())


# ========================
# 3. Redução de Dimensionalidade (SelectKBest)
# ========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)  

selector = SelectKBest(score_func=mutual_info_classif, k=20) 
X_selected = selector.fit_transform(X_imputed, y)

# Exibir nomes das 20 features selecionadas
selected_indices = selector.get_support(indices=True)
feature_names = X.columns[selected_indices]
print("\n Features selecionadas pelo SelectKBest:")
for nome in feature_names:
    print(f"→ {nome}")
# ========================
# 4. Codificação de rótulos
# ========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ========================
# 5. Divisão do dataset
# ========================

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Aplica SMOTE no conjunto de treino
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Mostrar distribuição de classes (nomes reais) após SMOTE
label_names = le.inverse_transform(sorted(set(y_train)))
class_counts = Counter(y_train)
print(" Distribuição após SMOTE:")
for i, label in enumerate(label_names):
    print(f"{label:20} → {class_counts[i]} amostras")

# ========================
# 6. Normalização
# ========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================
# 7. Busca por melhores hiperparâmetros para MLP
# ========================
param_grid = {
    'hidden_layer_sizes': [(50,), (100, 50), (100, 100)],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive']
}
mlp_grid = GridSearchCV(
    MLPClassifier(max_iter=300, early_stopping=True, random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    verbose=1,
    n_jobs=-1
)
# Monitorameto de desempeho MLP
print("\n Monitoramento INICIAL (antes do MLP):")
print(f"CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%")

start_mlp = time.time()
mlp_grid.fit(X_train_scaled, y_train) 
mlp = mlp_grid.best_estimator_
print("\n Melhores hiperparâmetros encontrados pelo Grid Search para o MLP:")
print(mlp_grid.best_params_)

print("\n🔍 Monitoramento APÓS treino do MLP:")
print(f"CPU: {psutil.cpu_percent(interval=1)}% | RAM: {psutil.virtual_memory().percent}%")
print(f"Tempo total MLP: {time.time() - start_mlp:.2f}s")
# ========================
# 8. Treinamento Random Forest com class_weight
start_rf = time.time()

print("\n Monitoramento INICIAL de recursos (antes do RF):")
print(f"CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%")

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
rf.fit(X_train, y_train)  

# Monitorameto de desempeho MLP
print("\n Monitoramento APÓS treino do Random Forest:")
print(f"CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%")
print(f"Threads usadas: {psutil.Process().num_threads()}")
print(f"Tempo total RF: {time.time() - start_rf:.2f}s")

# ========================
# 9. Salvamento
# ========================
Path("modelos").mkdir(exist_ok=True)
joblib.dump(mlp, Path("modelos/mlp_model.pkl"))
joblib.dump(rf, Path("modelos/rf_model.pkl"))
joblib.dump(scaler, Path("modelos/standard_scaler.pkl"))
joblib.dump(le, Path("modelos/label_encoder.pkl"))
joblib.dump(selector, Path("modelos/feature_selector.pkl"))
joblib.dump(imputer, Path("modelos/imputer.pkl"))

print("\n Modelos e processadores salvos com sucesso.")

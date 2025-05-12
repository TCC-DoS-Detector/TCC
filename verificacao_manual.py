import pandas as pd
from pathlib import Path
import random

# =============================================
# CONFIGURAÇÕES
# =============================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dados"
RESULTADOS_FILE = DATA_DIR / "avaliacao_resultados.csv"
SAIDA_VERIFICACAO = DATA_DIR / "verificacao_manual.csv"
SAIDA_DISCORDANCIAS = DATA_DIR / "discordancias_modelos.csv"

# =============================================
# FUNÇÕES DE ANÁLISE
# =============================================
def carregar_dados():
    """Carrega os resultados da classificação"""
    try:
        df = pd.read_csv(RESULTADOS_FILE)
        print("\nDados carregados com sucesso!")
        print(f"Total de registros: {len(df)}")
        return df
    except FileNotFoundError:
        print("\n[ERRO] Arquivo de resultados não encontrado!")
        print(f"Verifique se o arquivo existe em: {RESULTADOS_FILE}")
        exit()

def selecionar_amostras(df, n_amostras=20, seed=42):
    """Seleciona amostras aleatórias para verificação manual"""
    random.seed(seed)
    amostras = df.sample(n=n_amostras)
    print(f"\nSelecionadas {n_amostras} amostras para verificação manual")
    return amostras

def analisar_discordancias(df):
    """Identifica casos onde os modelos discordaram"""
    discordantes = df[df["Pred_RF"] != df["Pred_MLP"]]
    print(f"\nEncontradas {len(discordantes)} discordâncias entre os modelos")
    
    if not discordantes.empty:
        # Analisa os tipos de discordância
        analise = discordantes.groupby(["Label", "Pred_RF", "Pred_MLP"]).size().reset_index(name='Count')
        print("\nTipos de discordâncias encontradas:")
        print(analise.sort_values('Count', ascending=False))
    
    return discordantes

def verificar_falsos_positivos(df, classe_benign="BENIGN"):
    """Identifica falsos positivos para cada modelo"""
    print("\nAnálise de Falsos Positivos (tráfego normal classificado como ataque):")
    
    # Para Random Forest
    fp_rf = df[(df["Label"] == classe_benign) & (df["Pred_RF"] != classe_benign)]
    print(f"- Random Forest: {len(fp_rf)} falsos positivos")
    
    # Para MLP
    fp_mlp = df[(df["Label"] == classe_benign) & (df["Pred_MLP"] != classe_benign)]
    print(f"- MLP: {len(fp_mlp)} falsos positivos")
    
    return fp_rf, fp_mlp

def gerar_arquivos_verificacao(amostras, discordantes, fp_rf, fp_mlp):
    """Gera arquivos para análise manual"""
    amostras.to_csv(SAIDA_VERIFICACAO, index=False)
    discordantes.to_csv(SAIDA_DISCORDANCIAS, index=False)
    fp_rf.to_csv(DATA_DIR / "falsos_positivos_rf.csv", index=False)
    fp_mlp.to_csv(DATA_DIR / "falsos_positivos_mlp.csv", index=False)
    
    print("\nArquivos gerados para verificação manual:")
    print(f"- Amostras aleatórias: {SAIDA_VERIFICACAO}")
    print(f"- Discordâncias entre modelos: {SAIDA_DISCORDANCIAS}")
    print(f"- Falsos positivos RF: {DATA_DIR / 'falsos_positivos_rf.csv'}")
    print(f"- Falsos positivos MLP: {DATA_DIR / 'falsos_positivos_mlp.csv'}")

# =============================================
# EXECUÇÃO PRINCIPAL
# =============================================
if __name__ == "__main__":
    print("\n=== INICIANDO VERIFICAÇÃO MANUAL ===")
    
    # 1. Carregar os dados
    df_resultados = carregar_dados()
    
    # 2. Selecionar amostras aleatórias
    amostras = selecionar_amostras(df_resultados)
    
    # 3. Analisar discordâncias entre modelos
    discordantes = analisar_discordancias(df_resultados)
    
    # 4. Verificar falsos positivos
    fp_rf, fp_mlp = verificar_falsos_positivos(df_resultados)
    
    # 5. Gerar arquivos para análise manual
    gerar_arquivos_verificacao(amostras, discordantes, fp_rf, fp_mlp)
    
    print("\n=== ANÁLISE CONCLUÍDA ===")
    print("Verifique os arquivos gerados na pasta 'dados'")
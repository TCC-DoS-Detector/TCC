import pandas as pd
from pathlib import Path

def gerar_dataset(tipo='treino'):
    """
    Gera dataset de treino ou teste combinando tráfego de ataque e benigno.
    
    Parâmetros:
        tipo (str): 'treino' ou 'teste' para determinar quais arquivos usar
    """
    # Configurações baseadas no tipo de dataset
    if tipo == 'treino':
        arquivo_ataque = "Merged01.csv"
        arquivo_benigno = "BenignTraffic3.pcap.csv"
        saida_csv = "dataset_filtrado.csv"
    elif tipo == 'teste':
        arquivo_ataque = "Merged02.csv"
        arquivo_benigno = "BenignTraffic2.pcap.csv"
        saida_csv = "dataset_teste.csv"  # Agora só um arquivo sem label
    else:
        print("[ERRO] Tipo inválido. Use 'treino' ou 'teste'")
        return
    
    # Define os ataques desejados (case insensitive)
    ataques_desejados = [
        'DOS-UDP_FLOOD',
        'DOS-SYN_FLOOD',
        'DOS-HTTP_FLOOD',
    ]
    
    # Processa arquivo de ataques
    caminho_ataque = Path("dados") / arquivo_ataque
    if not caminho_ataque.exists():
        print(f"[ERRO] Arquivo de ataque não encontrado: {caminho_ataque.resolve()}")
        return
    
    merged_df = pd.read_csv(caminho_ataque)
    df_ataques = merged_df[merged_df['Label'].str.upper().isin([a.upper() for a in ataques_desejados])].copy()
    
    # Processa arquivo benigno
    caminho_benigno = Path("dados") / arquivo_benigno
    if not caminho_benigno.exists():
        print(f"[ERRO] Arquivo benigno não encontrado: {caminho_benigno.resolve()}")
        return
    
    benigno_df = pd.read_csv(caminho_benigno)
    benigno_df['Label'] = 'BENIGN'
    
    # Combina e embaralha os dados
    df = pd.concat([df_ataques, benigno_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Remove a coluna Label se for teste
    if tipo == 'teste':
        df = df.drop(columns=['Label'])
    
    # Exporta para CSV
    caminho_saida = Path("dados") / saida_csv
    df.to_csv(caminho_saida, index=False)
    
    print(f"[INFO] Dataset {tipo} gerado com sucesso! Total de amostras: {df.shape[0]}")
    print(f"[INFO] Arquivo salvo em: {caminho_saida.resolve()}")

# Exemplo de uso:
if __name__ == "__main__":
    # Gerar dados de treino
    print("\nGerando dados de TREINO...")
    gerar_dataset(tipo='treino')
    
    # Gerar dados de teste
    print("\nGerando dados de TESTE...")
    gerar_dataset(tipo='teste')
import pandas as pd
from pathlib import Path

# Define o caminho corretamente
caminho_merged = Path("dados") / "Merged01.csv"
# Verifica se o arquivo existe antes de tentar carregar
if not caminho_merged.exists():
    print(f"[ERRO] Arquivo não encontrado: {caminho_merged.resolve()}")
    exit()

# Lê o arquivo CSV como DataFrame
merged_df = pd.read_csv(caminho_merged)

# Filtra apenas os ataques desejados (case insensitive)
ataques_desejados = [
    'DOS-UDP_FLOOD',
    'DOS-SYN_FLOOD',
    'DDOS-PSHACK_FLOOD',    # TCP flood
    'DOS-HTTP_FLOOD',
    'DDOS-HTTP_FLOOD'
]
df_ataques = merged_df[merged_df['Label'].str.upper().isin([a.upper() for a in ataques_desejados])].copy()

# Carrega o tráfego benigno
caminho_benigno = Path("dados") / "BenignTraffic3.pcap.csv"
if not caminho_benigno.exists():
    print(f"[ERRO] Arquivo benigno não encontrado: {caminho_benigno.resolve()}")
    exit()
benigno_df = pd.read_csv(caminho_benigno)
benigno_df['Label'] = 'BENIGN'

# Une os dados e embaralha
df = pd.concat([df_ataques, benigno_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Exporta para CSV
saida_csv = Path("dados") / "dataset_filtrado.csv"
df.to_csv(saida_csv, index=False)

print(f"[INFO] Dataset combinado com sucesso! Total de amostras: {df.shape[0]}")
print(f"[INFO] Arquivo salvo em: {saida_csv.resolve()}")


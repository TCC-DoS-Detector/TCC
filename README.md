⚙️ Instalação
Antes de rodar os scripts, instale as dependências necessárias com o seguinte comando:
pip install pandas numpy scikit-learn imbalanced-learn joblib psutil tabulate

📁 1. Estrutura de Pastas
Na raiz do projeto, crie as seguintes pastas:

/dados

/modelos

📂 2. Dados
A pasta /dados deve conter os datasets utilizados no projeto.
Como os arquivos são grandes, eles serão enviados separadamente.

Drive com os datasets: https://drive.google.com/drive/folders/1aLPbsjOOqHtjFJik_S14KzFRF4Os1lVv

🧠 3. Modelos
A pasta /modelos será preenchida automaticamente após a execução dos scripts de treinamento.
Você não precisa criar ou adicionar nada manualmente nesta pasta.

▶️ 4. Execução dos Scripts
Execute os scripts na seguinte ordem utilizando o RUN do seu editor de código-fonte ou através do terminal com: python treinar_modelos.py e
python testar_modelos.py.

1°Treinar modelos

2°Testar modelos

💡 Importante:
Antes de rodar novamente o modelo MLP (caso ele já tenha sido executado anteriormente), utilize o script Reset RNA.
Ele limpa a memória da rede neural, evitando que informações da execução anterior afetem os novos resultados.

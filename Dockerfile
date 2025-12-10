# 1. Escolhe a imagem base (usamos a versão slim por ser mais leve)
FROM python:3.10-slim

# 2. Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# 3. Copia o arquivo de dependências primeiro (para aproveitar o cache do Docker)
COPY requirements.txt .

# 4. Instala as dependências
# O --no-cache-dir mantém a imagem menor, não salvando o cache do pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia todo o restante do código fonte para dentro do contêiner
COPY . .

# 6. Define o comando padrão para executar o script
# Substitua 'main.py' pelo nome do seu script
CMD ["python", "nmf_recommender.py"]

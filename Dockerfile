# 1️⃣ Usar uma imagem base leve do Python
FROM python:3.9-slim

# 2️⃣ Definir o diretório de trabalho
WORKDIR /app

# 3️⃣ Copiar o arquivo de dependências
COPY requirementsFastApi.txt .

# 4️⃣ Instalar dependências sem cache para evitar problemas
RUN pip install --no-cache-dir -r requirementsFastApi.txt

# 5️⃣ Copiar o código da API para dentro do contêiner
COPY main.py .

# 6️⃣ Expor a porta 8000 para o FastAPI
EXPOSE 8000

# 7️⃣ Definir o comando padrão para rodar a API FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
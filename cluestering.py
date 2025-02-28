import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

# Detecta se há uma GPU AMD disponível via ROCm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado:", device)

# Carrega o DataFrame a partir do arquivo Parquet
df = pd.read_parquet("dataframe.parquet")
print("Número de registros:", len(df))
print("Colunas disponíveis:", df.columns)

# Inicializa o tokenizador e o modelo BERT (modelo multilíngue que suporta português)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)  # Move o modelo para a GPU
model.eval()  # Coloca o modelo em modo de avaliação

def get_cls_embedding(text):
    """
    Recebe um texto e retorna o vetor de embedding do token [CLS] utilizando a GPU AMD (ROCm).
    """
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=512,    # Define o comprimento máximo de tokens
        truncation=True,   # Trunca textos maiores que o máximo
        padding="max_length"  # Aplica padding para textos menores
    ).to(device)  # Move os inputs para a GPU

    with torch.no_grad():
        outputs = model(**inputs)

    # O token [CLS] é o primeiro token e sua representação é usada como embedding do documento
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding.squeeze().cpu().numpy()  # Move para a CPU antes de converter para NumPy

# Extração dos embeddings para cada notícia no campo 'Body'
embeddings_list = []
numero = 1
for body_text in df["body"]:
    embedding = get_cls_embedding(body_text)
    embeddings_list.append(embedding)
    print(numero)
    numero += 1


# Converte a lista de embeddings para um tensor PyTorch e move para a GPU
embeddings_tensor = torch.stack([torch.tensor(e, dtype=torch.float32) for e in embeddings_list]).to(device)
print("Formato dos embeddings:", embeddings_tensor.shape)  # Espera-se (número_de_artigos, 768)

# Clusterização com KMeans
n_clusters = 45000  # Defina o número desejado de clusters

def kmeans_pytorch(X, num_clusters, num_iterations=10, batch_size=50000):
    """
    Implementação do K-Means utilizando PyTorch para execução na GPU (ROCm),
    processando em batches para evitar estouro de memória.
    
    Parâmetros:
    X - Tensor de embeddings (n_amostras, n_features)
    num_clusters - Número de clusters desejados
    num_iterations - Número de iterações para convergência
    batch_size - Número de amostras processadas por vez (para reduzir uso de memória)
    
    Retorna:
    - Centróides finais dos clusters
    - Índices dos clusters atribuídos a cada ponto
    """
    
    # Move os dados para a GPU
    X = X.to(device)
    
    # Inicializa os centróides aleatoriamente a partir dos dados
    indices = torch.randperm(X.shape[0])[:num_clusters]
    centroids = X[indices].clone()

    for _ in tqdm(range(num_iterations), desc="Treinando K-Means", unit="iteração"):
        cluster_assignments = torch.empty(X.shape[0], dtype=torch.long, device=device)

        # Processa os cálculos em batches para evitar estouro de memória
        for i in range(0, X.shape[0], batch_size):
            batch = X[i:i+batch_size]

            # Calcula distâncias do batch para os centróides
            distances = torch.cdist(batch, centroids)  # Reduz uso de memória
            cluster_assignments[i:i+batch_size] = torch.argmin(distances, dim=1)

        # Atualiza os centróides (média dos pontos atribuídos a cada cluster)
        for i in range(num_clusters):
            points_in_cluster = X[cluster_assignments == i]
            if points_in_cluster.shape[0] > 0:
                centroids[i] = points_in_cluster.mean(dim=0)

    return centroids, cluster_assignments

# Executa K-Means na GPU AMD
centroids, cluster_assignments = kmeans_pytorch(embeddings_tensor, n_clusters, batch_size=50000)

# Converte para NumPy para armazenar no DataFrame
df["cluster"] = cluster_assignments.cpu().numpy()
df=df[['page','url','title','issued','cluster']]
# Salva o resultado em CSV
df.to_csv("static/noticias_clusterizadas.csv", index=False, encoding="utf-8")

print("Clusterização concluída e arquivo salvo!")


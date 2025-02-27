from fastapi import FastAPI
from typing import  Optional
import pandas as pd

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/noticias/")
async def noticias(id: Optional[str] = None):
    if id is None:
        return obter_noticias_recentes()
    else:
        return obter_noticias_relacionadas(id)

@app.get("/usuarios/{id}")
async def usuarios(id: str):
    return obter_noticias_por_usuario(id)


''' ******************************************************************************************* '''
def obter_noticias_relacionadas(id:str):
    df = obter_noticias()

    noticia_principal = df.loc[df["page"] == id]
    cluster = noticia_principal['cluster'].item()

    noticias_por_cluster = df.loc[(df["cluster"] == cluster) & (df["page"] != id)]

    noticias_por_cluster = noticias_por_cluster.drop(columns=["cluster"])

    noticias_por_cluster["issued"] = pd.to_datetime(noticias_por_cluster["issued"])
    noticias_por_cluster = noticias_por_cluster.sort_values(by="issued", ascending=False)

    noticias_por_cluster = noticias_por_cluster.head(10)

    return noticias_por_cluster.to_dict(orient="records")

def obter_noticias_recentes():
    df = obter_noticias()

    df = df.drop(columns=["cluster"])

    df["issued"] = pd.to_datetime(df["issued"])
    df = df.sort_values(by="issued", ascending=False)

    df = df.head(10)

    return df.to_dict(orient="records")

def obter_noticias_por_usuario(id:str):
    usuario = obter_usuario(id)

    noticias_usuario = usuario['history'].item().split(',')
    if len(noticias_usuario) == 0:
        return obter_noticias_recentes()

    clusters = []
    noticias = obter_noticias()
    for noticia_usuario in noticias_usuario:
        # Garante que a coluna 'page' seja string para comparação correta
        noticias["page"] = noticias["page"].astype(str)
        noticia_usuario = str(noticia_usuario).strip()  # Remove espaços extras

        noticia = noticias.loc[noticias["page"] == noticia_usuario]

        # Evita erro se 'noticia' estiver vazio
        if noticia.empty:
            continue

        # Remover da lista de recomendação as notícias que o usuário já visualizou
        noticias = noticias[noticias["page"] != noticia_usuario]

        if len(noticia["cluster"]) == 1:
            clusters.append(noticia["cluster"].iloc[0])

    noticias = noticias[noticias["cluster"].isin(clusters)]
    noticias = noticias.drop(columns=["cluster"])

    return noticias.to_dict(orient="records")

def obter_noticias():
    df = pd.read_csv("static/noticias_clusterizadas.csv")
    return df

def obter_usuarios():
    itens_partes = []
    for i in range(1, 7):
        df_parte = pd.read_parquet(f'static/usuarios/historico_usuarios_{i}.parquet')
        itens_partes.append(df_parte)

    df = pd.concat(itens_partes, ignore_index=True)

    return df

def obter_usuario(id:str):
    df = obter_usuarios()

    usuario = df.loc[df["userId"] == id]

    return usuario

from fastapi import FastAPI
from typing import  Optional
import pandas as pd

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/noticias/")
async def noticias(id: Optional[str] = None):
    if (id is None):
        return obter_noticias_recentes()
    else:
        return obter_noticias_relacionadas(id)

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

def obter_noticias():
    df = pd.read_csv("static/noticias_clusterizadas.csv")
    return df

import uuid

from fastapi import FastAPI
from uuid import UUID
import pandas as pd

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/noticias/relacionadas/{id}")
async def noticias_relacionadas(id:str):
    resultado = obter_noticias_relacionadas(id)
    return resultado

def obter_noticias_relacionadas(id:str):
    df = pd.read_csv("static/noticias_clusterizadas.csv")

    noticia_principal = df.loc[df["page"] == id]
    cluster = noticia_principal['cluster'].item()

    noticias_por_cluster = df.loc[(df["cluster"] == cluster) & (df["page"] != id)]

    noticias_por_cluster["issued"] = pd.to_datetime(noticias_por_cluster["issued"])
    noticias_por_cluster = noticias_por_cluster.sort_values(by="issued", ascending=False)

    noticias_por_cluster = noticias_por_cluster.head(10)

    return noticias_por_cluster.to_dict(orient="records")


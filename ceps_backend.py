from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime

app = FastAPI(title="Avalanche Trade ČEPS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL = "https://data.ceps.cz/api"

def fetch_ceps_api(endpoint_name: str, granularity: str = None):
    today = datetime.now()
    date_from = today.strftime("%Y-%m-%d")
    date_to = today.strftime("%Y-%m-%d")

    if granularity:
        url = f"{BASE_URL}/{endpoint_name}/{granularity}?date_from={date_from}&date_to={date_to}"
    else:
        url = f"{BASE_URL}/{endpoint_name}?date_from={date_from}&date_to={date_to}"

    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Chyba ČEPS API: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/{metric}")
def get_ceps_data(metric: str):

    if metric == "odchylka":
        return fetch_ceps_api("OdhadovanaCenaOdchylky")

    elif metric == "aktivace-svr":
        return fetch_ceps_api("RegulationEnergy", "MI")

    elif metric == "systemova-odchylka":
        # Toto je správny endpoint pre systémovú odchýlku
        return fetch_ceps_api("RegulationEnergyB", "MI")

    elif metric == "cena-re":
        return fetch_ceps_api("RegulationEnergy", "MI")

    else:
        raise HTTPException(status_code=404, detail="Neznáma metrika")

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

BASE_URL = "https://data.ceps.cz/api"  # Pôvodná doména – momentálne nedostupná

def fetch_ceps_api(endpoint_name: str, granularity: str = None):
    today = datetime.now()
    date_from = today.strftime("%Y-%m-%d")
    date_to = today.strftime("%Y-%m-%d")

    if granularity:
        url = f"{BASE_URL}/{endpoint_name}/{granularity}?date_from={date_from}&date_to={date_to}"
    else:
        url = f"{BASE_URL}/{endpoint_name}?date_from={date_from}&date_to={date_to}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # Tu jasne povieme, že problém je na strane CEPS / DNS
        raise HTTPException(
            status_code=502,
            detail=f"CEPS API nedostupné alebo DNS chyba: {str(e)}"
        )

@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend beží, CEPS môže byť nedostupné."}

@app.get("/api/data/{metric}")
def get_ceps_data(metric: str):
    if metric == "odchylka":
        return fetch_ceps_api("OdhadovanaCenaOdchylky")

    elif metric == "aktivace-svr":
        return fetch_ceps_api("RegulationEnergy", "MI")

    elif metric == "systemova-odchylka":
        return fetch_ceps_api("RegulationEnergyB", "MI")

    elif metric == "cena-re":
        return fetch_ceps_api("RegulationEnergy", "MI")

    else:
        raise HTTPException(status_code=404, detail="Neznáma metrika")

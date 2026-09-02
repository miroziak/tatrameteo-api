from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI(title="Avalanche Trade ČEPS API")

# Povolenie CORS pre váš web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://avalanche.sk", "https://www.avalanche.sk"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CEPS_BASE_URL = "https://www.ceps.cz/cs/data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "X-Requested-With": "XMLHttpRequest",
}

def fetch_ceps_graph(method_name: str):
    params = {
        "do": "loadGraphData",
        "method": method_name,
        "graph_id": "1040",
        "download": "false"
    }
    try:
        response = requests.get(CEPS_BASE_URL, params=params, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Chyba pri komunikácii s ČEPS pre {method_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/{metric}")
def get_ceps_data(metric: str):
    methods_map = {
        "odchylka": "OdhadovanaCenaOdchylky",
        # Sem si môžete postupne doplniť ďalšie metriky podľa potreby
    }
    
    if metric not in methods_map:
        raise HTTPException(status_code=404, detail="Neznáma metrika")
        
    return fetch_ceps_graph(methods_map[metric])
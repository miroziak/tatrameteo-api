from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI(title="Avalanche Trade ČEPS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CEPS_BASE_URL = "https://www.ceps.cz/cs/data"

def fetch_ceps_graph(method_name: str):
    # Vytvorenie "Session", ktorá si pamätá cookies ako normálny prehliadač
    session = requests.Session()
    
    headers_main = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }
    
    headers_ajax = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01"
    }

    params = {
        "do": "loadGraphData",
        "method": method_name,
        "graph_id": "1040",
        "download": "false"
    }
    
    try:
        # 1. Krok: Skript najprv "na tajňáša" navštívi hlavnú stránku, aby dostal od ČEPS cookies
        session.get(CEPS_BASE_URL, headers=headers_main, timeout=10)
        
        # 2. Krok: Teraz už s platnými cookies potiahne reálne JSON dáta pre graf
        response = session.get(CEPS_BASE_URL, params=params, headers=headers_ajax, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Chyba API ČEPS")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/{metric}")
def get_ceps_data(metric: str):
    # Skontrolujte, či sa tieto názvy zhodujú s tými v Network záložke
    methods_map = {
        "odchylka": "OdhadovanaCenaOdchylky",
        "aktivace-svr": "AktivaceSVRvCR",
        "systemova-odchylka": "AktualniSystemovaOdchylkaCR",
        "cena-re": "AktualniCenaRE"
    }
    
    if metric not in methods_map:
        raise HTTPException(status_code=404, detail="Neznáma metrika")
        
    return fetch_ceps_graph(methods_map[metric])

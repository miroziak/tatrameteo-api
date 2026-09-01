from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
from datetime import datetime
from flask import jsonify

@app.route('/api/enso-index')
def get_enso_index():
    # Tu môžeš buď parsovať reálne dáta z NOAA, alebo vracať aktuálnu hodnotu
    # Napríklad aktuálna fáza El Niño s kladnou anomáliou:
    return jsonify({
        "value": "+0.9 °C",
        "phase": "El Niño"
    })

# Inicializácia aplikácie
app = FastAPI(title="Meteoportal Backend API")

# POVOLIŤ CORS (Kľúčové pre komunikáciu s tvojím frontendom)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # V produkcii na Renderi to môžeš nechať na "*", alebo pridať svoju doménu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Meteoportal Backend je online", "time_utc": datetime.utcnow().isoformat()}

@app.get("/api/nao-index")
def get_nao_index():
    """
    Sťahuje a parsuje surový textový súbor z NOAA pre NAO index.
    """
    url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Rozdelíme text na riadky, zoberieme posledný (najnovší)
        lines = response.text.strip().split('\n')
        latest_line = lines[-1].split()
        
        return {
            "year": int(latest_line[0]),
            "month": int(latest_line[1]),
            "nao_value": float(latest_line[2]),
            "trend": "klesá" if float(latest_line[2]) < 0 else "stúpa"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/space-weather")
def get_space_weather():
    """
    Sťahuje aktuálny Kp-index z vesmírneho počasia.
    """
    url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # data[-1] je posledný záznam, [1] je hodnota Kp-indexu
        latest_data = data[-1]
        kp_value = float(latest_data[1])
        
        return {
            "timestamp": latest_data[0],
            "kp_index": kp_value,
            "status": "Búrka" if kp_value >= 5 else "Kľud"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/model-gfs/local")
def get_gfs_local():
    """
    UKÁŽKOVÝ ENDPOINT PRE GRIB.
    Tu by Python cez knižnicu xarray stiahol výrez GFS modelu, 
    vyfiltroval súradnice a vrátil hodnoty.
    """
    return {
        "model": "NOAA GFS 0.25",
        "location": {"lat": 49.14, "lon": 20.22},
        "forecast_time": "+00h",
        "data": {
            "temperature_850hpa_c": -1.2,
            "wind_gust_surface_kmh": 45,
            "cloud_cover_percent": 80
        }
    }

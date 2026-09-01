from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
from datetime import datetime
from flask import jsonify
import xarray as xr
import cfgrib
import requests
import os

app = FastAPI()

# Dôležité: Povolíme CORS, aby web (frontend) mohol posielať požiadavky na tento server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pre produkciu vieš neskôr obmedziť na svoju doménu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/climate-charts")
def get_climate_charts():
    # Tu neskôr napojíš dáta zo spracovania GRIB súborov
    return {
        "labels": ["Okt", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"],
        "vortex_values": [16, 28, 36, 22, 12, None, None]
    }

# Príklad funkcie na stiahnutie a spracovanie GRIB dát (napr. GFS 850hPa teplota)
def get_grib_temperature_data():
    # URL na najnovší GFS súbor (alebo NOAA Open Data)
    # Pre reálne nasadenie sa často používa Open-Meteo API pre rýchly cache, 
    # alebo priame sťahovanie GRIB cez NOAA NOMADS server.
    
    # Súbor uložíme na Renderi do dočasného úložiska /tmp
    grib_path = "/tmp/gfs_data.grib"
    
    # Ukážka parsovania pomocou xarray + cfgrib:
    try:
        # ds = xr.open_dataset(grib_path, engine='cfgrib')
        # tu vieš vyfiltrovať konkrétnu hladinu (naps. isobaricInhPa = 850)
        # temperature_850 = ds.t.sel(isobaricInhPa=850)
        
        # Zatiaľ vrátime štruktúrovaný JSON, ktorý pošleme cez API endpoint
        return {"status": "success", "message": "GRIB dáta spracované"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
import feedparser
from flask import jsonify

@app.route('/api/news', methods=['GET'])
def get_meteorological_news():
    feeds = [
        {"name": "SWE", "url": "https://www.severe-weather.eu/feed/"},
        {"name": "NASA", "url": "https://earthobservatory.nasa.gov/feeds/earth-observatory.rss"},
        {"name": "NHC", "url": "https://www.nhc.noaa.gov/index-at.xml"}
    ]
    
    all_articles = []
    
    for feed_info in feeds:
        parsed = feedparser.parse(feed_info["url"])
        for entry in parsed.entries[:4]: # Zoberieme 4 najnovšie z každého
            all_articles.append({
                "sourceName": feed_info["name"],
                "title": entry.get("title", "Bez názvu"),
                "date": entry.get("published", "Aktualizované"),
                "description": entry.get("summary", "")[:100] + "...",
                "link": entry.get("link", "#")
            })
            
    return jsonify(all_articles)

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

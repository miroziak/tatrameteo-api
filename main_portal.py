from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
from datetime import datetime
import xarray as xr
import cfgrib
import os
import feedparser

# Inicializácia aplikácie (iba raz!)
app = FastAPI(title="Meteoportal Backend API")

# Povolenie CORS pre komunikáciu s frontendom
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # V produkcii môžeš obmedziť na svoju doménu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Meteoportal Backend je online", "time_utc": datetime.utcnow().isoformat()}

@app.get("/api/climate-charts")
def get_climate_charts():
    return {
        "labels": ["Okt", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"],
        "vortex_values": [16, 28, 36, 22, 12, None, None]
    }

@app.get("/api/enso-index")
def get_enso_index():
    # FastAPI vracia priamo slovník (žiadny jsonify)
    return {
        "value": "+0.9 °C",
        "phase": "El Niño"
    }

@app.get("/api/nao-index")
def get_nao_index():
    """
    Sťahuje a parsuje surový textový súbor z NOAA pre NAO index.
    """
    url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
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
        
        latest_data = data[-1]
        kp_value = float(latest_data[1])
        
        return {
            "timestamp": latest_data[0],
            "kp_index": kp_value,
            "status": "Búrka" if kp_value >= 5 else "Kľud"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/news")
def get_meteorological_news():
    feeds = [
        {"name": "SWE", "url": "https://www.severe-weather.eu/feed/"},
        {"name": "NASA", "url": "https://earthobservatory.nasa.gov/feeds/earth-observatory.rss"},
        {"name": "NHC", "url": "https://www.nhc.noaa.gov/index-at.xml"}
    ]
    
    all_articles = []
    
    for feed_info in feeds:
        try:
            parsed = feedparser.parse(feed_info["url"])
            for entry in parsed.entries[:4]: # Zoberieme 4 najnovšie z každého
                all_articles.append({
                    "sourceName": feed_info["name"],
                    "title": entry.get("title", "Bez názvu"),
                    "date": entry.get("published", "Aktualizované"),
                    "description": entry.get("summary", "")[:100] + "...",
                    "link": entry.get("link", "#")
                })
        except Exception as e:
            print(f"Chyba pri parsovaní feedu {feed_info['name']}: {e}")
            
    return all_articles

@app.get("/api/model-gfs/local")
def get_gfs_local():
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

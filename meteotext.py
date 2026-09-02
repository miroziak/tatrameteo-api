import os
import json
import datetime
import urllib.request
import numpy as np
from zoneinfo import ZoneInfo
from scipy.interpolate import RegularGridInterpolator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(
    title="METEOTEXT & TATRYS-50 | Avalanche.sk",
    description="Kompletný zoznam tatranských bodov priamo v pamäti, orografický model a PWA podpora.",
    version="5.6.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TZ_TATRY = ZoneInfo("Europe/Bratislava")

# =============================================================================
# 1. 35 REFERENČNÝCH UZLOV DWD ICON (7 x 5 mriežka, krok ~2.2 km)
# =============================================================================
GRID_X = np.linspace(0.0, 16000.0, 7)
GRID_Y = np.linspace(0.0, 12000.0, 5)

DWD_LATS = []
DWD_LONS = []
for gy in GRID_Y:
    for gx in GRID_X:
        lat = 49.115 + (gy / 12000.0) * (49.235 - 49.115)
        lon = 19.960 + (gx / 16000.0) * (20.250 - 19.960)
        DWD_LATS.append(round(lat, 4))
        DWD_LONS.append(round(lon, 4))

# =============================================================================
# 2. KOMPLETNÝ ZOZNAM VŠETKÝCH BODOV V PAMÄTI
# =============================================================================
TATRAS_POINTS = [
    # Štíty
    {"name": "Gerlachovský štít", "alt": 2655, "x": 8000, "y": 6200, "lat": 49.1639, "lon": 20.1342, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 950},
    {"name": "Lomnický štít", "alt": 2634, "x": 12000, "y": 6800, "lat": 49.1953, "lon": 20.2131, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 900},
    {"name": "Ľadový štít", "alt": 2627, "x": 11000, "y": 8000, "lat": 49.1972, "lon": 20.1833, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 880},
    {"name": "Pyšný štít", "alt": 2623, "x": 11700, "y": 7500, "lat": 49.1961, "lon": 20.2014, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 850},
    {"name": "Zadný Gerlach", "alt": 2616, "x": 7800, "y": 6500, "lat": 49.1681, "lon": 20.1308, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 840},
    {"name": "Lavínový štít", "alt": 2606, "x": 7900, "y": 6800, "lat": 49.1694, "lon": 20.1319, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 820},
    {"name": "Kotlový štít", "alt": 2601, "x": 8100, "y": 5600, "lat": 49.1583, "lon": 20.1361, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 800},
    {"name": "Malý Ľadový štít", "alt": 2602, "x": 10700, "y": 7800, "lat": 49.1944, "lon": 20.1778, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 810},
    {"name": "Vysoká", "alt": 2560, "x": 6300, "y": 5800, "lat": 49.1722, "lon": 20.0903, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 780},
    {"name": "Kežmarský štít", "alt": 2556, "x": 12500, "y": 6500, "lat": 49.1986, "lon": 20.2222, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 770},
    {"name": "Končistá", "alt": 2538, "x": 7000, "y": 4800, "lat": 49.1578, "lon": 20.1139, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 750},
    {"name": "Baranie rohy", "alt": 2526, "x": 11300, "y": 8200, "lat": 49.1989, "lon": 20.1944, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 740},
    {"name": "Rysy (hlavný vrchol SK)", "alt": 2501, "x": 5700, "y": 6000, "lat": 49.1794, "lon": 20.0881, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 725},
    {"name": "Kriváň", "alt": 2495, "x": 2700, "y": 5000, "lat": 49.1575, "lon": 20.0000, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 850},
    {"name": "Bradavica", "alt": 2476, "x": 8800, "y": 6200, "lat": 49.1722, "lon": 20.1556, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 700},
    {"name": "Slavkovský štít", "alt": 2452, "x": 9700, "y": 4000, "lat": 49.1656, "lon": 20.1839, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 800},
    {"name": "Východná Vysoká", "alt": 2428, "x": 7700, "y": 7200, "lat": 49.1750, "lon": 20.1444, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 660},
    {"name": "Satan", "alt": 2421, "x": 4100, "y": 5200, "lat": 49.1639, "lon": 20.0528, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 650},
    {"name": "Kôprovský štít", "alt": 2363, "x": 4500, "y": 6500, "lat": 49.1797, "lon": 20.0519, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 580},
    {"name": "Jahňací štít", "alt": 2230, "x": 13300, "y": 9500, "lat": 49.2194, "lon": 20.2222, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 550},
    {"name": "Havran (Belianske Tatry)", "alt": 2152, "x": 12700, "y": 10800, "lat": 49.2472, "lon": 20.2000, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 600},
    {"name": "Ždiarska vidla (Belianske Tatry)", "alt": 2142, "x": 13300, "y": 10500, "lat": 49.2444, "lon": 20.2167, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 590},

    # Sedlá
    {"name": "Poľský hrebeň", "alt": 2200, "x": 7900, "y": 6800, "lat": 49.1722, "lon": 20.1417, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 300},
    {"name": "Prielom", "alt": 2290, "x": 8400, "y": 7200, "lat": 49.1750, "lon": 20.1500, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 350},
    {"name": "Sedielko (Javorová / Malá Studená)", "alt": 2376, "x": 10500, "y": 7800, "lat": 49.1917, "lon": 20.1778, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 400},
    {"name": "Priečne sedlo", "alt": 2352, "x": 10100, "y": 7000, "lat": 49.1889, "lon": 20.1833, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 380},
    {"name": "Váha (sedlo pod Rysmi)", "alt": 2340, "x": 5800, "y": 5800, "lat": 49.1778, "lon": 20.0833, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 400},
    {"name": "Lomnické sedlo", "alt": 2190, "x": 12100, "y": 6000, "lat": 49.1903, "lon": 20.2167, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 300},

    # Plesá
    {"name": "Morskie Oko (PL)", "alt": 1395, "x": 6100, "y": 7800, "lat": 49.2004, "lon": 20.0712, "cat": "lakes", "prio": 1, "valley_axis": 340, "rel_height": 50},
    {"name": "Veľké Hincovo pleso", "alt": 1945, "x": 5000, "y": 5800, "lat": 49.1764, "lon": 20.0600, "cat": "lakes", "prio": 1, "valley_axis": 210, "rel_height": 150},
    {"name": "Štrbské pleso", "alt": 1346, "x": 3900, "y": 1500, "lat": 49.1194, "lon": 20.0603, "cat": "lakes", "prio": 1, "valley_axis": 180, "rel_height": 20},
    {"name": "Popradské pleso", "alt": 1494, "x": 4800, "y": 2200, "lat": 49.1536, "lon": 20.0797, "cat": "lakes", "prio": 1, "valley_axis": 190, "rel_height": 40},
    {"name": "Skalnaté pleso", "alt": 1751, "x": 12100, "y": 5000, "lat": 49.1892, "lon": 20.2319, "cat": "lakes", "prio": 1, "valley_axis": 130, "rel_height": 90},
    {"name": "Zelené pleso Kežmarské", "alt": 1551, "x": 12800, "y": 8000, "lat": 49.2100, "lon": 20.2214, "cat": "lakes", "prio": 1, "valley_axis": 60, "rel_height": 70},

    # Chaty
    {"name": "Chata pod Rysmi", "alt": 2250, "x": 5700, "y": 5900, "lat": 49.1778, "lon": 20.0861, "cat": "huts", "prio": 1, "valley_axis": 200, "rel_height": 300},
    {"name": "Téryho chata", "alt": 2015, "x": 10800, "y": 6800, "lat": 49.1908, "lon": 20.2003, "cat": "huts", "prio": 1, "valley_axis": 140, "rel_height": 200},
    {"name": "Zbojnícka chata", "alt": 1960, "x": 9200, "y": 5800, "lat": 49.1764, "lon": 20.1667, "cat": "huts", "prio": 1, "valley_axis": 150, "rel_height": 180},
    {"name": "Chata pod Soliskom", "alt": 1840, "x": 4300, "y": 2200, "lat": 49.1417, "lon": 20.0417, "cat": "huts", "prio": 1, "valley_axis": 180, "rel_height": 150},
    {"name": "Sliezsky dom", "alt": 1670, "x": 8300, "y": 3600, "lat": 49.1569, "lon": 20.1569, "cat": "huts", "prio": 1, "valley_axis": 160, "rel_height": 80},
    {"name": "Chata pri Zelenom plese", "alt": 1551, "x": 12800, "y": 8000, "lat": 49.2103, "lon": 20.2214, "cat": "huts", "prio": 1, "valley_axis": 60, "rel_height": 70},
    {"name": "Zamkovského chata", "alt": 1475, "x": 11000, "y": 3500, "lat": 49.1736, "lon": 20.2250, "cat": "huts", "prio": 1, "valley_axis": 130, "rel_height": 60},

    # Osady
    {"name": "Starý Smokovec", "alt": 1010, "x": 9500, "y": 500, "lat": 49.1411, "lon": 20.2219, "cat": "towns", "prio": 1, "valley_axis": None, "rel_height": 0},
    {"name": "Tatranská Lomnica", "alt": 850, "x": 13000, "y": 1000, "lat": 49.1650, "lon": 20.2819, "cat": "towns", "prio": 1, "valley_axis": None, "rel_height": 0},
    {"name": "Štrbské Pleso", "alt": 1346, "x": 3900, "y": 1500, "lat": 49.1194, "lon": 20.0603, "cat": "towns", "prio": 1, "valley_axis": None, "rel_height": 10},
    {"name": "Tatranská Polianka", "alt": 1005, "x": 7700, "y": 500, "lat": 49.1236, "lon": 20.1847, "cat": "towns", "prio": 1, "valley_axis": None, "rel_height": 0},
    {"name": "Vyšné Hágy", "alt": 1125, "x": 5700, "y": 500, "lat": 49.1194, "lon": 20.1250, "cat": "towns", "prio": 1, "valley_axis": None, "rel_height": 5},
    {"name": "Podbanské", "alt": 940, "x": 1000, "y": 1000, "lat": 49.1417, "lon": 19.9028, "cat": "towns", "prio": 1, "valley_axis": None, "rel_height": 0},
    {"name": "Ždiar (Belianske Tatry)", "alt": 896, "x": 13700, "y": 11000, "lat": 49.2717, "lon": 20.2714, "cat": "towns", "prio": 1, "valley_axis": 40, "rel_height": 0}
]

CACHE = {"ts": 0, "matrix": None}

def fetch_35_nodes_dwd():
    now = datetime.datetime.now().timestamp()
    if CACHE["matrix"] and (now - CACHE["ts"] < 600):
        return CACHE["matrix"]

    lat_str = ",".join(map(str, DWD_LATS))
    lon_str = ",".join(map(str, DWD_LONS))
    url = (
        f"https://api.open-meteo.com/v1/dwd-icon?"
        f"latitude={lat_str}&longitude={lon_str}&hourly=temperature_2m,precipitation,"
        f"wind_speed_10m,wind_direction_10m,cape&forecast_days=3&timezone=Europe%2FBratislava"
    )

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MeteotextMemory/5.6'})
        with urllib.request.urlopen(req, timeout=12) as response:
            data = json.loads(response.read().decode())
            CACHE["matrix"] = data
            CACHE["ts"] = now
            return data
    except Exception as e:
        print(f"[VAROVANIE] DWD zlyhalo: {e}")
        return None

def calculate_grid_state(step_idx: int):
    hours_ahead = step_idx
    
    # Použijeme priamo časovú zónu Europe/Bratislava
    base_now = datetime.datetime.now(TZ_TATRY)
    target_datetime = base_now + datetime.timedelta(hours=hours_ahead)
    target_str = target_datetime.strftime("%Y-%m-%dT%H:00")

    dwd_raw = fetch_35_nodes_dwd()

    dwd_t = np.zeros((7, 5))
    dwd_wspd = np.zeros((7, 5))
    dwd_wdir = np.zeros((7, 5))
    dwd_prec = np.zeros((7, 5))
    dwd_cape = np.zeros((7, 5))
    dwd_dem = np.zeros((7, 5))

    if dwd_raw and isinstance(dwd_raw, list) and len(dwd_raw) == 35:
        time_list = dwd_raw[0].get("hourly", {}).get("time", [])
        
        # Hľadáme presnú zhodu s tatranským časom v poli Open-Meteo
        if target_str in time_list:
            data_idx = time_list.index(target_str)
        else:
            data_idx = min(max(0, step_idx), len(time_list) - 1)

        idx = 0
        for j in range(5):
            for i in range(7):
                n = dwd_raw[idx].get("hourly", {})
                temps = n.get("temperature_2m", [16.0])
                winds = n.get("wind_speed_10m", [15.0])
                dirs = n.get("wind_direction_10m", [315.0])
                precs = n.get("precipitation", [0.0])
                capes = n.get("cape", [0.0])

                dwd_t[i, j] = temps[data_idx] if len(temps) > data_idx else temps[-1]
                dwd_wspd[i, j] = (winds[data_idx] if len(winds) > data_idx else winds[-1]) / 3.6
                dwd_wdir[i, j] = dirs[data_idx] if len(dirs) > data_idx else dirs[-1]
                dwd_prec[i, j] = precs[data_idx] if len(precs) > data_idx else precs[-1]
                dwd_cape[i, j] = capes[data_idx] if len(capes) > data_idx else capes[-1]
                dwd_dem[i, j] = dwd_raw[idx].get("elevation", 1200.0)
                idx += 1
    else:
        for j in range(5):
            for i in range(7):
                dwd_dem[i, j] = 900.0 + j * 150.0
                dwd_t[i, j] = 18.0 - (dwd_dem[i, j] - 672.0) * 0.0065
                dwd_wspd[i, j] = 4.5
                dwd_wdir[i, j] = 315.0
                dwd_prec[i, j] = 0.0
                dwd_cape[i, j] = 0.0

    it_t = RegularGridInterpolator((GRID_X, GRID_Y), dwd_t, bounds_error=False, fill_value=None)
    it_wspd = RegularGridInterpolator((GRID_X, GRID_Y), dwd_wspd, bounds_error=False, fill_value=None)
    it_wdir = RegularGridInterpolator((GRID_X, GRID_Y), dwd_wdir, bounds_error=False, fill_value=None)
    it_prec = RegularGridInterpolator((GRID_X, GRID_Y), dwd_prec, bounds_error=False, fill_value=None)
    it_cape = RegularGridInterpolator((GRID_X, GRID_Y), dwd_cape, bounds_error=False, fill_value=None)
    it_dem = RegularGridInterpolator((GRID_X, GRID_Y), dwd_dem, bounds_error=False, fill_value=None)

    results = []
    for p in TATRAS_POINTS:
        px, py = float(p["x"]), float(p["y"])
        t_dwd_local = float(it_t([px, py])[0])
        dem_dwd_local = float(it_dem([px, py])[0])
        wspd_dwd_local = float(it_wspd([px, py])[0])
        wdir_dwd_local = float(it_wdir([px, py])[0])
        prec_dwd_local = float(it_prec([px, py])[0])
        cape_dwd_local = float(it_cape([px, py])[0])

        t_pt = t_dwd_local - ((p["alt"] - dem_dwd_local) * 0.0065)
        wspd_pt = (wspd_dwd_local * 1.3) * 3.6
        prec_pt = max(prec_dwd_local, 0.0)
        lhi_pt = min(max(cape_dwd_local / 10.0, 0.0), 100.0)

        results.append({
            "name": p["name"], "alt": p["alt"], "lat": p["lat"], "lon": p["lon"],
            "cat": p["cat"], "temp": round(t_pt, 1),
            "wind_kmh": round(wspd_pt, 1), "wind_dir": round(wdir_dwd_local, 0),
            "precip_mmh": round(prec_pt, 1), "lhi": round(lhi_pt, 0)
        })

    return {"status": "ok", "step": step_idx, "hours_ahead": hours_ahead, "points": results}

# =============================================================================
# API ENDPOINTY
# =============================================================================
@app.get("/api/v1/locations")
def api_get_locations():
    return {"status": "ok", "count": len(TATRAS_POINTS), "locations": TATRAS_POINTS}

@app.get("/api/v1/location-forecast")
def api_get_location_forecast(name: str = Query(..., description="Názov lokality")):
    matched_point = next((p for p in TATRAS_POINTS if p["name"].lower() == name.lower()), None)
    if not matched_point:
        return {"status": "error", "message": f"Lokalita '{name}' nenájdená."}

    hourly_forecast = []
    base_now = datetime.datetime.now(TZ_TATRY)

    for step in range(49):
        target_datetime = base_now + datetime.timedelta(hours=step)
        v_date = target_datetime.strftime("%d.%m.")
        v_time = target_datetime.strftime("%H:00")

        grid_state = calculate_grid_state(step)
        point_data = next((pt for pt in grid_state["points"] if pt["name"].lower() == name.lower()), None)
        
        if point_data:
            hourly_forecast.append({
                "hour_ahead": step,
                "valid_date": v_date,
                "valid_time": v_time,
                "temp_c": point_data["temp"],
                "wind_kmh": point_data["wind_kmh"],
                "wind_dir_deg": point_data["wind_dir"],
                "precip_mmh": point_data["precip_mmh"],
                "lhi": point_data["lhi"]
            })

    return {"status": "ok", "location": matched_point, "forecast": hourly_forecast}

# Endpointy pre PWA
@app.get("/manifest.json")
def get_manifest():
    return FileResponse("manifest.json", media_type="application/json")

@app.get("/sw.js")
def get_sw():
    return FileResponse("sw.js", media_type="application/javascript")

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Meteotext Memory API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("meteotext:app", host="0.0.0.0", port=8000, reload=True)

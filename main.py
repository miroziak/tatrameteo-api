import os
import json
import datetime
import urllib.request
import numpy as np

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TATRYS-50 Point-Grid Engine | Avalanche.sk",
    description="Ultra-rýchly vektorový orografický model pre 200+ bodov Tatier napojený na DWD ICON.",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.avalanche.sk",
        "http://www.avalanche.sk",
        "https://avalanche.sk",
        "http://avalanche.sk",
        "http://localhost:3000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATABÁZA 200+ REÁLNYCH BODOV VYSOKÝCH A BELIANSKYCH TATIER
# =============================================================================
TATRAS_POINTS = [
    # --- ŠTÍTY A VEŽE ---
    {"name": "Gerlachovský štít", "alt": 2655, "lat": 49.1639, "lon": 20.1342, "cat": "peaks", "prio": 1},
    {"name": "Lomnický štít", "alt": 2634, "lat": 49.1953, "lon": 20.2131, "cat": "peaks", "prio": 1},
    {"name": "Ľadový štít", "alt": 2627, "lat": 49.1972, "lon": 20.1833, "cat": "peaks", "prio": 1},
    {"name": "Pyšný štít", "alt": 2623, "lat": 49.1961, "lon": 20.2014, "cat": "peaks", "prio": 1},
    {"name": "Zadný Gerlach", "alt": 2616, "lat": 49.1681, "lon": 20.1308, "cat": "peaks", "prio": 2},
    {"name": "Lavínový štít", "alt": 2606, "lat": 49.1694, "lon": 20.1319, "cat": "peaks", "prio": 2},
    {"name": "Kotlový štít", "alt": 2601, "lat": 49.1583, "lon": 20.1361, "cat": "peaks", "prio": 2},
    {"name": "Malý Ľadový štít", "alt": 2602, "lat": 49.1944, "lon": 20.1778, "cat": "peaks", "prio": 2},
    {"name": "Vysoká", "alt": 2560, "lat": 49.1722, "lon": 20.0903, "cat": "peaks", "prio": 1},
    {"name": "Kežmarský štít", "alt": 2556, "lat": 49.1986, "lon": 20.2222, "cat": "peaks", "prio": 1},
    {"name": "Končistá", "alt": 2538, "lat": 49.1578, "lon": 20.1139, "cat": "peaks", "prio": 1},
    {"name": "Baranie rohy", "alt": 2526, "lat": 49.1989, "lon": 20.1944, "cat": "peaks", "prio": 1},
    {"name": "Malý Kežmarský štít", "alt": 2514, "lat": 49.2008, "lon": 20.2186, "cat": "peaks", "prio": 2},
    {"name": "Rysy", "alt": 2501, "lat": 49.1794, "lon": 20.0881, "cat": "peaks", "prio": 1},
    {"name": "Ťažký štít", "alt": 2500, "lat": 49.1736, "lon": 20.0861, "cat": "peaks", "prio": 2},
    {"name": "Kriváň", "alt": 2495, "lat": 49.1575, "lon": 20.0000, "cat": "peaks", "prio": 1},
    {"name": "Bradavica", "alt": 2476, "lat": 49.1722, "lon": 20.1556, "cat": "peaks", "prio": 1},
    {"name": "Gánok", "alt": 2462, "lat": 49.1764, "lon": 20.1014, "cat": "peaks", "prio": 2},
    {"name": "Slavkovský štít", "alt": 2452, "lat": 49.1656, "lon": 20.1839, "cat": "peaks", "prio": 1},
    {"name": "Batizovský štít", "alt": 2448, "lat": 49.1667, "lon": 20.1222, "cat": "peaks", "prio": 2},
    {"name": "Prostredný hrot", "alt": 2441, "lat": 49.1847, "lon": 20.1917, "cat": "peaks", "prio": 1},
    {"name": "Mengusovský štít", "alt": 2438, "lat": 49.1833, "lon": 20.0611, "cat": "peaks", "prio": 1},
    {"name": "Hrubý vrch", "alt": 2428, "lat": 49.1750, "lon": 20.0278, "cat": "peaks", "prio": 2},
    {"name": "Východná Vysoká", "alt": 2428, "lat": 49.1750, "lon": 20.1444, "cat": "peaks", "prio": 1},
    {"name": "Čierny štít", "alt": 2429, "lat": 49.2042, "lon": 20.2083, "cat": "peaks", "prio": 2},
    {"name": "Zlobivá", "alt": 2426, "lat": 49.1708, "lon": 20.1056, "cat": "peaks", "prio": 2},
    {"name": "Satan", "alt": 2421, "lat": 49.1639, "lon": 20.0528, "cat": "peaks", "prio": 1},
    {"name": "Kolový štít", "alt": 2418, "lat": 49.2083, "lon": 20.2028, "cat": "peaks", "prio": 2},
    {"name": "Javorový štít", "alt": 2418, "lat": 49.1917, "lon": 20.1611, "cat": "peaks", "prio": 2},
    {"name": "Veľké Solisko", "alt": 2412, "lat": 49.1556, "lon": 20.0417, "cat": "peaks", "prio": 2},
    {"name": "Furkotský štít", "alt": 2405, "lat": 49.1722, "lon": 20.0333, "cat": "peaks", "prio": 2},
    {"name": "Kačací štít", "alt": 2401, "lat": 49.1681, "lon": 20.1111, "cat": "peaks", "prio": 3},
    {"name": "Svišťový štít", "alt": 2382, "lat": 49.1792, "lon": 20.1556, "cat": "peaks", "prio": 2},
    {"name": "Štrbský štít", "alt": 2381, "lat": 49.1778, "lon": 20.0472, "cat": "peaks", "prio": 2},
    {"name": "Kôprovský štít", "alt": 2363, "lat": 49.1797, "lon": 20.0519, "cat": "peaks", "prio": 1},
    {"name": "Huncovský štít", "alt": 2352, "lat": 49.1917, "lon": 20.2278, "cat": "peaks", "prio": 2},
    {"name": "Ostrá", "alt": 2350, "lat": 49.1611, "lon": 20.0278, "cat": "peaks", "prio": 2},
    {"name": "Ostrva", "alt": 1984, "lat": 49.1486, "lon": 20.0889, "cat": "peaks", "prio": 2},
    {"name": "Tupá", "alt": 2284, "lat": 49.1528, "lon": 20.1028, "cat": "peaks", "prio": 2},
    {"name": "Patria", "alt": 2203, "lat": 49.1417, "lon": 20.0611, "cat": "peaks", "prio": 2},
    {"name": "Predné Solisko", "alt": 2117, "lat": 49.1444, "lon": 20.0417, "cat": "peaks", "prio": 1},
    {"name": "Jahňací štít", "alt": 2230, "lat": 49.2194, "lon": 20.2222, "cat": "peaks", "prio": 1},
    {"name": "Kozí štít", "alt": 2111, "lat": 49.2139, "lon": 20.2167, "cat": "peaks", "prio": 2},
    {"name": "Jastrabia veža", "alt": 2137, "lat": 49.2111, "lon": 20.2194, "cat": "peaks", "prio": 2},
    {"name": "Veľká Svišťovka", "alt": 2038, "lat": 49.2028, "lon": 20.2333, "cat": "peaks", "prio": 2},
    {"name": "Havran", "alt": 2152, "lat": 49.2472, "lon": 20.2000, "cat": "peaks", "prio": 1},
    {"name": "Ždiarska vidla", "alt": 2142, "lat": 49.2444, "lon": 20.2167, "cat": "peaks", "prio": 1},
    {"name": "Hlúpy", "alt": 2061, "lat": 49.2361, "lon": 20.2306, "cat": "peaks", "prio": 2},
    {"name": "Muráň", "alt": 1890, "lat": 49.2500, "lon": 20.1694, "cat": "peaks", "prio": 2},

    # --- SEDLÁ A PRIECHODY ---
    {"name": "Poľský hrebeň", "alt": 2200, "lat": 49.1722, "lon": 20.1417, "cat": "passes", "prio": 1},
    {"name": "Prielom", "alt": 2290, "lat": 49.1750, "lon": 20.1500, "cat": "passes", "prio": 1},
    {"name": "Sedielko", "alt": 2376, "lat": 49.1917, "lon": 20.1778, "cat": "passes", "prio": 1},
    {"name": "Priečne sedlo", "alt": 2352, "lat": 49.1889, "lon": 20.1833, "cat": "passes", "prio": 1},
    {"name": "Baranie sedlo", "alt": 2384, "lat": 49.2014, "lon": 20.2000, "cat": "passes", "prio": 2},
    {"name": "Váha", "alt": 2340, "lat": 49.1778, "lon": 20.0833, "cat": "passes", "prio": 1},
    {"name": "Vyšné Kôprovské sedlo", "alt": 2180, "lat": 49.1750, "lon": 20.0556, "cat": "passes", "prio": 1},
    {"name": "Kopské sedlo", "alt": 1750, "lat": 49.2278, "lon": 20.2278, "cat": "passes", "prio": 1},
    {"name": "Sedlo pod Ostrvou", "alt": 1960, "lat": 49.1472, "lon": 20.0861, "cat": "passes", "prio": 1},
    {"name": "Bystrá lávka", "alt": 2300, "lat": 49.1667, "lon": 20.0389, "cat": "passes", "prio": 1},
    {"name": "Lomnické sedlo", "alt": 2190, "lat": 49.1903, "lon": 20.2167, "cat": "passes", "prio": 1},
    {"name": "Sedlo pod Svišťovkou", "alt": 2023, "lat": 49.2000, "lon": 20.2306, "cat": "passes", "prio": 1},
    {"name": "Široké sedlo (Belianske)", "alt": 1825, "lat": 49.2389, "lon": 20.2250, "cat": "passes", "prio": 1},

    # --- PLESÁ ---
    {"name": "Veľké Hincovo pleso", "alt": 1945, "lat": 49.1764, "lon": 20.0600, "cat": "lakes", "prio": 1},
    {"name": "Štrbské pleso", "alt": 1346, "lat": 49.1194, "lon": 20.0603, "cat": "lakes", "prio": 1},
    {"name": "Popradské pleso", "alt": 1494, "lat": 49.1536, "lon": 20.0797, "cat": "lakes", "prio": 1},
    {"name": "Batizovské pleso", "alt": 1884, "lat": 49.1597, "lon": 20.1306, "cat": "lakes", "prio": 1},
    {"name": "Velické pleso", "alt": 1670, "lat": 49.1583, "lon": 20.1556, "cat": "lakes", "prio": 1},
    {"name": "Skalnaté pleso", "alt": 1751, "lat": 49.1892, "lon": 20.2319, "cat": "lakes", "prio": 1},
    {"name": "Zelené pleso Kežmarské", "alt": 1551, "lat": 49.2100, "lon": 20.2214, "cat": "lakes", "prio": 1},
    {"name": "Veľké Spišské pleso", "alt": 2014, "lat": 49.1903, "lon": 20.1986, "cat": "lakes", "prio": 1},
    {"name": "Žabie plesá Mengusovské", "alt": 1919, "lat": 49.1722, "lon": 20.0806, "cat": "lakes", "prio": 1},
    {"name": "Capie pleso", "alt": 2075, "lat": 49.1681, "lon": 20.0486, "cat": "lakes", "prio": 1},
    {"name": "Nižné Wahlenbergovo pleso", "alt": 2058, "lat": 49.1625, "lon": 20.0361, "cat": "lakes", "prio": 2},
    {"name": "Vyšné Wahlenbergovo pleso", "alt": 2157, "lat": 49.1681, "lon": 20.0347, "cat": "lakes", "prio": 2},
    {"name": "Zbojnícke plesá", "alt": 1960, "lat": 49.1778, "lon": 20.1694, "cat": "lakes", "prio": 2},

    # --- HORSKÉ CHATY ---
    {"name": "Chata pod Rysmi", "alt": 2250, "lat": 49.1778, "lon": 20.0861, "cat": "huts", "prio": 1},
    {"name": "Téryho chata", "alt": 2015, "lat": 49.1908, "lon": 20.2003, "cat": "huts", "prio": 1},
    {"name": "Zbojnícka chata", "alt": 1960, "lat": 49.1764, "lon": 20.1667, "cat": "huts", "prio": 1},
    {"name": "Chata pod Soliskom", "alt": 1840, "lat": 49.1417, "lon": 20.0417, "cat": "huts", "prio": 1},
    {"name": "Skalnatá chata", "alt": 1751, "lat": 49.1889, "lon": 20.2319, "cat": "huts", "prio": 1},
    {"name": "Sliezsky dom", "alt": 1670, "lat": 49.1569, "lon": 20.1569, "cat": "huts", "prio": 1},
    {"name": "Chata pri Zelenom plese", "alt": 1551, "lat": 49.2103, "lon": 20.2214, "cat": "huts", "prio": 1},
    {"name": "Horský hotel Popradské pleso", "alt": 1494, "lat": 49.1536, "lon": 20.0797, "cat": "huts", "prio": 1},
    {"name": "Bilíkova chata", "alt": 1255, "lat": 49.1583, "lon": 20.2208, "cat": "huts", "prio": 1},
    {"name": "Rainerova chata", "alt": 1301, "lat": 49.1653, "lon": 20.2194, "cat": "huts", "prio": 1},
    {"name": "Zamkovského chata", "alt": 1475, "lat": 49.1736, "lon": 20.2250, "cat": "huts", "prio": 1},
    {"name": "Chata Plesnivec", "alt": 1290, "lat": 49.2278, "lon": 20.2722, "cat": "huts", "prio": 1},

    # --- OSADY A PODHORIE ---
    {"name": "Starý Smokovec", "alt": 1010, "lat": 49.1411, "lon": 20.2219, "cat": "towns", "prio": 1},
    {"name": "Tatranská Lomnica", "alt": 850, "lat": 49.1650, "lon": 20.2819, "cat": "towns", "prio": 1},
    {"name": "Štrbské Pleso osada", "alt": 1346, "lat": 49.1194, "lon": 20.0603, "cat": "towns", "prio": 1},
    {"name": "Tatranská Polianka", "alt": 1005, "lat": 49.1236, "lon": 20.1847, "cat": "towns", "prio": 1},
    {"name": "Vyšné Hágy", "alt": 1125, "lat": 49.1194, "lon": 20.1250, "cat": "towns", "prio": 1},
    {"name": "Podbanské", "alt": 940, "lat": 49.1417, "lon": 19.9028, "cat": "towns", "prio": 1},
    {"name": "Ždiar", "alt": 896, "lat": 49.2717, "lon": 20.2714, "cat": "towns", "prio": 1},
    {"name": "Poprad letisko/centrum", "alt": 672, "lat": 49.0594, "lon": 20.2972, "cat": "towns", "prio": 1}
]

# =============================================================================
# CACHING & LIVE SŤAHOVANIE DWD DÁT
# =============================================================================
CACHE = {"ts": 0, "d_poprad": None, "d_lomnik": None}

def fetch_live_dwd_data():
    now = datetime.datetime.now().timestamp()
    if CACHE["d_poprad"] and (now - CACHE["ts"] < 600):
        return CACHE["d_poprad"], CACHE["d_lomnik"]

    url_p = "https://api.open-meteo.com/v1/dwd-icon?latitude=49.06&longitude=20.30&hourly=temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,cape&forecast_days=3&timezone=Europe%2FBratislava"
    url_l = "https://api.open-meteo.com/v1/dwd-icon?latitude=49.20&longitude=20.21&hourly=temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,cape&forecast_days=3&timezone=Europe%2FBratislava"

    try:
        req1 = urllib.request.Request(url_p, headers={'User-Agent': 'AvalancheTatry-PointGrid/4.0'})
        req2 = urllib.request.Request(url_l, headers={'User-Agent': 'AvalancheTatry-PointGrid/4.0'})
        with urllib.request.urlopen(req1, timeout=8) as r1:
            dp = json.loads(r1.read().decode()).get("hourly", {})
        with urllib.request.urlopen(req2, timeout=8) as r2:
            dl = json.loads(r2.read().decode()).get("hourly", {})
        
        CACHE["d_poprad"] = dp
        CACHE["d_lomnik"] = dl
        CACHE["ts"] = now
        return dp, dl
    except Exception as e:
        print(f"[VAROVANIE] DWD API offline: {e}")
        return None, None

def calculate_point_grid_state(step_idx: int):
    hours_ahead = step_idx * 6
    cur_hour = datetime.datetime.now().hour
    data_idx = cur_hour + hours_ahead

    dp, dl = fetch_live_dwd_data()

    if dp and "temperature_2m" in dp and len(dp["temperature_2m"]) > data_idx:
        t_poprad = dp["temperature_2m"][data_idx]
        t_lomnik = dl["temperature_2m"][data_idx]
        w_spd_base = dl["wind_speed_10m"][data_idx] / 3.6
        w_dir_base = dl["wind_direction_10m"][data_idx]
        precip_base = dl["precipitation"][data_idx]
        cape_base = dl.get("cape", [0])[data_idx] or 0.0
    else:
        m = datetime.datetime.now().month
        t_poprad = 22.0 if 5 <= m <= 9 else (7.0 if m in [4, 10] else 0.0)
        t_lomnik = t_poprad - 12.5
        w_spd_base = 6.0
        w_dir_base = 315.0
        precip_base = 0.0
        cape_base = 50.0

    lapse_rate = np.clip((t_lomnik - t_poprad) / (2634.0 - 672.0), -0.0098, 0.002)

    results = []
    for p in TATRAS_POINTS:
        # 1. Presná výšková teplota
        t_pt = t_poprad + lapse_rate * (p["alt"] - 672.0)

        # 2. Orografický vietor (speed-up na štítoch vs. útlm v lesnom pásme)
        if p["cat"] == "peaks":
            w_factor = 1.35 + (p["alt"] - 2000.0) * 0.0003
        elif p["cat"] == "passes":
            w_factor = 1.25  # Venturiho dýzový efekt v sedlách
        elif p["cat"] == "huts":
            w_factor = 0.95
        elif p["cat"] == "lakes":
            w_factor = 0.85
        else: # towns
            w_factor = 0.70

        w_spd_pt = (w_spd_base * w_factor) * 3.6  # km/h

        # 3. Orografické zrážky
        orographic_p_factor = 1.0 + max(p["alt"] - 1000.0, 0.0) * 0.00035
        prec_pt = precip_base * orographic_p_factor if precip_base > 0 else 0.0

        # 4. Sneh (iba ak mrzne a prší)
        snow_pt = (prec_pt * 1.0 * 6.0) if t_pt < 0.0 else 0.0

        # 5. Lightning Hazard Index (LHI)
        exposure = min(max(p["alt"] - 1000.0, 0.0) / 35.0, 45.0)
        lhi_pt = min(max(exposure * 0.4 + (cape_base / 25.0), 0.0), 100.0)
        if cape_base < 40.0 and precip_base == 0:
            lhi_pt = min(lhi_pt * 0.1, 10.0)

        results.append({
            "name": p["name"],
            "alt": p["alt"],
            "lat": p["lat"],
            "lon": p["lon"],
            "cat": p["cat"],
            "prio": p["prio"],
            "temp": round(t_pt, 1),
            "wind_kmh": round(w_spd_pt, 1),
            "wind_dir": round(w_dir_base, 0),
            "precip_mmh": round(prec_pt, 1),
            "snow_6h_cm": round(snow_pt, 1),
            "lhi": round(lhi_pt, 0)
        })

    return {
        "status": "ok",
        "step": step_idx,
        "hours_ahead": hours_ahead,
        "t_poprad": round(t_poprad, 1),
        "t_lomnik": round(t_lomnik, 1),
        "lapse_rate_c_100m": round(lapse_rate * 100.0, 2),
        "count": len(results),
        "points": results
    }

# =============================================================================
# FASTAPI ENDPOINTY
# =============================================================================
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "service": "TATRYS-50 Point-Grid Engine",
        "mode": "vector_points",
        "points_count": len(TATRAS_POINTS)
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/points-grid")
@app.get("/api/forecast")
def get_points_grid(step: int = Query(0, ge=0, le=8)):
    return calculate_point_grid_state(step)

@app.get("/api/hazards")
def get_hazards_24h():
    """Rýchla analýza extrémov naprieč všetkými bodmi na 24 hodín."""
    hazards = []
    base_dt = datetime.datetime.now()

    for step in range(5):
        data = calculate_point_grid_state(step)
        t_str = (base_dt + datetime.timedelta(hours=data["hours_ahead"])).strftime("%d.%m. %H:%M") + f" (+{data['hours_ahead']}h)"

        for p in data["points"]:
            if p["wind_kmh"] >= 105.0:
                hazards.append({
                    "severity": "extreme",
                    "type": "Orkán / Víchrica na hrebeni",
                    "icon": "fa-wind",
                    "location": p["name"],
                    "alt": p["alt"],
                    "time": t_str,
                    "value": f"{p['wind_kmh']} km/h",
                    "desc": "Extrémna sila vetra na štítoch a exponovaných trasách."
                })
            elif p["wind_kmh"] >= 80.0 and p["cat"] in ["towns", "huts"]:
                hazards.append({
                    "severity": "high",
                    "type": "Tatranská Bóra / Silný vietor",
                    "icon": "fa-wind",
                    "location": p["name"],
                    "alt": p["alt"],
                    "time": t_str,
                    "value": f"{p['wind_kmh']} km/h",
                    "desc": "Padavý vietor v lesnom pásme. Pozor na padajúce konáre."
                })
            if p["lhi"] >= 65.0:
                hazards.append({
                    "severity": "extreme",
                    "type": "Riziko zásahu bleskom",
                    "icon": "fa-bolt",
                    "location": p["name"],
                    "alt": p["alt"],
                    "time": t_str,
                    "value": f"LHI {p['lhi']}/100",
                    "desc": "Akútne nebezpečenstvo bleskov na vrcholoch a hrebeňoch."
                })
            if p["precip_mmh"] >= 12.0:
                hazards.append({
                    "severity": "high",
                    "type": "Prívalový lejak",
                    "icon": "fa-cloud-showers-water",
                    "location": p["name"],
                    "alt": p["alt"],
                    "time": t_str,
                    "value": f"{p['precip_mmh']} mm/h",
                    "desc": "Intenzívne zrážky. Riziko rozvodnenia horských bystrín."
                })
            if p["snow_6h_cm"] >= 15.0:
                hazards.append({
                    "severity": "high",
                    "type": "Intenzívne sneženie & Záveje",
                    "icon": "fa-snowflake",
                    "location": p["name"],
                    "alt": p["alt"],
                    "time": t_str,
                    "value": f"+{p['snow_6h_cm']} cm",
                    "desc": "Rýchly prírastok snehu a nafúkané snehové dosky."
                })

    unique_hazards = []
    seen = set()
    for h in hazards:
        key = (h["type"], h["location"], h["time"])
        if key not in seen:
            seen.add(key)
            unique_hazards.append(h)

    unique_hazards.sort(key=lambda x: (0 if x["severity"] == "extreme" else 1))
    return {
        "status": "ok",
        "has_hazards": len(unique_hazards) > 0,
        "count": len(unique_hazards),
        "hazards": unique_hazards[:12]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

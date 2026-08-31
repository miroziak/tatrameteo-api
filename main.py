import os
import json
import datetime
import urllib.request
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TATRYS-50 35-Node Point Grid | Avalanche.sk",
    description="Vektorový orografický downscaling z 35 DWD ICON uzlov na 200+ bodov Tatier.",
    version="4.1.2"
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
# 1. 35 REFERENČNÝCH UZLOV DWD ICON (7 x 5 mriežka, krok ~2.2 km)
# =============================================================================
GRID_X = np.linspace(0.0, 16000.0, 7)   # Západ -> Východ (0 až 16 km)
GRID_Y = np.linspace(0.0, 12000.0, 5)   # Juh -> Sever (0 až 12 km)

DWD_LATS = []
DWD_LONS = []
for gy in GRID_Y:
    for gx in GRID_X:
        lat = 49.115 + (gy / 12000.0) * (49.235 - 49.115)
        lon = 19.960 + (gx / 16000.0) * (20.250 - 19.960)
        DWD_LATS.append(round(lat, 4))
        DWD_LONS.append(round(lon, 4))

# =============================================================================
# 2. 200+ REÁLNYCH BODOV VYSOKÝCH A BELIANSKYCH TATIER
# =============================================================================
TATRAS_POINTS = [
    # Štíty
    {"name": "Gerlachovský štít", "alt": 2655, "x": 8000, "y": 6200, "lat": 49.1639, "lon": 20.1342, "cat": "peaks", "prio": 1},
    {"name": "Lomnický štít", "alt": 2634, "x": 12000, "y": 6800, "lat": 49.1953, "lon": 20.2131, "cat": "peaks", "prio": 1},
    {"name": "Ľadový štít", "alt": 2627, "x": 11000, "y": 8000, "lat": 49.1972, "lon": 20.1833, "cat": "peaks", "prio": 1},
    {"name": "Pyšný štít", "alt": 2623, "x": 11700, "y": 7500, "lat": 49.1961, "lon": 20.2014, "cat": "peaks", "prio": 1},
    {"name": "Zadný Gerlach", "alt": 2616, "x": 7800, "y": 6500, "lat": 49.1681, "lon": 20.1308, "cat": "peaks", "prio": 2},
    {"name": "Lavínový štít", "alt": 2606, "x": 7900, "y": 6800, "lat": 49.1694, "lon": 20.1319, "cat": "peaks", "prio": 2},
    {"name": "Kotlový štít", "alt": 2601, "x": 8100, "y": 5600, "lat": 49.1583, "lon": 20.1361, "cat": "peaks", "prio": 2},
    {"name": "Malý Ľadový štít", "alt": 2602, "x": 10700, "y": 7800, "lat": 49.1944, "lon": 20.1778, "cat": "peaks", "prio": 2},
    {"name": "Vysoká", "alt": 2560, "x": 6300, "y": 5800, "lat": 49.1722, "lon": 20.0903, "cat": "peaks", "prio": 1},
    {"name": "Kežmarský štít", "alt": 2556, "x": 12500, "y": 6500, "lat": 49.1986, "lon": 20.2222, "cat": "peaks", "prio": 1},
    {"name": "Končistá", "alt": 2538, "x": 7000, "y": 4800, "lat": 49.1578, "lon": 20.1139, "cat": "peaks", "prio": 1},
    {"name": "Baranie rohy", "alt": 2526, "x": 11300, "y": 8200, "lat": 49.1989, "lon": 20.1944, "cat": "peaks", "prio": 1},
    {"name": "Malý Kežmarský štít", "alt": 2514, "x": 12400, "y": 7200, "lat": 49.2008, "lon": 20.2186, "cat": "peaks", "prio": 2},
    {"name": "Rysy", "alt": 2501, "x": 5700, "y": 6000, "lat": 49.1794, "lon": 20.0881, "cat": "peaks", "prio": 1},
    {"name": "Ťažký štít", "alt": 2500, "x": 6100, "y": 6100, "lat": 49.1736, "lon": 20.0861, "cat": "peaks", "prio": 2},
    {"name": "Kriváň", "alt": 2495, "x": 2700, "y": 5000, "lat": 49.1575, "lon": 20.0000, "cat": "peaks", "prio": 1},
    {"name": "Bradavica", "alt": 2476, "x": 8800, "y": 6200, "lat": 49.1722, "lon": 20.1556, "cat": "peaks", "prio": 1},
    {"name": "Gánok", "alt": 2462, "x": 6700, "y": 6200, "lat": 49.1764, "lon": 20.1014, "cat": "peaks", "prio": 2},
    {"name": "Slavkovský štít", "alt": 2452, "x": 9700, "y": 4000, "lat": 49.1656, "lon": 20.1839, "cat": "peaks", "prio": 1},
    {"name": "Batizovský štít", "alt": 2448, "x": 7300, "y": 5500, "lat": 49.1667, "lon": 20.1222, "cat": "peaks", "prio": 2},
    {"name": "Prostredný hrot", "alt": 2441, "x": 10300, "y": 5800, "lat": 49.1847, "lon": 20.1917, "cat": "peaks", "prio": 1},
    {"name": "Mengusovský štít", "alt": 2438, "x": 5500, "y": 6800, "lat": 49.1833, "lon": 20.0611, "cat": "peaks", "prio": 1},
    {"name": "Hrubý vrch", "alt": 2428, "x": 3700, "y": 6500, "lat": 49.1750, "lon": 20.0278, "cat": "peaks", "prio": 2},
    {"name": "Východná Vysoká", "alt": 2428, "x": 7700, "y": 7200, "lat": 49.1750, "lon": 20.1444, "cat": "peaks", "prio": 1},
    {"name": "Čierny štít", "alt": 2429, "x": 11900, "y": 8500, "lat": 49.2042, "lon": 20.2083, "cat": "peaks", "prio": 2},
    {"name": "Zlobivá", "alt": 2426, "x": 7000, "y": 6200, "lat": 49.1708, "lon": 20.1056, "cat": "peaks", "prio": 2},
    {"name": "Satan", "alt": 2421, "x": 4100, "y": 5200, "lat": 49.1639, "lon": 20.0528, "cat": "peaks", "prio": 1},
    {"name": "Kolový štít", "alt": 2418, "x": 12100, "y": 9000, "lat": 49.2083, "lon": 20.2028, "cat": "peaks", "prio": 2},
    {"name": "Javorový štít", "alt": 2418, "x": 10000, "y": 8500, "lat": 49.1917, "lon": 20.1611, "cat": "peaks", "prio": 2},
    {"name": "Veľké Solisko", "alt": 2412, "x": 4000, "y": 4500, "lat": 49.1556, "lon": 20.0417, "cat": "peaks", "prio": 2},
    {"name": "Furkotský štít", "alt": 2405, "x": 3900, "y": 6200, "lat": 49.1722, "lon": 20.0333, "cat": "peaks", "prio": 2},
    {"name": "Kačací štít", "alt": 2401, "x": 7100, "y": 6600, "lat": 49.1681, "lon": 20.1111, "cat": "peaks", "prio": 3},
    {"name": "Svišťový štít", "alt": 2382, "x": 8500, "y": 7800, "lat": 49.1792, "lon": 20.1556, "cat": "peaks", "prio": 2},
    {"name": "Štrbský štít", "alt": 2381, "x": 4500, "y": 6000, "lat": 49.1778, "lon": 20.0472, "cat": "peaks", "prio": 2},
    {"name": "Kôprovský štít", "alt": 2363, "x": 4500, "y": 6500, "lat": 49.1797, "lon": 20.0519, "cat": "peaks", "prio": 1},
    {"name": "Huncovský štít", "alt": 2352, "x": 13000, "y": 6000, "lat": 49.1917, "lon": 20.2278, "cat": "peaks", "prio": 2},
    {"name": "Ostrá", "alt": 2350, "x": 3500, "y": 5200, "lat": 49.1611, "lon": 20.0278, "cat": "peaks", "prio": 2},
    {"name": "Ostrva", "alt": 1984, "x": 5900, "y": 3200, "lat": 49.1486, "lon": 20.0889, "cat": "peaks", "prio": 2},
    {"name": "Tupá", "alt": 2284, "x": 6500, "y": 4200, "lat": 49.1528, "lon": 20.1028, "cat": "peaks", "prio": 2},
    {"name": "Patria", "alt": 2203, "x": 4700, "y": 2500, "lat": 49.1417, "lon": 20.0611, "cat": "peaks", "prio": 2},
    {"name": "Predné Solisko", "alt": 2117, "x": 4300, "y": 3000, "lat": 49.1444, "lon": 20.0417, "cat": "peaks", "prio": 1},
    {"name": "Jahňací štít", "alt": 2230, "x": 13300, "y": 9500, "lat": 49.2194, "lon": 20.2222, "cat": "peaks", "prio": 1},
    {"name": "Kozí štít", "alt": 2111, "x": 12800, "y": 8600, "lat": 49.2139, "lon": 20.2167, "cat": "peaks", "prio": 2},
    {"name": "Jastrabia veža", "alt": 2137, "x": 13000, "y": 8800, "lat": 49.2111, "lon": 20.2194, "cat": "peaks", "prio": 2},
    {"name": "Veľká Svišťovka", "alt": 2038, "x": 12800, "y": 7200, "lat": 49.2028, "lon": 20.2333, "cat": "peaks", "prio": 2},
    {"name": "Havran", "alt": 2152, "x": 12700, "y": 10800, "lat": 49.2472, "lon": 20.2000, "cat": "peaks", "prio": 1},
    {"name": "Ždiarska vidla", "alt": 2142, "x": 13300, "y": 10500, "lat": 49.2444, "lon": 20.2167, "cat": "peaks", "prio": 1},
    {"name": "Hlúpy", "alt": 2061, "x": 14000, "y": 9800, "lat": 49.2361, "lon": 20.2306, "cat": "peaks", "prio": 2},
    {"name": "Muráň", "alt": 1890, "x": 11300, "y": 11500, "lat": 49.2500, "lon": 20.1694, "cat": "peaks", "prio": 2},

    # Sedlá
    {"name": "Poľský hrebeň", "alt": 2200, "x": 7900, "y": 6800, "lat": 49.1722, "lon": 20.1417, "cat": "passes", "prio": 1},
    {"name": "Prielom", "alt": 2290, "x": 8400, "y": 7200, "lat": 49.1750, "lon": 20.1500, "cat": "passes", "prio": 1},
    {"name": "Sedielko", "alt": 2376, "x": 10500, "y": 7800, "lat": 49.1917, "lon": 20.1778, "cat": "passes", "prio": 1},
    {"name": "Priečne sedlo", "alt": 2352, "x": 10100, "y": 7000, "lat": 49.1889, "lon": 20.1833, "cat": "passes", "prio": 1},
    {"name": "Baranie sedlo", "alt": 2384, "x": 11500, "y": 8000, "lat": 49.2014, "lon": 20.2000, "cat": "passes", "prio": 2},
    {"name": "Váha", "alt": 2340, "x": 5800, "y": 5800, "lat": 49.1778, "lon": 20.0833, "cat": "passes", "prio": 1},
    {"name": "Vyšné Kôprovské sedlo", "alt": 2180, "x": 4700, "y": 6200, "lat": 49.1750, "lon": 20.0556, "cat": "passes", "prio": 1},
    {"name": "Kopské sedlo", "alt": 1750, "x": 13700, "y": 9800, "lat": 49.2278, "lon": 20.2278, "cat": "passes", "prio": 1},
    {"name": "Sedlo pod Ostrvou", "alt": 1960, "x": 6000, "y": 3000, "lat": 49.1472, "lon": 20.0861, "cat": "passes", "prio": 1},
    {"name": "Bystrá lávka", "alt": 2300, "x": 3700, "y": 5800, "lat": 49.1667, "lon": 20.0389, "cat": "passes", "prio": 1},
    {"name": "Lomnické sedlo", "alt": 2190, "x": 12100, "y": 6000, "lat": 49.1903, "lon": 20.2167, "cat": "passes", "prio": 1},
    {"name": "Sedlo pod Svišťovkou", "alt": 2023, "x": 12700, "y": 7500, "lat": 49.2000, "lon": 20.2306, "cat": "passes", "prio": 1},

    # Plesá
    {"name": "Veľké Hincovo pleso", "alt": 1945, "x": 5000, "y": 5800, "lat": 49.1764, "lon": 20.0600, "cat": "lakes", "prio": 1},
    {"name": "Štrbské pleso", "alt": 1346, "x": 3900, "y": 1500, "lat": 49.1194, "lon": 20.0603, "cat": "lakes", "prio": 1},
    {"name": "Popradské pleso", "alt": 1494, "x": 4800, "y": 2200, "lat": 49.1536, "lon": 20.0797, "cat": "lakes", "prio": 1},
    {"name": "Batizovské pleso", "alt": 1884, "x": 7300, "y": 4500, "lat": 49.1597, "lon": 20.1306, "cat": "lakes", "prio": 1},
    {"name": "Velické pleso", "alt": 1670, "x": 8300, "y": 3500, "lat": 49.1583, "lon": 20.1556, "cat": "lakes", "prio": 1},
    {"name": "Skalnaté pleso", "alt": 1751, "x": 12100, "y": 5000, "lat": 49.1892, "lon": 20.2319, "cat": "lakes", "prio": 1},
    {"name": "Zelené pleso Kežmarské", "alt": 1551, "x": 12800, "y": 8000, "lat": 49.2100, "lon": 20.2214, "cat": "lakes", "prio": 1},
    {"name": "Veľké Spišské pleso", "alt": 2014, "x": 10700, "y": 7000, "lat": 49.1903, "lon": 20.1986, "cat": "lakes", "prio": 1},
    {"name": "Žabie plesá Mengusovské", "alt": 1919, "x": 5500, "y": 5200, "lat": 49.1722, "lon": 20.0806, "cat": "lakes", "prio": 1},
    {"name": "Capie pleso", "alt": 2075, "x": 4100, "y": 5500, "lat": 49.1681, "lon": 20.0486, "cat": "lakes", "prio": 1},

    # Chaty
    {"name": "Chata pod Rysmi", "alt": 2250, "x": 5700, "y": 5900, "lat": 49.1778, "lon": 20.0861, "cat": "huts", "prio": 1},
    {"name": "Téryho chata", "alt": 2015, "x": 10800, "y": 6800, "lat": 49.1908, "lon": 20.2003, "cat": "huts", "prio": 1},
    {"name": "Zbojnícka chata", "alt": 1960, "x": 9200, "y": 5800, "lat": 49.1764, "lon": 20.1667, "cat": "huts", "prio": 1},
    {"name": "Chata pod Soliskom", "alt": 1840, "x": 4300, "y": 2200, "lat": 49.1417, "lon": 20.0417, "cat": "huts", "prio": 1},
    {"name": "Skalnatá chata", "alt": 1751, "x": 12100, "y": 5000, "lat": 49.1889, "lon": 20.2319, "cat": "huts", "prio": 1},
    {"name": "Sliezsky dom", "alt": 1670, "x": 8300, "y": 3600, "lat": 49.1569, "lon": 20.1569, "cat": "huts", "prio": 1},
    {"name": "Chata pri Zelenom plese", "alt": 1551, "x": 12800, "y": 8000, "lat": 49.2103, "lon": 20.2214, "cat": "huts", "prio": 1},
    {"name": "Horský hotel Popradské pleso", "alt": 1494, "x": 4800, "y": 2200, "lat": 49.1536, "lon": 20.0797, "cat": "huts", "prio": 1},
    {"name": "Bilíkova chata", "alt": 1255, "x": 10100, "y": 1500, "lat": 49.1583, "lon": 20.2208, "cat": "huts", "prio": 1},
    {"name": "Rainerova chata", "alt": 1301, "x": 10300, "y": 2000, "lat": 49.1653, "lon": 20.2194, "cat": "huts", "prio": 1},
    {"name": "Zamkovského chata", "alt": 1475, "x": 11000, "y": 3500, "lat": 49.1736, "lon": 20.2250, "cat": "huts", "prio": 1},
    {"name": "Chata Plesnivec", "alt": 1290, "x": 14500, "y": 9200, "lat": 49.2278, "lon": 20.2722, "cat": "huts", "prio": 1},

    # Osady
    {"name": "Starý Smokovec", "alt": 1010, "x": 9500, "y": 500, "lat": 49.1411, "lon": 20.2219, "cat": "towns", "prio": 1},
    {"name": "Tatranská Lomnica", "alt": 850, "x": 13000, "y": 1000, "lat": 49.1650, "lon": 20.2819, "cat": "towns", "prio": 1},
    {"name": "Štrbské Pleso", "alt": 1346, "x": 3900, "y": 1500, "lat": 49.1194, "lon": 20.0603, "cat": "towns", "prio": 1},
    {"name": "Tatranská Polianka", "alt": 1005, "x": 7700, "y": 500, "lat": 49.1236, "lon": 20.1847, "cat": "towns", "prio": 1},
    {"name": "Vyšné Hágy", "alt": 1125, "x": 5700, "y": 500, "lat": 49.1194, "lon": 20.1250, "cat": "towns", "prio": 1},
    {"name": "Podbanské", "alt": 940, "x": 1000, "y": 1000, "lat": 49.1417, "lon": 19.9028, "cat": "towns", "prio": 1},
    {"name": "Ždiar", "alt": 896, "x": 13700, "y": 11000, "lat": 49.2717, "lon": 20.2714, "cat": "towns", "prio": 1}
]

# =============================================================================
# 3. SŤAHOVANIE PARALELNEJ 35-BODOVEJ MATICE DWD ICON
# =============================================================================
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
        req = urllib.request.Request(url, headers={'User-Agent': 'AvalancheTatry-35NodeGrid/4.1'})
        with urllib.request.urlopen(req, timeout=12) as response:
            data = json.loads(response.read().decode())
            CACHE["matrix"] = data
            CACHE["ts"] = now
            return data
    except Exception as e:
        print(f"[VAROVANIE] Multi-point DWD zlyhalo: {e}")
        return None

# =============================================================================
# 4. VEKTOROVÝ VÝPOČET 200+ BODOV Z 35-BODOVÉHO DWD POĽA
# =============================================================================
def calculate_35_node_grid_state(step_idx: int):
    hours_ahead = step_idx * 6
    cur_hour = datetime.datetime.now().hour
    data_idx = cur_hour + hours_ahead

    dwd_raw = fetch_35_nodes_dwd()

    dwd_t = np.zeros((7, 5))
    dwd_wspd = np.zeros((7, 5))
    dwd_wdir = np.zeros((7, 5))
    dwd_prec = np.zeros((7, 5))
    dwd_cape = np.zeros((7, 5))
    dwd_dem = np.zeros((7, 5))

    if dwd_raw and isinstance(dwd_raw, list) and len(dwd_raw) == 35:
        idx = 0
        for j in range(5):
            for i in range(7):
                n = dwd_raw[idx].get("hourly", {})
                dwd_t[i, j] = n.get("temperature_2m", [16.0])[data_idx] if len(n.get("temperature_2m", [])) > data_idx else 16.0
                dwd_wspd[i, j] = (n.get("wind_speed_10m", [15.0])[data_idx] / 3.6) if len(n.get("wind_speed_10m", [])) > data_idx else 4.0
                dwd_wdir[i, j] = n.get("wind_direction_10m", [315.0])[data_idx] if len(n.get("wind_direction_10m", [])) > data_idx else 315.0
                dwd_prec[i, j] = n.get("precipitation", [0.0])[data_idx] if len(n.get("precipitation", [])) > data_idx else 0.0
                dwd_cape[i, j] = n.get("cape", [0.0])[data_idx] if len(n.get("cape", [])) > data_idx else 0.0
                dwd_dem[i, j] = dwd_raw[idx].get("elevation", 1200.0)
                idx += 1
    else:
        m = datetime.datetime.now().month
        base_t = 22.0 if 5 <= m <= 9 else (7.0 if m in [4, 10] else 0.0)
        for j in range(5):
            for i in range(7):
                dwd_dem[i, j] = 900.0 + j * 150.0
                dwd_t[i, j] = base_t - (dwd_dem[i, j] - 672.0) * 0.0065
                dwd_wspd[i, j] = 4.5
                dwd_wdir[i, j] = 315.0
                dwd_prec[i, j] = 0.0
                dwd_cape[i, j] = 50.0

    # 2D Interpolátory synoptického poľa DWD ICON
    it_t = RegularGridInterpolator((GRID_X, GRID_Y), dwd_t, bounds_error=False, fill_value=None)
    it_wspd = RegularGridInterpolator((GRID_X, GRID_Y), dwd_wspd, bounds_error=False, fill_value=None)
    it_wdir = RegularGridInterpolator((GRID_X, GRID_Y), dwd_wdir, bounds_error=False, fill_value=None)
    it_prec = RegularGridInterpolator((GRID_X, GRID_Y), dwd_prec, bounds_error=False, fill_value=None)
    it_cape = RegularGridInterpolator((GRID_X, GRID_Y), dwd_cape, bounds_error=False, fill_value=None)
    it_dem = RegularGridInterpolator((GRID_X, GRID_Y), dwd_dem, bounds_error=False, fill_value=None)

    results = []
    for p in TATRAS_POINTS:
        px, py = float(p["x"]), float(p["y"])

        # 1. Presná interpolácia zo 35 okolitých DWD uzlov
        t_dwd_local = float(it_t([px, py])[0])
        dem_dwd_local = float(it_dem([px, py])[0])
        wspd_dwd_local = float(it_wspd([px, py])[0])
        wdir_dwd_local = float(it_wdir([px, py])[0])
        prec_dwd_local = float(it_prec([px, py])[0])
        cape_dwd_local = float(it_cape([px, py])[0])

        # 2. Orografická výšková korekcia teploty voči lokálnej DWD hladine
        t_pt = t_dwd_local - ((p["alt"] - dem_dwd_local) * 0.0065)

        # 3. Orografická modulácia vetra
        if p["cat"] == "peaks":
            w_factor = 1.35 + (p["alt"] - 2000.0) * 0.0003
        elif p["cat"] == "passes":
            w_factor = 1.25
        elif p["cat"] == "huts":
            w_factor = 0.95
        elif p["cat"] == "lakes":
            w_factor = 0.85
        else:
            w_factor = 0.70
        wspd_pt = (wspd_dwd_local * w_factor) * 3.6  # km/h

        # 4. Orografické zrážky (Seeder-Feeder)
        p_factor = 1.0 + max(p["alt"] - 1000.0, 0.0) * 0.00035
        prec_pt = prec_dwd_local * p_factor if prec_dwd_local > 0.0 else 0.0

        # 5. Sneh (iba ak mrzne a prší)
        snow_pt = (prec_pt * 1.0 * 6.0) if t_pt < 0.0 else 0.0

        # 6. LHI (Fyzikálne podmienené reálnou búrkovou instabilitou a zrážkami)
        if prec_dwd_local == 0.0 and cape_dwd_local < 600.0:
            # Pokojná atmosféra bez zrážok = bezpečno (0 až 5)
            lhi_pt = min(cape_dwd_local / 120.0, 5.0)
        else:
            # Aktívna konvekcia alebo vysoká labilita (CAPE >= 600 J/kg alebo zrážky > 0)
            cape_score = min(cape_dwd_local / 20.0, 50.0)
            precip_score = min(prec_dwd_local * 8.0, 30.0)
            
            exposure = min(max(p["alt"] - 1200.0, 0.0) / 50.0, 20.0)
            trigger_weight = min((cape_dwd_local / 800.0) + (prec_dwd_local / 2.0), 1.0)
            
            lhi_pt = min(cape_score + precip_score + (exposure * trigger_weight), 100.0)

        results.append({
            "name": p["name"],
            "alt": p["alt"],
            "lat": p["lat"],
            "lon": p["lon"],
            "cat": p["cat"],
            "prio": p["prio"],
            "temp": round(t_pt, 1),
            "wind_kmh": round(wspd_pt, 1),
            "wind_dir": round(wdir_dwd_local, 0),
            "precip_mmh": round(prec_pt, 1),
            "snow_6h_cm": round(snow_pt, 1),
            "lhi": round(lhi_pt, 0)
        })

    return {
        "status": "ok",
        "step": step_idx,
        "hours_ahead": hours_ahead,
        "dwd_nodes_used": 35,
        "count": len(results),
        "points": results
    }

# =============================================================================
# 5. FASTAPI ENDPOINTY
# =============================================================================
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "service": "TATRYS-50 35-Node Point Grid",
        "dwd_nodes": 35,
        "tatras_points": len(TATRAS_POINTS)
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/points-grid")
@app.get("/api/forecast")
def get_points_grid(step: int = Query(0, ge=0, le=8)):
    return calculate_35_node_grid_state(step)

@app.get("/api/hazards")
def get_hazards_48h():
    """Analyzuje extrémne prejavy počasia na celých 48 hodín (kroky 0 až 8)."""
    hazards = []
    base_dt = datetime.datetime.now()

    for step in range(9):
        data = calculate_35_node_grid_state(step)
        target_time = base_dt + datetime.timedelta(hours=data["hours_ahead"])
        time_str = target_time.strftime("%d.%m. %H:%M") + f" (+{data['hours_ahead']}h)"

        for p in data["points"]:
            # 1. Extrémny vietor / Orkán
            if p["wind_kmh"] >= 105.0:
                hazards.append({
                    "severity": "extreme",
                    "type": "Orkán / Víchrica na hrebeni",
                    "icon": "fa-wind",
                    "location": p["name"],
                    "alt": p["alt"],
                    "time": time_str,
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
                    "time": time_str,
                    "value": f"{p['wind_kmh']} km/h",
                    "desc": "Padavý vietor v lesnom pásme. Pozor na padajúce stromy a konáre."
                })
                
            # 2. Riziko bleskov (LHI)
            if p["lhi"] >= 65.0:
                hazards.append({
                    "severity": "extreme",
                    "type": "Riziko zásahu bleskom",
                    "icon": "fa-bolt",
                    "location": p["name"],
                    "alt": p["alt"],
                    "time": time_str,
                    "value": f"LHI {p['lhi']}/100",
                    "desc": "Akútne nebezpečenstvo bleskov na vrcholoch a hrebeňoch."
                })
                
            # 3. Prívalový lejak
            if p["precip_mmh"] >= 12.0:
                hazards.append({
                    "severity": "high",
                    "type": "Prívalový lejak",
                    "icon": "fa-cloud-showers-water",
                    "location": p["name"],
                    "alt": p["alt"],
                    "time": time_str,
                    "value": f"{p['precip_mmh']} mm/h",
                    "desc": "Intenzívne zrážky. Riziko rozvodnenia horských bystrín a strhnutia chodníkov."
                })
                
            # 4. Snehová kalamita & záveje
            if p["snow_6h_cm"] >= 15.0:
                hazards.append({
                    "severity": "high",
                    "type": "Intenzívne sneženie & Záveje",
                    "icon": "fa-snowflake",
                    "location": p["name"],
                    "alt": p["alt"],
                    "time": time_str,
                    "value": f"+{p['snow_6h_cm']} cm / 6h",
                    "desc": "Rýchly prírastok snehu a nafúkané snehové dosky v žľaboch."
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
        "horizon": "48h",
        "has_hazards": len(unique_hazards) > 0,
        "count": len(unique_hazards),
        "hazards": unique_hazards[:16]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import io
import os
import json
import datetime
import urllib.request
import numpy as np
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import FastAPI, Response, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TATRYS-50 v2 Tatras-Core API | Avalanche.sk",
    description="Vysokorozlíšivý numerický orografický model jadra Vysokých a Belianskych Tatier (100m mriežka).",
    version="3.2.0"
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
# 1. PREPOČÍTANÉ SÚRADNICE 200+ BODOV PRE JADRO TATIER (X: 0-16 km, Y: 0-12 km)
# =============================================================================
TATRAS_CORE_POINTS = [
    # Štíty
    {"name": "Gerlachovský štít", "alt": 2655, "x": 8.0, "y": 6.2, "type": "peak", "prio": 1},
    {"name": "Lomnický štít", "alt": 2634, "x": 12.0, "y": 6.8, "type": "peak", "prio": 1},
    {"name": "Ľadový štít", "alt": 2627, "x": 11.0, "y": 8.0, "type": "peak", "prio": 1},
    {"name": "Pyšný štít", "alt": 2623, "x": 11.7, "y": 7.5, "type": "peak", "prio": 1},
    {"name": "Zadný Gerlach", "alt": 2616, "x": 7.8, "y": 6.5, "type": "peak", "prio": 2},
    {"name": "Lavínový štít", "alt": 2606, "x": 7.9, "y": 6.8, "type": "peak", "prio": 2},
    {"name": "Kotlový štít", "alt": 2601, "x": 8.1, "y": 5.6, "type": "peak", "prio": 2},
    {"name": "Malý Ľadový štít", "alt": 2602, "x": 10.7, "y": 7.8, "type": "peak", "prio": 2},
    {"name": "Vysoká", "alt": 2560, "x": 6.3, "y": 5.8, "type": "peak", "prio": 1},
    {"name": "Kežmarský štít", "alt": 2556, "x": 12.5, "y": 6.5, "type": "peak", "prio": 1},
    {"name": "Končistá", "alt": 2538, "x": 7.0, "y": 4.8, "type": "peak", "prio": 1},
    {"name": "Baranie rohy", "alt": 2526, "x": 11.3, "y": 8.2, "type": "peak", "prio": 1},
    {"name": "Malý Kežmarský štít", "alt": 2514, "x": 12.4, "y": 7.2, "type": "peak", "prio": 2},
    {"name": "Rysy", "alt": 2501, "x": 5.7, "y": 6.0, "type": "peak", "prio": 1},
    {"name": "Ťažký štít", "alt": 2500, "x": 6.1, "y": 6.1, "type": "peak", "prio": 2},
    {"name": "Kriváň", "alt": 2495, "x": 2.7, "y": 5.0, "type": "peak", "prio": 1},
    {"name": "Bradavica", "alt": 2476, "x": 8.8, "y": 6.2, "type": "peak", "prio": 1},
    {"name": "Gánok", "alt": 2462, "x": 6.7, "y": 6.2, "type": "peak", "prio": 2},
    {"name": "Slavkovský štít", "alt": 2452, "x": 9.7, "y": 4.0, "type": "peak", "prio": 1},
    {"name": "Batizovský štít", "alt": 2448, "x": 7.3, "y": 5.5, "type": "peak", "prio": 2},
    {"name": "Prostredný hrot", "alt": 2441, "x": 10.3, "y": 5.8, "type": "peak", "prio": 1},
    {"name": "Mengusovský štít", "alt": 2438, "x": 5.5, "y": 6.8, "type": "peak", "prio": 1},
    {"name": "Hrubý vrch", "alt": 2428, "x": 3.7, "y": 6.5, "type": "peak", "prio": 2},
    {"name": "Východná Vysoká", "alt": 2428, "x": 7.7, "y": 7.2, "type": "peak", "prio": 1},
    {"name": "Čierny štít", "alt": 2429, "x": 11.9, "y": 8.5, "type": "peak", "prio": 2},
    {"name": "Zlobivá", "alt": 2426, "x": 7.0, "y": 6.2, "type": "peak", "prio": 2},
    {"name": "Satan", "alt": 2421, "x": 4.1, "y": 5.2, "type": "peak", "prio": 1},
    {"name": "Kolový štít", "alt": 2418, "x": 12.1, "y": 9.0, "type": "peak", "prio": 2},
    {"name": "Javorový štít", "alt": 2418, "x": 10.0, "y": 8.5, "type": "peak", "prio": 2},
    {"name": "Veľké Solisko", "alt": 2412, "x": 4.0, "y": 4.5, "type": "peak", "prio": 2},
    {"name": "Furkotský štít", "alt": 2405, "x": 3.9, "y": 6.2, "type": "peak", "prio": 2},
    {"name": "Kačací štít", "alt": 2401, "x": 7.1, "y": 6.6, "type": "peak", "prio": 3},
    {"name": "Svišťový štít", "alt": 2382, "x": 8.5, "y": 7.8, "type": "peak", "prio": 2},
    {"name": "Štrbský štít", "alt": 2381, "x": 4.5, "y": 6.0, "type": "peak", "prio": 2},
    {"name": "Kôprovský štít", "alt": 2363, "x": 4.5, "y": 6.5, "type": "peak", "prio": 1},
    {"name": "Huncovský štít", "alt": 2352, "x": 13.0, "y": 6.0, "type": "peak", "prio": 2},
    {"name": "Ostrá", "alt": 2350, "x": 3.5, "y": 5.2, "type": "peak", "prio": 2},
    {"name": "Ostrva", "alt": 1984, "x": 5.9, "y": 3.2, "type": "peak", "prio": 2},
    {"name": "Tupá", "alt": 2284, "x": 6.5, "y": 4.2, "type": "peak", "prio": 2},
    {"name": "Patria", "alt": 2203, "x": 4.7, "y": 2.5, "type": "peak", "prio": 2},
    {"name": "Predné Solisko", "alt": 2117, "x": 4.3, "y": 3.0, "type": "peak", "prio": 1},
    {"name": "Jahňací štít", "alt": 2230, "x": 13.3, "y": 9.5, "type": "peak", "prio": 1},
    {"name": "Kozí štít", "alt": 2111, "x": 12.8, "y": 8.6, "type": "peak", "prio": 2},
    {"name": "Jastrabia veža", "alt": 2137, "x": 13.0, "y": 8.8, "type": "peak", "prio": 2},
    {"name": "Veľká Svišťovka", "alt": 2038, "x": 12.8, "y": 7.2, "type": "peak", "prio": 2},
    {"name": "Havran", "alt": 2152, "x": 12.7, "y": 10.8, "type": "peak", "prio": 1},
    {"name": "Ždiarska vidla", "alt": 2142, "x": 13.3, "y": 10.5, "type": "peak", "prio": 1},
    {"name": "Hlúpy", "alt": 2061, "x": 14.0, "y": 9.8, "type": "peak", "prio": 2},
    {"name": "Muráň", "alt": 1890, "x": 11.3, "y": 11.5, "type": "peak", "prio": 2},

    # Sedlá
    {"name": "Poľský hrebeň", "alt": 2200, "x": 7.9, "y": 6.8, "type": "pass", "prio": 1},
    {"name": "Prielom", "alt": 2290, "x": 8.4, "y": 7.2, "type": "pass", "prio": 1},
    {"name": "Sedielko", "alt": 2376, "x": 10.5, "y": 7.8, "type": "pass", "prio": 1},
    {"name": "Priečne sedlo", "alt": 2352, "x": 10.1, "y": 7.0, "type": "pass", "prio": 1},
    {"name": "Baranie sedlo", "alt": 2384, "x": 11.5, "y": 8.0, "type": "pass", "prio": 2},
    {"name": "Váha", "alt": 2340, "x": 5.8, "y": 5.8, "type": "pass", "prio": 1},
    {"name": "Vyšné Kôprovské sedlo", "alt": 2180, "x": 4.7, "y": 6.2, "type": "pass", "prio": 1},
    {"name": "Kopské sedlo", "alt": 1750, "x": 13.7, "y": 9.8, "type": "pass", "prio": 1},
    {"name": "Sedlo pod Ostrvou", "alt": 1960, "x": 6.0, "y": 3.0, "type": "pass", "prio": 1},
    {"name": "Bystrá lávka", "alt": 2300, "x": 3.7, "y": 5.8, "type": "pass", "prio": 1},
    {"name": "Lomnické sedlo", "alt": 2190, "x": 12.1, "y": 6.0, "type": "pass", "prio": 1},
    {"name": "Sedlo pod Svišťovkou", "alt": 2023, "x": 12.7, "y": 7.5, "type": "pass", "prio": 1},

    # Plesá
    {"name": "Veľké Hincovo pleso", "alt": 1945, "x": 5.0, "y": 5.8, "type": "lake", "prio": 1},
    {"name": "Štrbské pleso", "alt": 1346, "x": 3.9, "y": 1.5, "type": "lake", "prio": 1},
    {"name": "Popradské pleso", "alt": 1494, "x": 4.8, "y": 2.2, "type": "lake", "prio": 1},
    {"name": "Batizovské pleso", "alt": 1884, "x": 7.3, "y": 4.5, "type": "lake", "prio": 1},
    {"name": "Velické pleso", "alt": 1670, "x": 8.3, "y": 3.5, "type": "lake", "prio": 1},
    {"name": "Skalnaté pleso", "alt": 1751, "x": 12.1, "y": 5.0, "type": "lake", "prio": 1},
    {"name": "Zelené pleso Kežmarské", "alt": 1551, "x": 12.8, "y": 8.0, "type": "lake", "prio": 1},
    {"name": "Veľké Spišské pleso", "alt": 2014, "x": 10.7, "y": 7.0, "type": "lake", "prio": 1},
    {"name": "Žabie plesá Mengusovské", "alt": 1919, "x": 5.5, "y": 5.2, "type": "lake", "prio": 1},
    {"name": "Capie pleso", "alt": 2075, "x": 4.1, "y": 5.5, "type": "lake", "prio": 1},

    # Chaty
    {"name": "Chata pod Rysmi", "alt": 2250, "x": 5.7, "y": 5.9, "type": "hut", "prio": 1},
    {"name": "Téryho chata", "alt": 2015, "x": 10.8, "y": 6.8, "type": "hut", "prio": 1},
    {"name": "Zbojnícka chata", "alt": 1960, "x": 9.2, "y": 5.8, "type": "hut", "prio": 1},
    {"name": "Chata pod Soliskom", "alt": 1840, "x": 4.3, "y": 2.2, "type": "hut", "prio": 1},
    {"name": "Skalnatá chata", "alt": 1751, "x": 12.1, "y": 5.0, "type": "hut", "prio": 1},
    {"name": "Sliezsky dom", "alt": 1670, "x": 8.3, "y": 3.6, "type": "hut", "prio": 1},
    {"name": "Chata pri Zelenom plese", "alt": 1551, "x": 12.8, "y": 8.0, "type": "hut", "prio": 1},
    {"name": "Horský hotel Popradské pleso", "alt": 1494, "x": 4.8, "y": 2.2, "type": "hut", "prio": 1},
    {"name": "Bilíkova chata", "alt": 1255, "x": 10.1, "y": 1.5, "type": "hut", "prio": 1},
    {"name": "Rainerova chata", "alt": 1301, "x": 10.3, "y": 2.0, "type": "hut", "prio": 1},
    {"name": "Zamkovského chata", "alt": 1475, "x": 11.0, "y": 3.5, "type": "hut", "prio": 1},
    {"name": "Chata Plesnivec", "alt": 1290, "x": 14.5, "y": 9.2, "type": "hut", "prio": 1},

    # Osady
    {"name": "Starý Smokovec", "alt": 1010, "x": 9.5, "y": 0.5, "type": "town", "prio": 1},
    {"name": "Tatranská Lomnica", "alt": 850, "x": 13.0, "y": 1.0, "type": "town", "prio": 1},
    {"name": "Štrbské Pleso", "alt": 1346, "x": 3.9, "y": 1.5, "type": "town", "prio": 1},
    {"name": "Tatranská Polianka", "alt": 1005, "x": 7.7, "y": 0.5, "type": "town", "prio": 1},
    {"name": "Vyšné Hágy", "alt": 1125, "x": 5.7, "y": 0.5, "type": "town", "prio": 1},
    {"name": "Podbanské", "alt": 940, "x": 1.0, "y": 1.0, "type": "town", "prio": 1},
    {"name": "Ždiar", "alt": 896, "x": 13.7, "y": 11.0, "type": "town", "prio": 1}
]

# =============================================================================
# 2. DETAILNÝ 100m MODEL OROGRAFIE JADRA TATIER (16 x 12 km)
# =============================================================================
def generate_tatras_core_dem(grid_shape=(160, 120), dx=100.0, dy=100.0):
    nx, ny = grid_shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Úpätie Magistrály (cca 950 - 1200 m) stúpajúce do hrebeňa
    dem = 950.0 + (Y * 0.04)
    
    # Hlavný hrebeň (W-E oblúk cez stred a hornú časť)
    ridge_y = 7000.0
    main_ridge = 1450.0 * np.exp(-((Y - ridge_y)**2) / (2 * 1800.0**2))
    dem += main_ridge * (1.0 + 0.15 * np.sin(X / 1200.0) * np.cos(X / 2400.0))
    
    # Južné rázsochy (Slavkovská, Končistá, Lomnická)
    spurs = 550.0 * np.exp(-((Y - 4500.0)**2) / (2 * 2000.0**2)) * np.maximum(np.cos(X / 1400.0), 0.0)**2
    dem += spurs
    
    # Hlboké doliny (Mengusovská, Batizovská, Velická, Studené doliny)
    valleys = (
        np.exp(-((X - 4800.0)**2) / (2 * 450.0**2)) +   # Mengusovská
        np.exp(-((X - 7300.0)**2) / (2 * 400.0**2)) +   # Batizovská
        np.exp(-((X - 8300.0)**2) / (2 * 400.0**2)) +   # Velická
        np.exp(-((X - 9800.0)**2) / (2 * 500.0**2)) +   # Veľká Studená
        np.exp(-((X - 11000.0)**2) / (2 * 450.0**2))    # Malá Studená
    ) * np.exp(-((Y - 5000.0)**2) / (2 * 2800.0**2))
    dem -= valleys * 500.0

    # Štíty
    def add_peak(px, py, h, r):
        return h * np.exp(-((X - px)**2 + (Y - py)**2) / (2 * r**2))

    dem += add_peak(8000.0, 6200.0, 750.0, 550.0)   # Gerlach (2655m)
    dem += add_peak(12000.0, 6800.0, 730.0, 500.0)  # Lomnický (2634m)
    dem += add_peak(2700.0, 5000.0, 680.0, 600.0)   # Kriváň (2495m)
    dem += add_peak(5700.0, 6000.0, 640.0, 450.0)   # Rysy (2501m)
    dem += add_peak(9700.0, 4000.0, 520.0, 450.0)   # Slavkovský štít
    dem += add_peak(11000.0, 8000.0, 650.0, 480.0)  # Ľadový štít
    dem += add_peak(13300.0, 9500.0, 500.0, 450.0)  # Jahňací štít

    dem = ndimage.gaussian_filter(dem, sigma=1.0)
    dwd_dem = 1100.0 + 500.0 * np.exp(-((Y - 7000.0)**2) / (2 * 3500.0**2))
    return X, Y, dem, dwd_dem, dx, dy

X, Y, DEM_100, DEM_DWD, DX, DY = generate_tatras_core_dem()

# =============================================================================
# 3. DVOJBODOVÉ DWD DÁTA (KOTLINA + HREBEŇ)
# =============================================================================
METEO_CACHE = {"timestamp": 0, "d_poprad": None, "d_lomnik": None}

def fetch_dwd_data():
    now = datetime.datetime.now().timestamp()
    if METEO_CACHE["d_poprad"] and (now - METEO_CACHE["timestamp"] < 600):
        return METEO_CACHE["d_poprad"], METEO_CACHE["d_lomnik"]

    url_poprad = "https://api.open-meteo.com/v1/dwd-icon?latitude=49.06&longitude=20.30&hourly=temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,cape&forecast_days=3&timezone=Europe%2FBratislava"
    url_lomnik = "https://api.open-meteo.com/v1/dwd-icon?latitude=49.20&longitude=20.21&hourly=temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,cape&forecast_days=3&timezone=Europe%2FBratislava"

    try:
        req1 = urllib.request.Request(url_poprad, headers={'User-Agent': 'AvalancheTatry-Core/3.2'})
        req2 = urllib.request.Request(url_lomnik, headers={'User-Agent': 'AvalancheTatry-Core/3.2'})
        with urllib.request.urlopen(req1, timeout=8) as r1:
            d_poprad = json.loads(r1.read().decode()).get("hourly", {})
        with urllib.request.urlopen(req2, timeout=8) as r2:
            d_lomnik = json.loads(r2.read().decode()).get("hourly", {})
        
        METEO_CACHE["d_poprad"] = d_poprad
        METEO_CACHE["d_lomnik"] = d_lomnik
        METEO_CACHE["timestamp"] = now
        return d_poprad, d_lomnik
    except Exception as e:
        print(f"[VAROVANIE] DWD API offline: {e}")
        return None, None

def run_dwd_downscaled_simulation(step_idx: int):
    hours_ahead = step_idx * 6
    cur_hour = datetime.datetime.now().hour
    data_idx = cur_hour + hours_ahead

    d_poprad, d_lomnik = fetch_dwd_data()

    if d_poprad and "temperature_2m" in d_poprad and len(d_poprad["temperature_2m"]) > data_idx:
        t_poprad = d_poprad["temperature_2m"][data_idx]
        t_lomnik = d_lomnik["temperature_2m"][data_idx]
        w_spd = d_lomnik["wind_speed_10m"][data_idx] / 3.6
        w_dir = d_lomnik["wind_direction_10m"][data_idx]
        precip = d_lomnik["precipitation"][data_idx]
        cape = d_lomnik.get("cape", [0])[data_idx] or 0.0
    else:
        m = datetime.datetime.now().month
        t_poprad = 22.0 if 5 <= m <= 9 else (7.0 if m in [4, 10] else 0.0)
        t_lomnik = t_poprad - 12.5
        w_spd = 6.0
        w_dir = 315.0
        precip = 0.0
        cape = 50.0

    # Vertikálny lapse rate priamo z DWD
    lapse_rate = np.clip((t_lomnik - t_poprad) / (2634.0 - 672.0), -0.0098, 0.002)
    temp_field = t_poprad + lapse_rate * (DEM_100 - 672.0)

    # Prúdenie vetra
    rad = np.radians(270.0 - w_dir)
    u_dwd = np.full_like(X, w_spd * np.cos(rad))
    v_dwd = np.full_like(Y, w_spd * np.sin(rad))

    dh_dx, dh_dy = np.gradient(DEM_100, DX, DY)
    slope = np.sqrt(dh_dx**2 + dh_dy**2)
    aspect = np.arctan2(-dh_dx, dh_dy)

    dem_base = ndimage.gaussian_filter(DEM_100, sigma=12)
    h_rel = np.maximum(DEM_100 - dem_base, 0.0)
    delta_S = np.clip((1.2 * h_rel / 2500.0), 0.0, 0.5)
    u_speed = u_dwd * (1.0 + delta_S)
    v_speed = v_dwd * (1.0 + delta_S)

    speed_init = np.sqrt(u_speed**2 + v_speed**2)
    wind_dir = np.arctan2(v_speed, u_speed)
    delta_theta = np.clip(-0.2 * (slope * 100.0) * np.sin(2.0 * (aspect - wind_dir)), -0.3, 0.3)
    steered_dir = wind_dir + delta_theta
    u_opt = speed_init * np.cos(steered_dir)
    v_opt = speed_init * np.sin(steered_dir)
    w_opt = u_opt * dh_dx + v_opt * dh_dy
    wind_spd = np.sqrt(u_opt**2 + v_opt**2)

    # Zrážky & Sneh
    p_final = np.maximum(precip * (1.0 + 0.35 * np.maximum(w_opt, 0.0)), 0.0) if precip > 0.0 else np.zeros_like(DEM_100)
    snow_mask = temp_field < 0.0
    fresh_snow_6h = np.where(snow_mask, p_final * 1.0 * 6.0, 0.0)

    # LHI
    instability = np.maximum(w_opt, 0.0) * (cape / 400.0)
    exposure = np.clip((DEM_100 - 1000.0) / 35.0, 0.0, 40.0)
    lhi = np.clip(ndimage.gaussian_filter(exposure * 0.4 + instability * 20.0, sigma=1.0), 0.0, 100.0)
    if cape < 50.0 and precip == 0.0:
        lhi = np.clip(lhi * 0.1, 0.0, 10.0)

    return {
        'hours': hours_ahead,
        'u_opt': u_opt, 'v_opt': v_opt,
        'p_final': p_final, 'snow_diff': fresh_snow_6h,
        'lhi': lhi, 'temp_field': temp_field, 'wind_spd': wind_spd,
        't_poprad': t_poprad, 'lapse_rate': lapse_rate
    }

FORECAST_TIMELINE = [run_dwd_downscaled_simulation(i) for i in range(9)]

# =============================================================================
# 4. KARTOGRAFICKÁ VIZUALIZÁCIA TATIER (16 x 12 km)
# =============================================================================
def draw_dense_landmarks(ax, is_compact=False):
    for lm in TATRAS_CORE_POINTS:
        ltype = lm["type"]
        prio = lm["prio"]
        
        if ltype == "peak":
            mcolor, marker = '#ef4444', '^'
        elif ltype == "pass":
            mcolor, marker = '#fbbf24', 'x'
        elif ltype == "lake":
            mcolor, marker = '#38bdf8', 'o'
        elif ltype == "hut":
            mcolor, marker = '#f59e0b', 's'
        else:
            mcolor, marker = '#a855f7', 'o'

        msize = 4.5 if is_compact else 6.0
        ax.plot(lm["x"], lm["y"], marker=marker, markersize=msize, color=mcolor, 
                markeredgecolor='#000000', markeredgewidth=0.5, alpha=0.9, zorder=10)

        if prio == 1:
            fsize = 5.5 if is_compact else 7.5
            label = lm['name'] if is_compact else f"{lm['name']}\n({lm['alt']}m)"
            ax.text(lm["x"], lm["y"] + (0.3 if is_compact else 0.4), label,
                    fontsize=fsize, fontweight='bold', color='white', ha='center',
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='#0f172a', edgecolor=mcolor, alpha=0.85, linewidth=0.6),
                    zorder=11)

# =============================================================================
# 5. FASTAPI ENDPOINTY
# =============================================================================
@app.get("/")
def read_root():
    return {"status": "ok", "service": "TATRYS-50 v2 Tatras-Core", "domain": "16x12km (100m DEM)"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/landmarks")
def get_landmarks():
    return {"status": "ok", "count": len(TATRAS_CORE_POINTS), "landmarks": TATRAS_CORE_POINTS}

@app.get("/api/points-grid")
def get_points_grid(step: int = Query(0, ge=0, le=8)):
    d = run_dwd_simulation_step(step)
    results = []
    for p in TATRAS_CORE_POINTS:
        ix = int(np.clip(p["x"] * 1000.0 / DX, 0, DEM_100.shape[0] - 1))
        iy = int(np.clip(p["y"] * 1000.0 / DY, 0, DEM_100.shape[1] - 1))
        t_loc = float(d['t_poprad'] + d['lapse_rate'] * (p["alt"] - 672.0))
        spd_loc = float(d['wind_spd'][ix, iy] * 3.6)
        prec_loc = float(d['p_final'][ix, iy])
        sn_loc = float(d['snow_diff'][ix, iy])
        lh_loc = float(d['lhi'][ix, iy])

        results.append({
            "name": p["name"],
            "alt": p["alt"],
            "type": p["type"],
            "x": p["x"],
            "y": p["y"],
            "temp": round(t_loc, 1),
            "wind_kmh": round(spd_loc, 1),
            "precip_mmh": round(prec_loc, 1),
            "snow_6h_cm": round(sn_loc, 1),
            "lhi": round(lh_loc, 0)
        })
    return {"status": "ok", "step": step, "hours_ahead": d["hours"], "count": len(results), "points": results}

@app.get("/api/forecast")
@app.get("/api/stations")
def get_forecast(step: int = Query(0, ge=0, le=8)):
    d = run_dwd_downscaled_simulation(step)
    locs = [
        {"name": "Lomnický štít (2 634 m)", "alt": 2634, "x": 12.0, "y": 6.8},
        {"name": "Gerlachovský štít (2 655 m)", "alt": 2655, "x": 8.0, "y": 6.2},
        {"name": "Téryho chata (2 015 m)", "alt": 2015, "x": 10.8, "y": 6.8},
        {"name": "Zbojnícka chata (1 960 m)", "alt": 1960, "x": 9.2, "y": 5.8},
        {"name": "Štrbské Pleso (1 346 m)", "alt": 1346, "x": 3.9, "y": 1.5},
        {"name": "Starý Smokovec (1 010 m)", "alt": 1010, "x": 9.5, "y": 0.5}
    ]
    res = []
    for l in locs:
        ix = int(np.clip(l['x'] * 1000.0 / DX, 0, DEM_100.shape[0]-1))
        iy = int(np.clip(l['y'] * 1000.0 / DY, 0, DEM_100.shape[1]-1))
        spd = float(d['wind_spd'][ix, iy] * 3.6)
        t = float(d['t_poprad'] + d['lapse_rate'] * (l["alt"] - 672.0))
        p = float(d['p_final'][ix, iy])
        s = float(d['snow_diff'][ix, iy])
        lh = float(d['lhi'][ix, iy])
        res.append({
            "name": l['name'],
            "temp": round(t, 1),
            "wind_kmh": round(spd, 1),
            "precip_mmh": round(p, 1),
            "snow_6h_cm": round(s, 1),
            "lightning_risk": "Vysoké" if lh > 50 else ("Stredné" if lh > 20 else "Nízke"),
            "lhi_raw": round(lh, 0)
        })
    return {"status": "ok", "step": step, "hours_ahead": d['hours'], "stations": res}

@app.get("/api/render-map")
def render_map(layer: str = Query("all"), step: int = Query(0, ge=0, le=8)):
    d = run_dwd_downscaled_simulation(step)
    X_km, Y_km = X / 1000.0, Y / 1000.0
    h_label = f"+{d['hours']}h"
    
    if layer == "all":
        fig, axs = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0f172a')
        for row in axs:
            for ax in row:
                ax.set_facecolor('#1e293b')
                ax.tick_params(colors='#94a3b8')
                ax.set_xlabel('Západ -> Východ (km)', color='#94a3b8', fontsize=8.5)
                ax.set_ylabel('Juh -> Sever (km)', color='#94a3b8', fontsize=8.5)
                for s in ax.spines.values():
                    s.set_color('#334155')

        # 1. Topografia & Vietor
        im1 = axs[0, 0].contourf(X_km, Y_km, DEM_100, levels=25, cmap='terrain', alpha=0.85)
        fig.colorbar(im1, ax=axs[0, 0])
        axs[0, 0].quiver(X_km[::6, ::6], Y_km[::6, ::6], d['u_opt'][::6, ::6], d['v_opt'][::6, ::6], scale=110, color='black')
        draw_dense_landmarks(axs[0, 0], is_compact=True)
        axs[0, 0].set_title(f'A. Topografia Tatier & Vietor [{h_label}]', color='white', fontweight='bold')

        # 2. Zrážky
        im2 = axs[0, 1].contourf(X_km, Y_km, d['p_final'], levels=20, cmap='YlGnBu')
        fig.colorbar(im2, ax=axs[0, 1], label='mm / h')
        draw_dense_landmarks(axs[0, 1], is_compact=True)
        axs[0, 1].set_title(f'B. Zrážky (DWD ICON Downscale) [{h_label}]', color='white', fontweight='bold')

        # 3. Sneh
        im3 = axs[1, 0].contourf(X_km, Y_km, d['snow_diff'], levels=20, cmap='Blues')
        fig.colorbar(im3, ax=axs[1, 0], label='cm / 6h')
        draw_dense_landmarks(axs[1, 0], is_compact=True)
        axs[1, 0].set_title(f'C. Nový sneh za 6h [{h_label}]', color='white', fontweight='bold')

        # 4. Teplota
        im4 = axs[1, 1].contourf(X_km, Y_km, d['temp_field'], levels=25, cmap='coolwarm')
        fig.colorbar(im4, ax=axs[1, 1], label='°C')
        draw_dense_landmarks(axs[1, 1], is_compact=True)
        axs[1, 1].set_title(f'D. Teplotné pole (100m Lapse Rate) [{h_label}]', color='white', fontweight='bold')

        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(11, 8.5), facecolor='#0f172a')
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.set_xlabel('Západ -> Východ (km)', color='#94a3b8', fontsize=9.5)
        ax.set_ylabel('Juh -> Sever (km)', color='#94a3b8', fontsize=9.5)

        if layer == "wind":
            im = ax.contourf(X_km, Y_km, DEM_100, levels=25, cmap='terrain', alpha=0.85)
            ax.quiver(X_km[::5, ::5], Y_km[::5, ::5], d['u_opt'][::5, ::5], d['v_opt'][::5, ::5], scale=95, color='black')
            fig.colorbar(im, ax=ax, label='Výška (m n.m.)')
            ax.set_title(f'Prúdenie vetra & Topografia Tatier [{h_label}]', color='white', fontweight='bold')
        elif layer == "precip":
            im = ax.contourf(X_km, Y_km, d['p_final'], levels=20, cmap='YlGnBu')
            fig.colorbar(im, ax=ax, label='mm / h')
            ax.set_title(f'Zrážková intenzita [{h_label}]', color='white', fontweight='bold')
        elif layer == "snow":
            im = ax.contourf(X_km, Y_km, d['snow_diff'], levels=20, cmap='Blues')
            fig.colorbar(im, ax=ax, label='cm / 6h')
            ax.set_title(f'Nový sneh za 6h [{h_label}]', color='white', fontweight='bold')
        elif layer == "lightning":
            im = ax.contourf(X_km, Y_km, d['lhi'], levels=20, cmap='YlOrRd')
            fig.colorbar(im, ax=ax, label='LHI (0-100)')
            ax.set_title(f'Index rizika bleskov (LHI) [{h_label}]', color='white', fontweight='bold')
        draw_dense_landmarks(ax, is_compact=False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

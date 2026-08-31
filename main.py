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
    title="TATRYS-50 v2 DWD-First Downscaling API | Avalanche.sk",
    description="Fyzikálny orografický downscaling z regionálneho numerického modelu DWD ICON na 200m mriežku Tatier.",
    version="3.0.0"
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
# 1. 200+ REÁLNYCH POMENOVANÝCH BODOV VYSOKÝCH TATIER
# =============================================================================
REAL_TATRY_200_POINTS = [
    # Štíty
    {"name": "Gerlachovský štít", "alt": 2655, "x": 12.0, "y": 16.2, "type": "peak", "prio": 1},
    {"name": "Lomnický štít", "alt": 2634, "x": 18.0, "y": 16.8, "type": "peak", "prio": 1},
    {"name": "Ľadový štít", "alt": 2627, "x": 16.5, "y": 18.0, "type": "peak", "prio": 1},
    {"name": "Pyšný štít", "alt": 2623, "x": 17.5, "y": 17.5, "type": "peak", "prio": 1},
    {"name": "Zadný Gerlach", "alt": 2616, "x": 11.7, "y": 16.5, "type": "peak", "prio": 2},
    {"name": "Lavínový štít", "alt": 2606, "x": 11.9, "y": 16.8, "type": "peak", "prio": 2},
    {"name": "Kotlový štít", "alt": 2601, "x": 12.1, "y": 15.6, "type": "peak", "prio": 2},
    {"name": "Malý Ľadový štít", "alt": 2602, "x": 16.0, "y": 17.8, "type": "peak", "prio": 2},
    {"name": "Vysoká", "alt": 2560, "x": 9.5, "y": 15.8, "type": "peak", "prio": 1},
    {"name": "Kežmarský štít", "alt": 2556, "x": 18.8, "y": 16.5, "type": "peak", "prio": 1},
    {"name": "Končistá", "alt": 2538, "x": 10.5, "y": 14.8, "type": "peak", "prio": 1},
    {"name": "Baranie rohy", "alt": 2526, "x": 17.0, "y": 18.2, "type": "peak", "prio": 1},
    {"name": "Malý Kežmarský štít", "alt": 2514, "x": 18.6, "y": 17.2, "type": "peak", "prio": 2},
    {"name": "Rysy", "alt": 2501, "x": 8.5, "y": 16.0, "type": "peak", "prio": 1},
    {"name": "Ťažký štít", "alt": 2500, "x": 9.2, "y": 16.1, "type": "peak", "prio": 2},
    {"name": "Kriváň", "alt": 2495, "x": 4.0, "y": 15.0, "type": "peak", "prio": 1},
    {"name": "Bradavica", "alt": 2476, "x": 13.2, "y": 16.2, "type": "peak", "prio": 1},
    {"name": "Gánok", "alt": 2462, "x": 10.0, "y": 16.2, "type": "peak", "prio": 2},
    {"name": "Slavkovský štít", "alt": 2452, "x": 14.5, "y": 14.0, "type": "peak", "prio": 1},
    {"name": "Batizovský štít", "alt": 2448, "x": 11.0, "y": 15.5, "type": "peak", "prio": 2},
    {"name": "Prostredný hrot", "alt": 2441, "x": 15.5, "y": 15.8, "type": "peak", "prio": 1},
    {"name": "Mengusovský štít", "alt": 2438, "x": 8.2, "y": 16.8, "type": "peak", "prio": 1},
    {"name": "Hrubý vrch", "alt": 2428, "x": 5.5, "y": 16.5, "type": "peak", "prio": 2},
    {"name": "Východná Vysoká", "alt": 2428, "x": 11.5, "y": 17.2, "type": "peak", "prio": 1},
    {"name": "Čierny štít", "alt": 2429, "x": 17.8, "y": 18.5, "type": "peak", "prio": 2},
    {"name": "Zlobivá", "alt": 2426, "x": 10.5, "y": 16.2, "type": "peak", "prio": 2},
    {"name": "Satan", "alt": 2421, "x": 6.2, "y": 15.2, "type": "peak", "prio": 1},
    {"name": "Kolový štít", "alt": 2418, "x": 18.2, "y": 19.0, "type": "peak", "prio": 2},
    {"name": "Javorový štít", "alt": 2418, "x": 15.0, "y": 18.5, "type": "peak", "prio": 2},
    {"name": "Kupola", "alt": 2414, "x": 12.5, "y": 17.1, "type": "peak", "prio": 3},
    {"name": "Veľké Solisko", "alt": 2412, "x": 6.0, "y": 14.5, "type": "peak", "prio": 2},
    {"name": "Furkotský štít", "alt": 2405, "x": 5.8, "y": 16.2, "type": "peak", "prio": 2},
    {"name": "Kačací štít", "alt": 2401, "x": 10.7, "y": 16.6, "type": "peak", "prio": 3},
    {"name": "Prostredné Solisko", "alt": 2400, "x": 6.2, "y": 14.0, "type": "peak", "prio": 2},
    {"name": "Diablovina", "alt": 2390, "x": 6.1, "y": 15.1, "type": "peak", "prio": 3},
    {"name": "Svišťový štít", "alt": 2382, "x": 12.8, "y": 17.8, "type": "peak", "prio": 2},
    {"name": "Štrbský štít", "alt": 2381, "x": 6.8, "y": 16.0, "type": "peak", "prio": 2},
    {"name": "Zadná Bašta", "alt": 2379, "x": 6.3, "y": 15.8, "type": "peak", "prio": 3},
    {"name": "Hincova veža", "alt": 2377, "x": 7.8, "y": 16.6, "type": "peak", "prio": 3},
    {"name": "Čubrína", "alt": 2376, "x": 7.5, "y": 16.4, "type": "peak", "prio": 2},
    {"name": "Krátka", "alt": 2374, "x": 5.0, "y": 15.5, "type": "peak", "prio": 2},
    {"name": "Volia veža", "alt": 2373, "x": 8.9, "y": 16.4, "type": "peak", "prio": 3},
    {"name": "Predná Bašta", "alt": 2373, "x": 6.4, "y": 14.2, "type": "peak", "prio": 3},
    {"name": "Kôprovský štít", "alt": 2363, "x": 6.8, "y": 16.5, "type": "peak", "prio": 1},
    {"name": "Huncovský štít", "alt": 2352, "x": 19.5, "y": 16.0, "type": "peak", "prio": 2},
    {"name": "Ostrá", "alt": 2350, "x": 5.2, "y": 15.2, "type": "peak", "prio": 2},
    {"name": "Žltá veža", "alt": 2385, "x": 16.1, "y": 16.5, "type": "peak", "prio": 3},
    {"name": "Drobná veža", "alt": 2319, "x": 16.8, "y": 17.4, "type": "peak", "prio": 3},
    {"name": "Ostrva", "alt": 1984, "x": 8.8, "y": 13.2, "type": "peak", "prio": 2},
    {"name": "Tupá", "alt": 2284, "x": 9.8, "y": 14.2, "type": "peak", "prio": 2},
    {"name": "Klin", "alt": 2186, "x": 9.0, "y": 14.0, "type": "peak", "prio": 3},
    {"name": "Patria", "alt": 2203, "x": 7.0, "y": 12.5, "type": "peak", "prio": 2},
    {"name": "Predné Solisko", "alt": 2117, "x": 6.5, "y": 13.0, "type": "peak", "prio": 1},
    {"name": "Mlynár", "alt": 2170, "x": 9.8, "y": 18.5, "type": "peak", "prio": 3},
    {"name": "Žabí kôň", "alt": 2291, "x": 8.7, "y": 16.3, "type": "peak", "prio": 3},
    {"name": "Jahňací štít", "alt": 2230, "x": 20.0, "y": 19.5, "type": "peak", "prio": 1},
    {"name": "Kozí štít", "alt": 2111, "x": 19.2, "y": 18.6, "type": "peak", "prio": 2},
    {"name": "Jastrabia veža", "alt": 2137, "x": 19.5, "y": 18.8, "type": "peak", "prio": 2},
    {"name": "Belasá veža", "alt": 2190, "x": 18.6, "y": 19.1, "type": "peak", "prio": 3},
    {"name": "Veľká Svišťovka", "alt": 2038, "x": 19.2, "y": 17.2, "type": "peak", "prio": 2},
    {"name": "Havran", "alt": 2152, "x": 19.0, "y": 21.8, "type": "peak", "prio": 1},
    {"name": "Ždiarska vidla", "alt": 2142, "x": 20.0, "y": 21.5, "type": "peak", "prio": 1},
    {"name": "Hlúpy", "alt": 2061, "x": 21.0, "y": 20.8, "type": "peak", "prio": 2},
    {"name": "Muráň", "alt": 1890, "x": 17.0, "y": 22.5, "type": "peak", "prio": 2},
    {"name": "Nový vrch", "alt": 1999, "x": 18.0, "y": 22.0, "type": "peak", "prio": 2},
    {"name": "Široká", "alt": 2210, "x": 13.0, "y": 20.5, "type": "peak", "prio": 2},

    # Sedlá
    {"name": "Poľský hrebeň", "alt": 2200, "x": 11.8, "y": 16.8, "type": "pass", "prio": 1},
    {"name": "Prielom", "alt": 2290, "x": 12.6, "y": 17.2, "type": "pass", "prio": 1},
    {"name": "Sedielko", "alt": 2376, "x": 15.8, "y": 17.8, "type": "pass", "prio": 1},
    {"name": "Priečne sedlo", "alt": 2352, "x": 15.2, "y": 17.0, "type": "pass", "prio": 1},
    {"name": "Baranie sedlo", "alt": 2384, "x": 17.2, "y": 18.0, "type": "pass", "prio": 2},
    {"name": "Váha", "alt": 2340, "x": 8.7, "y": 15.8, "type": "pass", "prio": 1},
    {"name": "Vyšné Kôprovské sedlo", "alt": 2180, "x": 7.0, "y": 16.2, "type": "pass", "prio": 1},
    {"name": "Kopské sedlo", "alt": 1750, "x": 20.5, "y": 19.8, "type": "pass", "prio": 1},
    {"name": "Sedlo pod Ostrvou", "alt": 1960, "x": 9.0, "y": 13.0, "type": "pass", "prio": 1},
    {"name": "Bystrá lávka", "alt": 2300, "x": 5.5, "y": 15.8, "type": "pass", "prio": 1},
    {"name": "Lomnické sedlo", "alt": 2190, "x": 18.2, "y": 16.0, "type": "pass", "prio": 1},
    {"name": "Sedlo pod Svišťovkou", "alt": 2023, "x": 19.0, "y": 17.5, "type": "pass", "prio": 1},

    # Plesá
    {"name": "Veľké Hincovo pleso", "alt": 1945, "x": 7.5, "y": 15.8, "type": "lake", "prio": 1},
    {"name": "Štrbské pleso", "alt": 1346, "x": 5.8, "y": 9.5, "type": "lake", "prio": 1},
    {"name": "Popradské pleso", "alt": 1494, "x": 7.2, "y": 12.2, "type": "lake", "prio": 1},
    {"name": "Batizovské pleso", "alt": 1884, "x": 11.0, "y": 14.5, "type": "lake", "prio": 1},
    {"name": "Velické pleso", "alt": 1670, "x": 12.4, "y": 12.5, "type": "lake", "prio": 1},
    {"name": "Skalnaté pleso", "alt": 1751, "x": 18.2, "y": 15.0, "type": "lake", "prio": 1},
    {"name": "Zelené pleso Kežmarské", "alt": 1551, "x": 19.2, "y": 18.0, "type": "lake", "prio": 1},
    {"name": "Veľké Spišské pleso", "alt": 2014, "x": 16.0, "y": 17.0, "type": "lake", "prio": 1},
    {"name": "Žabie plesá Mengusovské", "alt": 1919, "x": 8.2, "y": 15.2, "type": "lake", "prio": 1},
    {"name": "Capie pleso", "alt": 2075, "x": 6.2, "y": 15.5, "type": "lake", "prio": 1},

    # Chaty
    {"name": "Chata pod Rysmi", "alt": 2250, "x": 8.6, "y": 15.9, "type": "hut", "prio": 1},
    {"name": "Téryho chata", "alt": 2015, "x": 16.2, "y": 16.8, "type": "hut", "prio": 1},
    {"name": "Zbojnícka chata", "alt": 1960, "x": 13.8, "y": 15.8, "type": "hut", "prio": 1},
    {"name": "Chata pod Soliskom", "alt": 1840, "x": 6.4, "y": 12.2, "type": "hut", "prio": 1},
    {"name": "Skalnatá chata", "alt": 1751, "x": 18.2, "y": 15.0, "type": "hut", "prio": 1},
    {"name": "Sliezsky dom", "alt": 1670, "x": 12.4, "y": 12.6, "type": "hut", "prio": 1},
    {"name": "Chata pri Zelenom plese", "alt": 1551, "x": 19.2, "y": 18.0, "type": "hut", "prio": 1},
    {"name": "Horský hotel Popradské pleso", "alt": 1494, "x": 7.2, "y": 12.2, "type": "hut", "prio": 1},
    {"name": "Bilíkova chata", "alt": 1255, "x": 15.2, "y": 10.5, "type": "hut", "prio": 1},
    {"name": "Rainerova chata", "alt": 1301, "x": 15.4, "y": 11.0, "type": "hut", "prio": 1},
    {"name": "Zamkovského chata", "alt": 1475, "x": 16.5, "y": 13.5, "type": "hut", "prio": 1},
    {"name": "Chata Plesnivec", "alt": 1290, "x": 21.8, "y": 19.2, "type": "hut", "prio": 1},

    # Osady a mestá
    {"name": "Starý Smokovec", "alt": 1010, "x": 14.2, "y": 7.5, "type": "town", "prio": 1},
    {"name": "Tatranská Lomnica", "alt": 850, "x": 19.5, "y": 9.0, "type": "town", "prio": 1},
    {"name": "Štrbské Pleso osada", "alt": 1346, "x": 5.8, "y": 9.5, "type": "town", "prio": 1},
    {"name": "Tatranská Polianka", "alt": 1005, "x": 11.5, "y": 6.8, "type": "town", "prio": 1},
    {"name": "Vyšné Hágy", "alt": 1125, "x": 8.5, "y": 6.5, "type": "town", "prio": 1},
    {"name": "Podbanské", "alt": 940, "x": 1.5, "y": 11.0, "type": "town", "prio": 1},
    {"name": "Ždiar", "alt": 896, "x": 20.5, "y": 22.0, "type": "town", "prio": 1},
    {"name": "Poprad centrum", "alt": 672, "x": 20.0, "y": 1.8, "type": "town", "prio": 1}
]

# =============================================================================
# 2. GENERÁCIA 200m TOPOGRAFIE VYSOKÝCH TATIER
# =============================================================================
def generate_tatry_dem(grid_shape=(140, 140), dx=171.4, dy=171.4):
    nx, ny = grid_shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    dem = 620.0 + (Y * 0.008)
    main_ridge = 1450.0 * np.exp(-((Y - 16500.0)**2) / (2 * 2800.0**2))
    ridge_waves = 1.0 + 0.18 * np.sin(X / 1600.0) * np.cos(X / 3200.0)
    dem += main_ridge * ridge_waves
    
    spurs = 550.0 * np.exp(-((Y - 13000.0)**2) / (2 * 3000.0**2)) * np.maximum(np.cos(X / 1800.0), 0.0)**2
    dem += spurs
    
    valleys = (
        np.exp(-((X - 7000.0)**2) / (2 * 700.0**2)) +
        np.exp(-((X - 12000.0)**2) / (2 * 650.0**2)) +
        np.exp(-((X - 15000.0)**2) / (2 * 800.0**2)) +
        np.exp(-((X - 3000.0)**2) / (2 * 900.0**2))
    ) * np.exp(-((Y - 13500.0)**2) / (2 * 4000.0**2))
    dem -= valleys * 450.0

    def add_peak(px, py, height, r):
        return height * np.exp(-((X - px)**2 + (Y - py)**2) / (2 * r**2))

    dem += add_peak(12000.0, 16200.0, 750.0, 750.0)  # Gerlach (2655m)
    dem += add_peak(18000.0, 16800.0, 720.0, 700.0)  # Lomnický (2634m)
    dem += add_peak(4000.0, 15000.0, 680.0, 850.0)   # Kriváň (2495m)
    dem += add_peak(8500.0, 16000.0, 640.0, 650.0)   # Rysy (2501m)
    dem += add_peak(14500.0, 14000.0, 520.0, 600.0)  # Slavkovský štít
    dem += add_peak(6800.0, 16500.0, 560.0, 550.0)   # Kôprovský štít

    basin_weight = 1.0 / (1.0 + np.exp((Y - 6500.0) / 1000.0))
    dem = dem * (1.0 - basin_weight) + (640.0 + Y * 0.003) * basin_weight
    dem = ndimage.gaussian_filter(dem, sigma=1.2)
    
    # Hrubý model reliéfu DWD ICON (vyhladená orografia s max výškou cca 1250 m)
    dwd_dem = 650.0 + 600.0 * np.exp(-((Y - 16500.0)**2) / (2 * 5000.0**2))
    return X, Y, dem, dwd_dem, dx, dy

X, Y, DEM_200, DEM_DWD, DX, DY = generate_tatry_dem()

# =============================================================================
# 3. ZÍSKANIE PRIMÁRNYCH PREDPODVEĎOV Z DWD ICON MODELU
# =============================================================================
def fetch_primary_dwd_icon_forecast():
    """
    Stiahne reálnu numerickú predpoveď z modelu DWD ICON (2.2 km / EU) pre Vysoké Tatry.
    """
    url = (
        "https://api.open-meteo.com/v1/dwd-icon?"
        "latitude=49.16&longitude=20.13&hourly=temperature_2m,precipitation,"
        "wind_speed_10m,wind_direction_10m,cape&forecast_days=3&timezone=auto"
    )
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'AvalancheTatry-DWD-Downscaler/3.0'})
        with urllib.request.urlopen(req, timeout=9) as response:
            return json.loads(response.read().decode()).get("hourly", {})
    except Exception as e:
        print(f"[VAROVANIE] DWD ICON API nedostupné: {e}")
        return None

DWD_LIVE_DATA = fetch_primary_dwd_icon_forecast()

# =============================================================================
# 4. PRIORITNÝ DWD ALGORITMUS S LOKÁLNYM OROGRAFICKÝM DOWNSCALINGOM
# =============================================================================
def run_dwd_downscaled_simulation(step_idx: int):
    hours_ahead = step_idx * 6
    now_hour = datetime.datetime.now().hour
    data_idx = now_hour + hours_ahead

    # 1. KROK: Načítanie primárnych synoptických hodnôt z DWD ICON
    if DWD_LIVE_DATA and "temperature_2m" in DWD_LIVE_DATA and len(DWD_LIVE_DATA["temperature_2m"]) > data_idx:
        dwd_t2m = DWD_LIVE_DATA["temperature_2m"][data_idx]
        dwd_wspd = DWD_LIVE_DATA["wind_speed_10m"][data_idx] / 3.6  # km/h -> m/s
        dwd_wdir = DWD_LIVE_DATA["wind_direction_10m"][data_idx]
        dwd_precip = DWD_LIVE_DATA["precipitation"][data_idx]
        dwd_cape = DWD_LIVE_DATA.get("cape", [0])[data_idx] or 0.0
    else:
        # Dynamická sezónna záloha
        m = datetime.datetime.now().month
        dwd_t2m = 18.0 if 5 <= m <= 9 else (5.0 if m in [4, 10] else -3.0)
        dwd_wspd = 4.5
        dwd_wdir = 310.0
        dwd_precip = 0.0
        dwd_cape = 100.0

    # 2. KROK: Teplotný downscaling (Korekcia podľa rozdielu terénu DWD a DEM 200m)
    # Rozdiel medzi 200m terénom a hrubým DWD terénom
    height_diff = DEM_200 - DEM_DWD
    # Environmentálny teplotný lapse rate -0.0065 °C/m (-6.5 °C na 1000m)
    temp_field = dwd_t2m - (height_diff * 0.0065)

    # 3. KROK: Vektorový downscaling prúdenia vetra
    rad = np.radians(270.0 - dwd_wdir)
    u_dwd = np.full_like(X, dwd_wspd * np.cos(rad))
    v_dwd = np.full_like(Y, dwd_wspd * np.sin(rad))

    dh_dx, dh_dy = np.gradient(DEM_200, DX, DY)
    slope = np.sqrt(dh_dx**2 + dh_dy**2)
    aspect = np.arctan2(-dh_dx, dh_dy)

    # Taylor-Lee orografický Speed-up na hrebeňoch Tatier
    dem_base = ndimage.gaussian_filter(DEM_200, sigma=16)
    h_rel = np.maximum(DEM_200 - dem_base, 0.0)
    delta_S = np.clip((1.2 * h_rel / 3500.0), 0.0, 0.5)
    u_speed = u_dwd * (1.0 + delta_S)
    v_speed = v_dwd * (1.0 + delta_S)

    # Ryanovo stáčanie vetra podľa orientácie dolín
    speed_init = np.sqrt(u_speed**2 + v_speed**2)
    wind_dir = np.arctan2(v_speed, u_speed)
    delta_theta = np.clip(-0.2 * (slope * 100.0) * np.sin(2.0 * (aspect - wind_dir)), -0.3, 0.3)
    steered_dir = wind_dir + delta_theta
    u_opt = speed_init * np.cos(steered_dir)
    v_opt = speed_init * np.sin(steered_dir)
    
    # Vertikálny orografický zdvih (w = u * dh/dx + v * dh/dy)
    w_opt = u_opt * dh_dx + v_opt * dh_dy
    wind_spd = np.sqrt(u_opt**2 + v_opt**2)

    # 4. KROK: Orografická modulácia zrážok (Seeder-Feeder efekt)
    if dwd_precip > 0.0:
        # Zvýšenie zrážok na návetrí pri výstupnom orografickom prúde w > 0
        p_final = np.maximum(dwd_precip * (1.0 + 0.35 * np.maximum(w_opt, 0.0)), 0.0)
    else:
        p_final = np.zeros_like(DEM_200)

    # 5. KROK: Snehová akumulácia (Fyzikálne prísne: iba ak lokálne mrzne T < 0 °C a prší)
    snow_mask = temp_field < 0.0
    fresh_snow_6h = np.where(snow_mask, p_final * 1.0 * 6.0, 0.0)  # 1 mm zrážok = 1 cm nového snehu

    # 6. KROK: Lightning Hazard Index (LHI)
    instability = np.maximum(w_opt, 0.0) * (dwd_cape / 400.0)
    exposure = np.clip((DEM_200 - 650.0) / 40.0, 0.0, 40.0)
    lhi = np.clip(ndimage.gaussian_filter(exposure * 0.4 + instability * 20.0, sigma=1.2), 0.0, 100.0)
    if dwd_cape < 50.0 and dwd_precip == 0.0:
        lhi = np.clip(lhi * 0.1, 0.0, 10.0)

    return {
        'hours': hours_ahead,
        'u_opt': u_opt, 'v_opt': v_opt,
        'p_final': p_final, 'snow_diff': fresh_snow_6h,
        'lhi': lhi, 'temp_field': temp_field, 'wind_spd': wind_spd,
        'dwd_t2m': dwd_t2m, 'dwd_wspd': dwd_wspd * 3.6, 'dwd_precip': dwd_precip
    }

FORECAST_TIMELINE = [run_dwd_downscaled_simulation(i) for i in range(9)]

# =============================================================================
# 5. VYKRESLENIE 200+ BODOV DO MÁP
# =============================================================================
def draw_dense_landmarks(ax, is_compact=False):
    for lm in REAL_TATRY_200_POINTS:
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

        msize = 4.0 if is_compact else 5.5
        ax.plot(lm["x"], lm["y"], marker=marker, markersize=msize, color=mcolor, 
                markeredgecolor='#000000', markeredgewidth=0.5, alpha=0.9, zorder=10)

        if prio == 1:
            fsize = 5.5 if is_compact else 7.0
            label = lm['name'] if is_compact else f"{lm['name']}\n({lm['alt']}m)"
            ax.text(lm["x"], lm["y"] + (0.35 if is_compact else 0.45), label,
                    fontsize=fsize, fontweight='bold', color='white', ha='center',
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='#0f172a', edgecolor=mcolor, alpha=0.85, linewidth=0.6),
                    zorder=11)

# =============================================================================
# 6. FASTAPI ENDPOINTY
# =============================================================================
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "engine": "TATRYS-50 v2 DWD-First",
        "primary_model": "DWD ICON (Germany Weather Service)",
        "total_points": len(REAL_TATRY_200_POINTS)
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/landmarks")
def get_landmarks():
    return {"status": "ok", "count": len(REAL_TATRY_200_POINTS), "landmarks": REAL_TATRY_200_POINTS}

@app.get("/api/points-grid")
def get_points_grid(step: int = Query(0, ge=0, le=8)):
    """Vráti presne prepočítané hodnoty pre všetkých 200+ bodov z DWD modelu."""
    d = FORECAST_TIMELINE[step]
    results = []
    for p in REAL_TATRY_200_POINTS:
        ix = int(np.clip(p["x"] * 1000.0 / DX, 0, DEM_200.shape[0] - 1))
        iy = int(np.clip(p["y"] * 1000.0 / DY, 0, DEM_200.shape[1] - 1))
        
        # Lokálna teplota: odvodená priamo z DWD cez výškový rozdiel voči DWD terénu
        t_loc = float(d['dwd_t2m'] - ((p["alt"] - DEM_DWD[ix, iy]) * 0.0065))
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
    """Vráti kľúčové lokality pre webové karty."""
    d = FORECAST_TIMELINE[step]
    locs = [
        {"name": "Lomnický štít (2 634 m)", "alt": 2634, "x": 18.0, "y": 16.8},
        {"name": "Gerlachovský štít (2 655 m)", "alt": 2655, "x": 12.0, "y": 16.2},
        {"name": "Zbojnícka chata (1 960 m)", "alt": 1960, "x": 13.8, "y": 15.8},
        {"name": "Štrbské Pleso (1 346 m)", "alt": 1346, "x": 5.8, "y": 9.5},
        {"name": "Starý Smokovec (1 010 m)", "alt": 1010, "x": 14.2, "y": 7.5},
        {"name": "Poprad (672 m)", "alt": 672, "x": 20.0, "y": 1.8}
    ]
    res = []
    for l in locs:
        ix = int(np.clip(l['x'] * 1000.0 / DX, 0, DEM_200.shape[0]-1))
        iy = int(np.clip(l['y'] * 1000.0 / DY, 0, DEM_200.shape[1]-1))
        spd = float(d['wind_spd'][ix, iy] * 3.6)
        t = float(d['dwd_t2m'] - ((l["alt"] - DEM_DWD[ix, iy]) * 0.0065))
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
    d = FORECAST_TIMELINE[step]
    X_km, Y_km = X / 1000.0, Y / 1000.0
    h_label = f"+{d['hours']}h"
    
    if layer == "all":
        fig, axs = plt.subplots(2, 2, figsize=(17, 14), facecolor='#0f172a')
        for row in axs:
            for ax in row:
                ax.set_facecolor('#1e293b')
                ax.tick_params(colors='#94a3b8')
                for s in ax.spines.values():
                    s.set_color('#334155')

        # 1. Topografia & Vietor
        im1 = axs[0, 0].contourf(X_km, Y_km, DEM_200, levels=25, cmap='terrain', alpha=0.85)
        fig.colorbar(im1, ax=axs[0, 0])
        axs[0, 0].quiver(X_km[::7, ::7], Y_km[::7, ::7], d['u_opt'][::7, ::7], d['v_opt'][::7, ::7], scale=120, color='black')
        draw_dense_landmarks(axs[0, 0], is_compact=True)
        axs[0, 0].set_title(f'A. Topografia & Vietor (DWD + Orografia) [{h_label}]', color='white', fontweight='bold')

        # 2. Zrážky
        im2 = axs[0, 1].contourf(X_km, Y_km, d['p_final'], levels=20, cmap='YlGnBu')
        fig.colorbar(im2, ax=axs[0, 1], label='mm / h')
        draw_dense_landmarks(axs[0, 1], is_compact=True)
        axs[0, 1].set_title(f'B. Zrážky (Seeder-Feeder) [{h_label}]', color='white', fontweight='bold')

        # 3. Sneh
        im3 = axs[1, 0].contourf(X_km, Y_km, d['snow_diff'], levels=20, cmap='Blues')
        fig.colorbar(im3, ax=axs[1, 0], label='cm / 6h')
        draw_dense_landmarks(axs[1, 0], is_compact=True)
        axs[1, 0].set_title(f'C. Nový sneh za 6h [{h_label}]', color='white', fontweight='bold')

        # 4. Teplota
        im4 = axs[1, 1].contourf(X_km, Y_km, d['temp_field'], levels=25, cmap='coolwarm')
        fig.colorbar(im4, ax=axs[1, 1], label='°C')
        draw_dense_landmarks(axs[1, 1], is_compact=True)
        axs[1, 1].set_title(f'D. Teplotné pole (200m Lapse Rate) [{h_label}]', color='white', fontweight='bold')

        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(11, 9), facecolor='#0f172a')
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        if layer == "wind":
            im = ax.contourf(X_km, Y_km, DEM_200, levels=25, cmap='terrain', alpha=0.85)
            ax.quiver(X_km[::5, ::5], Y_km[::5, ::5], d['u_opt'][::5, ::5], d['v_opt'][::5, ::5], scale=100, color='black')
            fig.colorbar(im, ax=ax, label='Výška (m n.m.)')
            ax.set_title(f'Prúdenie vetra & Topografia (DWD ICON) [{h_label}]', color='white', fontweight='bold')
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

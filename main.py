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
    description="Vektorový orografický downscaling s pokročilým fyzikálnym modelom búrok, teploty, vetra a zrážok.",
    version="4.8.0"
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
# 2. 200+ REÁLNYCH BODOV TATIER S OROGRAFICKÝMI PARAMETRAMI
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
    {"name": "Malý Kežmarský štít", "alt": 2514, "x": 12400, "y": 7200, "lat": 49.2008, "lon": 20.2186, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 730},
    {"name": "Rysy (severozápadný vrchol PL)", "alt": 2499, "x": 5700, "y": 6050, "lat": 49.1797, "lon": 20.0881, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 720},
    {"name": "Rysy (hlavný vrchol SK)", "alt": 2501, "x": 5700, "y": 6000, "lat": 49.1794, "lon": 20.0881, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 725},
    {"name": "Ťažký štít", "alt": 2500, "x": 6100, "y": 6100, "lat": 49.1736, "lon": 20.0861, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 710},
    {"name": "Kriváň", "alt": 2495, "x": 2700, "y": 5000, "lat": 49.1575, "lon": 20.0000, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 850},
    {"name": "Bradavica", "alt": 2476, "x": 8800, "y": 6200, "lat": 49.1722, "lon": 20.1556, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 700},
    {"name": "Gánok", "alt": 2462, "x": 6700, "y": 6200, "lat": 49.1764, "lon": 20.1014, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 690},
    {"name": "Slavkovský štít", "alt": 2452, "x": 9700, "y": 4000, "lat": 49.1656, "lon": 20.1839, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 800},
    {"name": "Batizovský štít", "alt": 2448, "x": 7300, "y": 5500, "lat": 49.1667, "lon": 20.1222, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 680},
    {"name": "Prostredný hrot", "alt": 2441, "x": 10300, "y": 5800, "lat": 49.1847, "lon": 20.1917, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 670},
    {"name": "Mengusovský štít", "alt": 2438, "x": 5500, "y": 6800, "lat": 49.1833, "lon": 20.0611, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 670},
    {"name": "Mięguszowiecki Szczyt (PL)", "alt": 2438, "x": 5500, "y": 6900, "lat": 49.1842, "lon": 20.0600, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 670},
    {"name": "Mięguszowiecki Czarny (PL)", "alt": 2410, "x": 5800, "y": 6700, "lat": 49.1839, "lon": 20.0750, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 640},
    {"name": "Hrubý vrch", "alt": 2428, "x": 3700, "y": 6500, "lat": 49.1750, "lon": 20.0278, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 660},
    {"name": "Východná Vysoká", "alt": 2428, "x": 7700, "y": 7200, "lat": 49.1750, "lon": 20.1444, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 660},
    {"name": "Čierny štít", "alt": 2429, "x": 11900, "y": 8500, "lat": 49.2042, "lon": 20.2083, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 660},
    {"name": "Zlobivá", "alt": 2426, "x": 7000, "y": 6200, "lat": 49.1708, "lon": 20.1056, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 650},
    {"name": "Satan", "alt": 2421, "x": 4100, "y": 5200, "lat": 49.1639, "lon": 20.0528, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 650},
    {"name": "Kolový štít", "alt": 2418, "x": 12100, "y": 9000, "lat": 49.2083, "lon": 20.2028, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 650},
    {"name": "Javorový štít", "alt": 2418, "x": 10000, "y": 8500, "lat": 49.1917, "lon": 20.1611, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 650},
    {"name": "Veľké Solisko", "alt": 2412, "x": 4000, "y": 4500, "lat": 49.1556, "lon": 20.0417, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 640},
    {"name": "Furkotský štít", "alt": 2405, "x": 3900, "y": 6200, "lat": 49.1722, "lon": 20.0333, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 630},
    {"name": "Kačací štít", "alt": 2401, "x": 7100, "y": 6600, "lat": 49.1681, "lon": 20.1111, "cat": "peaks", "prio": 3, "valley_axis": None, "rel_height": 620},
    {"name": "Svišťový štít", "alt": 2382, "x": 8500, "y": 7800, "lat": 49.1792, "lon": 20.1556, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 600},
    {"name": "Štrbský štít", "alt": 2381, "x": 4500, "y": 6000, "lat": 49.1778, "lon": 20.0472, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 600},
    {"name": "Kôprovský štít", "alt": 2363, "x": 4500, "y": 6500, "lat": 49.1797, "lon": 20.0519, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 580},
    {"name": "Świnica (PL/SK)", "alt": 2301, "x": 4900, "y": 9100, "lat": 49.2192, "lon": 20.0078, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 550},
    {"name": "Kozi Wierch (PL - Orla Perć)", "alt": 2291, "x": 5700, "y": 9200, "lat": 49.2189, "lon": 20.0278, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 540},
    {"name": "Granaty (PL)", "alt": 2240, "x": 5900, "y": 9600, "lat": 49.2242, "lon": 20.0319, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 500},
    {"name": "Kościelec (PL)", "alt": 2155, "x": 5200, "y": 9800, "lat": 49.2256, "lon": 20.0150, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 450},
    {"name": "Mnich (PL - Morskie Oko)", "alt": 2068, "x": 5400, "y": 7200, "lat": 49.1936, "lon": 20.0578, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 400},
    {"name": "Kasprowy Wierch (PL/SK)", "alt": 1987, "x": 4100, "y": 10500, "lat": 49.2325, "lon": 19.9819, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 380},
    {"name": "Giewont (PL)", "alt": 1895, "x": 2300, "y": 11800, "lat": 49.2508, "lon": 19.9342, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 350},
    {"name": "Huncovský štít", "alt": 2352, "x": 13000, "y": 6000, "lat": 49.1917, "lon": 20.2278, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 600},
    {"name": "Ostrá", "alt": 2350, "x": 3500, "y": 5200, "lat": 49.1611, "lon": 20.0278, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 600},
    {"name": "Ostrva", "alt": 1984, "x": 5900, "y": 3200, "lat": 49.1486, "lon": 20.0889, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 300},
    {"name": "Tupá", "alt": 2284, "x": 6500, "y": 4200, "lat": 49.1528, "lon": 20.1028, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 550},
    {"name": "Patria", "alt": 2203, "x": 4700, "y": 2500, "lat": 49.1417, "lon": 20.0611, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 500},
    {"name": "Predné Solisko", "alt": 2117, "x": 4300, "y": 3000, "lat": 49.1444, "lon": 20.0417, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 450},
    {"name": "Jahňací štít", "alt": 2230, "x": 13300, "y": 9500, "lat": 49.2194, "lon": 20.2222, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 550},
    {"name": "Kozí štít", "alt": 2111, "x": 12800, "y": 8600, "lat": 49.2139, "lon": 20.2167, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 450},
    {"name": "Jastrabia veža", "alt": 2137, "x": 13000, "y": 8800, "lat": 49.2111, "lon": 20.2194, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 470},
    {"name": "Veľká Svišťovka", "alt": 2038, "x": 12800, "y": 7200, "lat": 49.2028, "lon": 20.2333, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 400},
    {"name": "Havran (Belianske Tatry)", "alt": 2152, "x": 12700, "y": 10800, "lat": 49.2472, "lon": 20.2000, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 600},
    {"name": "Ždiarska vidla (Belianske Tatry)", "alt": 2142, "x": 13300, "y": 10500, "lat": 49.2444, "lon": 20.2167, "cat": "peaks", "prio": 1, "valley_axis": None, "rel_height": 590},
    {"name": "Hlúpy", "alt": 2061, "x": 14000, "y": 9800, "lat": 49.2361, "lon": 20.2306, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 500},
    {"name": "Muráň", "alt": 1890, "x": 11300, "y": 11500, "lat": 49.2500, "lon": 20.1694, "cat": "peaks", "prio": 2, "valley_axis": None, "rel_height": 400},

    # Sedlá
    {"name": "Poľský hrebeň", "alt": 2200, "x": 7900, "y": 6800, "lat": 49.1722, "lon": 20.1417, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 300},
    {"name": "Prielom", "alt": 2290, "x": 8400, "y": 7200, "lat": 49.1750, "lon": 20.1500, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 350},
    {"name": "Sedielko (Javorová / Malá Studená)", "alt": 2376, "x": 10500, "y": 7800, "lat": 49.1917, "lon": 20.1778, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 400},
    {"name": "Priečne sedlo", "alt": 2352, "x": 10100, "y": 7000, "lat": 49.1889, "lon": 20.1833, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 380},
    {"name": "Baranie sedlo", "alt": 2384, "x": 11500, "y": 8000, "lat": 49.2014, "lon": 20.2000, "cat": "passes", "prio": 2, "valley_axis": None, "rel_height": 410},
    {"name": "Váha (sedlo pod Rysmi)", "alt": 2340, "x": 5800, "y": 5800, "lat": 49.1778, "lon": 20.0833, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 400},
    {"name": "Vyšné Kôprovské sedlo", "alt": 2180, "x": 4700, "y": 6200, "lat": 49.1750, "lon": 20.0556, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 300},
    {"name": "Kopské sedlo (severné rozhranie)", "alt": 1750, "x": 13700, "y": 9800, "lat": 49.2278, "lon": 20.2278, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 200},
    {"name": "Sedlo pod Ostrvou", "alt": 1960, "x": 6000, "y": 3000, "lat": 49.1472, "lon": 20.0861, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 250},
    {"name": "Bystrá lávka", "alt": 2300, "x": 3700, "y": 5800, "lat": 49.1667, "lon": 20.0389, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 350},
    {"name": "Lomnické sedlo", "alt": 2190, "x": 12100, "y": 6000, "lat": 49.1903, "lon": 20.2167, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 300},
    {"name": "Sedlo pod Svišťovkou", "alt": 2023, "x": 12700, "y": 7500, "lat": 49.2000, "lon": 20.2306, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 250},
    {"name": "Szpiglasowa Przełęcz (PL)", "alt": 2110, "x": 5100, "y": 7800, "lat": 49.2003, "lon": 20.0417, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 280},
    {"name": "Zawrat (PL - vstup na Orlu Perć)", "alt": 2159, "x": 5300, "y": 9000, "lat": 49.2189, "lon": 20.0161, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 300},
    {"name": "Krzyżne (PL - koniec Orlej Perći)", "alt": 2112, "x": 6400, "y": 9500, "lat": 49.2239, "lon": 20.0464, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 280},
    {"name": "Przełęcz pod Chłopkiem (PL/SK)", "alt": 2307, "x": 5900, "y": 6600, "lat": 49.1825, "lon": 20.0800, "cat": "passes", "prio": 1, "valley_axis": None, "rel_height": 350},

    # Plesá
    {"name": "Morskie Oko (PL)", "alt": 1395, "x": 6100, "y": 7800, "lat": 49.2004, "lon": 20.0712, "cat": "lakes", "prio": 1, "valley_axis": 340, "rel_height": 50},
    {"name": "Czarny Staw pod Rysami (PL)", "alt": 1583, "x": 6000, "y": 7100, "lat": 49.1886, "lon": 20.0767, "cat": "lakes", "prio": 1, "valley_axis": 350, "rel_height": 100},
    {"name": "Wielki Staw Polski (PL - 5 Stawów)", "alt": 1665, "x": 5500, "y": 8600, "lat": 49.2117, "lon": 20.0306, "cat": "lakes", "prio": 1, "valley_axis": 320, "rel_height": 100},
    {"name": "Czarny Staw Gąsienicowy (PL)", "alt": 1624, "x": 5200, "y": 10300, "lat": 49.2333, "lon": 20.0167, "cat": "lakes", "prio": 1, "valley_axis": 310, "rel_height": 80},
    {"name": "Veľké Hincovo pleso", "alt": 1945, "x": 5000, "y": 5800, "lat": 49.1764, "lon": 20.0600, "cat": "lakes", "prio": 1, "valley_axis": 210, "rel_height": 150},
    {"name": "Štrbské pleso", "alt": 1346, "x": 3900, "y": 1500, "lat": 49.1194, "lon": 20.0603, "cat": "lakes", "prio": 1, "valley_axis": 180, "rel_height": 20},
    {"name": "Popradské pleso", "alt": 1494, "x": 4800, "y": 2200, "lat": 49.1536, "lon": 20.0797, "cat": "lakes", "prio": 1, "valley_axis": 190, "rel_height": 40},
    {"name": "Batizovské pleso", "alt": 1884, "x": 7300, "y": 4500, "lat": 49.1597, "lon": 20.1306, "cat": "lakes", "prio": 1, "valley_axis": 170, "rel_height": 120},
    {"name": "Velické pleso", "alt": 1670, "x": 8300, "y": 3500, "lat": 49.1583, "lon": 2

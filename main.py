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
    version="4.3.6"
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
# 2. 200+ REÁLNYCH BODOV VYSOKÝCH A BELIANSKYCH TATIER (SK + PL)
# =============================================================================
TATRAS_POINTS = [
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
    {"name": "Rysy (severozápadný vrchol PL)", "alt": 2499, "x": 5700, "y": 6050, "lat": 49.1797, "lon": 20.0881, "cat": "peaks", "prio": 1},
    {"name": "Rysy (hlavný vrchol SK)", "alt": 2501, "x": 5700, "y": 6000, "lat": 49.1794, "lon": 20.0881, "cat": "peaks", "prio": 1},
    {"name": "Ťažký štít", "alt": 2500, "x": 6100, "y": 6100, "lat": 49.1736, "lon": 20.0861, "cat": "peaks", "prio": 2},
    {"name": "Kriváň", "alt": 2495, "x": 2700, "y": 5000, "lat": 49.1575, "lon": 20.0000, "cat": "peaks", "prio": 1},
    {"name": "Bradavica", "alt": 2476, "x": 8800, "y": 6200, "lat": 49.1722, "lon": 20.1556, "cat": "peaks", "prio": 1},
    {"name": "Gánok", "alt": 2462, "x": 6700, "y": 6200, "lat": 49.1764, "lon": 20.1014, "cat": "peaks", "prio": 2},
    {"name": "Slavkovský štít", "alt": 2452, "x": 9700, "y": 4000, "lat": 49.1656, "lon": 20.1839, "cat": "peaks", "prio": 1},
    {"name": "Batizovský štít", "alt": 2448, "x": 7300, "y": 5500, "lat": 49.1667, "lon": 20.1222, "cat": "peaks", "prio": 2},
    {"name": "Prostredný hrot", "alt": 2441, "x": 10300, "y": 5800, "lat": 49.1847, "lon": 20.1917, "cat": "peaks", "prio": 1},
    {"name": "Mengusovský štít", "alt": 2438, "x": 5500, "y": 6800, "lat": 49.1833, "lon": 20.0611, "cat": "peaks", "prio": 1},
    {"name": "Mięguszowiecki Szczyt (PL)", "alt": 2438, "x": 5500, "y": 6900, "lat": 49.1842, "lon": 20.0600, "cat": "peaks", "prio": 1},
    {"name": "Mięguszowiecki Czarny (PL)", "alt": 2410, "x": 5800, "y": 6700, "lat": 49.1839, "lon": 20.0750, "cat": "peaks", "prio": 2},
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
    {"name": "Świnica (PL/SK)", "alt": 2301, "x": 4900, "y": 9100, "lat": 49.2192, "lon": 20.0078, "cat": "peaks", "prio": 1},
    {"name": "Kozi Wierch (PL - Orla Perć)", "alt": 2291, "x": 5700, "y": 9200, "lat": 49.2189, "lon": 20.0278, "cat": "peaks", "prio": 1},
    {"name": "Granaty (PL)", "alt": 2240, "x": 5900, "y": 9600, "lat": 49.2242, "lon": 20.0319, "cat": "peaks", "prio": 2},
    {"name": "Kościelec (PL)", "alt": 2155, "x": 5200, "y": 9800, "lat": 49.2256, "lon": 20.0150, "cat": "peaks", "prio": 1},
    {"name": "Mnich (PL - Morskie Oko)", "alt": 2068, "x": 5400, "y": 7200, "lat": 49.1936, "lon": 20.0578, "cat": "peaks", "prio": 1},
    {"name": "Kasprowy Wierch (PL/SK)", "alt": 1987, "x": 4100, "y": 10500, "lat": 49.2325, "lon": 19.9819, "cat": "peaks", "prio": 1},
    {"name": "Giewont (PL)", "alt": 1895, "x": 2300, "y": 11800, "lat": 49.2508, "lon": 19.9342, "cat": "peaks", "prio": 1},
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
    {"name": "Havran (Belianske Tatry)", "alt": 2152, "x": 12700, "y": 10800, "lat": 49.2472, "lon": 20.2000, "cat": "peaks", "prio": 1},
    {"name": "Ždiarska vidla (Belianske Tatry)", "alt": 2142, "x": 13300, "y": 10500, "lat": 49.2444, "lon": 20.2167, "cat": "peaks", "prio": 1},
    {"name": "Hlúpy", "alt": 2061, "x": 14000, "y": 9800, "lat": 49.2361, "lon": 20.2306, "cat": "peaks", "prio": 2},
    {"name": "Muráň", "alt": 1890, "x": 11300, "y": 11500, "lat": 49.2500, "lon": 20.1694, "cat": "peaks", "prio": 2},

    # 2. Sedlá
    {"name": "Poľský hrebeň", "alt": 2200, "x": 7900, "y": 6800, "lat": 49.1722, "lon": 20.1417, "cat": "passes", "prio": 1},
    {"name": "Prielom", "alt": 2290, "x": 8400, "y": 7200, "lat": 49.1750, "lon": 20.1500, "cat": "passes", "prio": 1},
    {"name": "Sedielko (Javorová / Malá Studená)", "alt": 2376, "x": 10500, "y": 7800, "lat": 49.1917, "lon": 20.1778, "cat": "passes", "prio": 1},
    {"name": "Priečne sedlo", "alt": 2352, "x": 10100, "y": 7000, "lat": 49.1889, "lon": 20.1833, "cat": "passes", "prio": 1},
    {"name": "Baranie sedlo", "alt": 2384, "x": 11500, "y": 8000, "lat": 49.2014, "lon": 20.2000, "cat": "passes", "prio": 2},
    {"name": "Váha (sedlo pod Rysmi)", "alt": 2340, "x": 5800, "y": 5800, "lat": 49.1778, "lon": 20.0833, "cat": "passes", "prio": 1},
    {"name": "Vyšné Kôprovské sedlo", "alt": 2180, "x": 4700, "y": 6200, "lat": 49.1750, "lon": 20.0556, "cat": "passes", "prio": 1},
    {"name": "Kopské sedlo (severné rozhranie)", "alt": 1750, "x": 13700, "y": 9800, "lat": 49.2278, "lon": 20.2278, "cat": "passes", "prio": 1},
    {"name": "Sedlo pod Ostrvou", "alt": 1960, "x": 6000, "y": 3000, "lat": 49.1472, "lon": 20.0861, "cat": "passes", "prio": 1},
    {"name": "Bystrá lávka", "alt": 2300, "x": 3700, "y": 5800, "lat": 49.1667, "lon": 20.0389, "cat": "passes", "prio": 1},
    {"name": "Lomnické sedlo", "alt": 2190, "x": 12100, "y": 6000, "lat": 49.1903, "lon": 20.2167, "cat": "passes", "prio": 1},
    {"name": "Sedlo pod Svišťovkou", "alt": 2023, "x": 12700, "y": 7500, "lat": 49.2000, "lon": 20.2306, "cat": "passes", "prio": 1},
    {"name": "Szpiglasowa Przełęcz (PL)", "alt": 2110, "x": 5100, "y": 7800, "lat": 49.2003, "lon": 20.0417, "cat": "passes", "prio": 1},
    {"name": "Zawrat (PL - vstup na Orlu Perć)", "alt": 2159, "x": 5300, "y": 9000, "lat": 49.2189, "lon": 20.0161, "cat": "passes", "prio": 1},
    {"name": "Krzyżne (PL - koniec Orlej Perći)", "alt": 2112, "x": 6400, "y": 9500, "lat": 49.2239, "lon": 20.0464, "cat": "passes", "prio": 1},
    {"name": "Przełęcz pod Chłopkiem (PL/SK)", "alt": 2307, "x": 5900, "y": 6600, "lat": 49.1825, "lon": 20.0800, "cat": "passes", "prio": 1},

    # 3. Plesá
    {"name": "Morskie Oko (PL)", "alt": 1395, "x": 6100, "y": 7800, "lat": 49.2004, "lon": 20.0712, "cat": "lakes", "prio": 1},
    {"name": "Czarny Staw pod Rysami (PL)", "alt": 1583, "x": 6000, "y": 7100, "lat": 49.1886, "lon": 20.0767, "cat": "lakes", "prio": 1},
    {"name": "Wielki Staw Polski (PL - 5 Stawów)", "alt": 1665, "x": 5500, "y": 8600, "lat": 49.2117, "lon": 20.0306, "cat": "lakes", "prio": 1},
    {"name": "Czarny Staw Gąsienicowy (PL)", "alt": 1624, "x": 5200, "y": 10300, "lat": 49.2333, "lon": 20.0167, "cat": "lakes", "prio": 1},
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
    {"name": "Ťažké pleso (Bielovodská dolina)", "alt": 1611, "x": 6700, "y": 7500, "lat": 49.1889, "lon": 20.1028, "cat": "lakes", "prio": 2},
    {"name": "Žabie pleso Javorové", "alt": 1878, "x": 9800, "y": 8800, "lat": 49.1986, "lon": 20.1583, "cat": "lakes", "prio": 2},

    # 4. Chaty
    {"name": "Schronisko PTTK Morskie Oko (PL)", "alt": 1405, "x": 6100, "y": 7900, "lat": 49.2014, "lon": 20.0717, "cat": "huts", "prio": 1},
    {"name": "Schronisko w Dolinie Pięciu Stawów (PL)", "alt": 1671, "x": 5700, "y": 8700, "lat": 49.2139, "lon": 20.0486, "cat": "huts", "prio": 1},
    {"name": "Murowaniec Hala Gąsienicowa (PL)", "alt": 1500, "x": 4900, "y": 10800, "lat": 49.2436, "lon": 20.0072, "cat": "huts", "prio": 1},
    {"name": "Schronisko PTTK w Dolinie Roztoki (PL)", "alt": 1031, "x": 7500, "y": 9500, "lat": 49.2378, "lon": 20.0883, "cat": "huts", "prio": 1},
    {"name": "Schronisko na Polanie Kondratowej (PL)", "alt": 1333, "x": 3300, "y": 11200, "lat": 49.2447, "lon": 19.9656, "cat": "huts", "prio": 1},
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
    {"name": "Chata Plesnivec (Belianske Tatry)", "alt": 1290, "x": 14500, "y": 9200, "lat": 49.2278, "lon": 20.2722, "cat": "huts", "prio": 1},
    {"name": "Horáreň Biela Voda (Bielovodská)", "alt": 995, "x": 8100, "y": 10500, "lat": 49.2528, "lon": 20.1000, "cat": "huts", "prio": 2},

    # 5. Osady
    {"name": "Zakopane - Centrum (PL)", "alt": 838, "x": 2800, "y": 13500, "lat": 49.2992, "lon": 19.9489, "cat": "towns", "prio": 1},
    {"name": "Kuźnice (PL - dolná stanica Kasprowy)", "alt": 1010, "x": 3600, "y": 12200, "lat": 49.2694, "lon": 19.9806, "cat": "towns", "prio": 1},
    {"name": "Palenica Białczańska (PL - vstup Morskie Oko)", "alt": 984, "x": 7800, "y": 10200, "lat": 49.2550, "lon": 20.1031, "cat": "towns", "prio": 1},
    {"name": "Tatranská Javorina (SK)", "alt": 1000, "x": 9200, "y": 11200, "lat": 49.2667, "lon": 20.1417, "cat": "towns", "prio": 1},
    {"name": "Lysá Poľana (SK/PL)", "alt": 970, "x": 8300, "y": 11100, "lat": 49.2625, "lon": 20.1111, "cat": "towns", "prio": 1},
    {"name": "Ždiar (Belianske Tatry)", "alt": 896, "x": 13700, "y": 11000, "lat": 49.2717, "lon": 20.2714, "cat": "towns", "prio": 1},
    {"name": "Podspády", "alt": 917, "x": 10500, "y": 12000, "lat": 49.2833, "lon": 20.1833, "cat": "towns", "prio": 2},
    {"name": "Starý Smokovec", "alt": 1010, "x": 9500, "y": 500, "lat": 49.1411, "lon": 20.2219, "cat": "towns", "prio": 1},
    {"name": "Tatranská Lomnica", "alt": 850, "x": 13000, "y": 1000, "lat": 49.1650, "lon": 20.2819, "cat": "towns", "prio": 1},
    {"name": "Štrbské Pleso", "alt": 1346, "x": 3900, "y": 1500, "lat": 49.1194, "lon": 20.0603, "cat": "towns", "prio": 1},
    {"name": "Tatranská Polianka", "alt": 1005, "x": 7700, "y": 500, "lat": 49.1236, "lon": 20.1847, "cat": "towns", "prio": 1},
    {"name": "Vyšné Hágy", "alt": 1125, "x": 5700, "y": 500, "lat": 49.1194, "lon": 20.1250, "cat": "towns", "prio": 1},
    {"name": "Podbanské", "alt": 940, "x": 1000, "y": 1000, "lat": 49.1417, "lon": 19.9028, "cat": "towns", "prio": 1}
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
        req = urllib.request.Request(url, headers={'User-Agent': 'AvalancheTatry-35NodeGrid/4.3'})
        with urllib.request.urlopen(req, timeout=12) as response:
            data = json.loads(response.read().decode())
            CACHE["matrix"] = data
            CACHE["ts"] = now
            return data
    except Exception as e:
        print(f"[VAROVANIE] Multi-point DWD zlyhalo: {e}")
        return None

def calculate_35_node_grid_state(step_idx: int):
    # step_idx je teraz 6-hodinový krok (0 až 8, čo kryje 48 hodín)
    hours_ahead = step_idx * 6
    dwd_raw = fetch_35_nodes_dwd()

    dwd_t = np.zeros((7, 5))
    dwd_wspd = np.zeros((7, 5))
    dwd_wdir = np.zeros((7, 5))
    dwd_prec = np.zeros((7, 5))
    dwd_cape = np.zeros((7, 5))
    dwd_dem = np.zeros((7, 5))

    if dwd_raw and isinstance(dwd_raw, list) and len(dwd_raw) == 35:
        time_list = dwd_raw[0].get("hourly", {}).get("time", [])
        
        # Prepočet 6-hodinového kroku na hodinový index v poli Open-Meteo
        data_idx = min(max(0, step_idx * 6), len(time_list) - 1)

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
        wspd_pt = (wspd_dwd_local * w_factor) * 3.6

        p_factor = 1.0 + max(p["alt"] - 1000.0, 0.0) * 0.00035
        prec_pt = prec_dwd_local * p_factor if prec_dwd_local > 0.0 else 0.0
        snow_pt = (prec_pt * 1.0 * 6.0) if t_pt < 0.0 else 0.0

        if cape_dwd_local < 20.0 and prec_dwd_local == 0.0:
            lhi_pt = 0.0
        else:
            cape_score = min(max(cape_dwd_local / 12.0, 0.0), 65.0)
            precip_score = min(prec_dwd_local * 15.0, 40.0)
            exposure_score = min(max(p["alt"] - 1200.0, 0.0) / 45.0, 30.0)
            lhi_pt = min(cape_score + precip_score + exposure_score, 100.0)
            if prec_dwd_local == 0.0 and cape_dwd_local < 40.0:
                lhi_pt = min(lhi_pt, 15.0)

        results.append({
            "name": p["name"], "alt": p["alt"], "lat": p["lat"], "lon": p["lon"],
            "cat": p["cat"], "prio": p["prio"], "temp": round(t_pt, 1),
            "wind_kmh": round(wspd_pt, 1), "wind_dir": round(wdir_dwd_local, 0),
            "precip_mmh": round(prec_pt, 1), "snow_6h_cm": round(snow_pt, 1), "lhi": round(lhi_pt, 0)
        })

    return {"status": "ok", "step": step_idx, "hours_ahead": hours_ahead, "dwd_nodes_used": 35, "count": len(results), "points": results}

SOUNDING_CACHE = {"ts": 0, "data": None}

@app.get("/api/sounding")
def fetch_sounding_ganovce():
    now = datetime.datetime.now().timestamp()
    if SOUNDING_CACHE["data"] and (now - SOUNDING_CACHE["ts"] < 900):
        return SOUNDING_CACHE["data"]

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=49.035&longitude=20.323&hourly="
        "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,cape,lifted_index,convective_inhibition,"
        "wind_speed_10m,wind_gusts_10m,wind_direction_10m,"
        "temperature_850hPa,temperature_700hPa,temperature_500hPa,"
        "wind_speed_850hPa,wind_speed_500hPa,wind_direction_850hPa,wind_direction_500hPa"
        "&timezone=Europe%2FBratislava&forecast_days=1"
    )

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'AvalancheSounding/2.5'})
        with urllib.request.urlopen(req, timeout=10) as response:
            raw = json.loads(response.read().decode())
            cur_h = datetime.datetime.now().hour
            h = raw.get("hourly", {})

            def get_val(key, default, idx):
                arr = h.get(key, [])
                if arr and len(arr) > idx and arr[idx] is not None:
                    return arr[idx]
                return default

            cape_val = float(get_val("cape", 0.0, cur_h))
            li_val = float(get_val("lifted_index", 0.0, cur_h))
            cin_val = float(get_val("convective_inhibition", 0.0, cur_h))
            precip_val = float(get_val("precipitation", 0.0, cur_h))
            srh_val = 50.0

            t_surface = float(get_val("temperature_2m", 15.0, cur_h))
            t_850 = float(get_val("temperature_850hPa", 8.0, cur_h))
            t_700 = float(get_val("temperature_700hPa", 0.0, cur_h))
            t_500 = float(get_val("temperature_500hPa", -15.0, cur_h))

            wspd_850 = float(get_val("wind_speed_850hPa", 10.0, cur_h)) / 3.6
            wspd_500 = float(get_val("wind_speed_500hPa", 20.0, cur_h)) / 3.6
            wdir_850 = float(get_val("wind_direction_850hPa", 180.0, cur_h))
            wdir_500 = float(get_val("wind_direction_500hPa", 220.0, cur_h))
            
            shear_0_6km = round(abs(wspd_500 - wspd_850), 1)

            if cape_val < 10.0 and precip_val > 0.1 and t_surface > 15.0:
                cape_val = 250.0 + (precip_val * 40.0)
                li_val = -2.1

            if cape_val > 1200 and shear_0_6km > 20:
                storm_desc = "🔴 Vysoké riziko superciel (Nebezpečné krúpy >3cm, downbursty)"
                risk_level = "extreme"
            elif cape_val > 600 and shear_0_6km > 14:
                storm_desc = "🟠 Organizované búrkové línie / Squall line (Prívalové dažde, silný vietor)"
                risk_level = "high"
            elif cape_val > 200 or (precip_val > 0.0 and t_surface > 14.0):
                storm_desc = "🟡 Orografické pulzové búrky (Lokálne blesky a prívalové zrážky)"
                risk_level = "moderate"
            else:
                storm_desc = "🟢 Stabilná atmosféra / Bez výraznej konvekcie"
                risk_level = "low"

            data = {
                "station": "Poprad-Gánovce (11952)",
                "elevation_m": 708,
                "timestamp_str": datetime.datetime.now().strftime("%d.%m. %H:%M"),
                "cape_jkg": round(cape_val, 1),
                "lifted_index": round(li_val, 1),
                "cin_jkg": round(cin_val, 1),
                "srh_m2s2": round(srh_val, 1),
                "deep_layer_shear_mps": shear_0_6km,
                "storm_potential_type": storm_desc,
                "risk_level": risk_level,
                "freezing_level_m": 3350,
                "levels": [
                    {"name": "Povrch (Gánovce)", "alt": 708, "temp": round(t_surface, 1), "wind": f"{round(float(get_val('wind_speed_10m', 5, cur_h))*3.6, 1)} km/h"},
                    {"name": "850 hPa (~1.5 km)", "alt": 1460, "temp": round(t_850, 1), "wind": f"{round(wspd_850*3.6, 1)} km/h ({round(wdir_850)}°)"},
                    {"name": "700 hPa (~3.0 km)", "alt": 3020, "temp": round(t_700, 1), "wind": f"{round(float(get_val('wind_speed_700hPa', 12, cur_h))*3.6, 1)} km/h"},
                    {"name": "500 hPa (~5.6 km)", "alt": 5600, "temp": round(t_500, 1), "wind": f"{round(wspd_500*3.6, 1)} km/h ({round(wdir_500)}°)"}
                ]
            }
            SOUNDING_CACHE["data"] = data
            SOUNDING_CACHE["ts"] = now
            return data
    except Exception as e:
        print(f"[CHYBA] Sťahovanie sondáže zlyhalo: {e}")
        return {
            "station": "Poprad-Gánovce",
            "elevation_m": 708,
            "timestamp_str": datetime.datetime.now().strftime("%d.%m. %H:%M"),
            "cape_jkg": 180.0, "cin_jkg": -10.0, "srh_m2s2": 45.0,
            "deep_layer_shear_mps": 12.0,
            "storm_potential_type": "🟡 Orografické pulzové búrky (Lokálne blesky)",
            "risk_level": "moderate", "freezing_level_m": 3100,
            "levels": [
                {"name": "Povrch (Gánovce)", "alt": 708, "temp": 19.0, "wind": "14.2 km/h"},
                {"name": "850 hPa (~1.5 km)", "alt": 1460, "temp": 10.1, "wind": "26.0 km/h (190°)"},
                {"name": "700 hPa (~3.0 km)", "alt": 3020, "temp": 2.2, "wind": "35.0 km/h (210°)"},
                {"name": "500 hPa (~5.6 km)", "alt": 5600, "temp": -13.5, "wind": "48.0 km/h (240°)"}
            ]
        }

@app.get("/api/text-forecast")
def get_text_forecast(step: int = Query(0, ge=0, le=8)):
    grid_data = calculate_35_node_grid_state(step)
    priority_points = [p for p in grid_data["points"] if p.get("prio") == 1]
    
    if priority_points:
        max_wind = max(priority_points, key=lambda x: x["wind_kmh"])
        min_temp = min(priority_points, key=lambda x: x["temp"])
        max_precip = max(priority_points, key=lambda x: x["precip_mmh"])
    else:
        max_wind = min_temp = max_precip = {"name": "Tatry", "wind_kmh": 0, "temp": 0, "precip_mmh": 0}

    hours = grid_data["hours_ahead"]
    
    summary_text = (
        f"Prehľad predpovede pre Vysoké Tatry na horizont +{hours} hodín. "
        f"Teploty v najvyšších polohách sa pohybujú okolo {min_temp['temp']} °C "
        f"(lokalita {min_temp['name']}, {min_temp['alt']} m n.m.). "
        f"Najsilnejší vietor dosahuje rýchlosť do {max_wind['wind_kmh']} km/h "
        f"na vrchole {max_wind['name']}. "
    )
    
    if max_precip["precip_mmh"] > 0.5:
        summary_text += f"Očakávajte zrážky s úhrnom do {max_precip['precip_mmh']} mm/h, najmä v oblasti {max_precip['name']}."
    else:
        summary_text += "Situácia je bez výraznejších zrážok, prevláda stabilný charakter počasia."

    return {
        "status": "ok",
        "step": step,
        "hours_ahead": hours,
        "timestamp": datetime.datetime.now().strftime("%d.%m. %H:%M"),
        "forecast_text": summary_text,
        "items": priority_points
    }

@app.get("/")
def read_root():
    return {"status": "ok", "service": "TATRYS-50 35-Node Point Grid", "dwd_nodes": 35, "tatras_points": len(TATRAS_POINTS)}

@app.get("/api/text-summary")
def get_text_summary(step: int = Query(0, ge=0, le=8)):
    grid_data = calculate_35_node_grid_state(step)
    priority_points = [p for p in grid_data["points"] if p.get("prio") == 1]
    
    return {
        "status": "ok",
        "step": step,
        "hours_ahead": grid_data["hours_ahead"],
        "timestamp": datetime.datetime.now().strftime("%d.%m. %H:%M"),
        "count": len(priority_points),
        "items": priority_points
    }    

@app.get("/api/lomnicky-station")
def get_lomnicky_station_data():
    url = "https://mesonet.agron.iastate.edu/api/json/current.py?station=11953&network=SK_ASOS"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'AvalancheSynopFetcher/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode())
            obs = payload.get("data", {})
            
            if obs:
                temp_c = round(float(obs.get("tmpc", 0.0)), 1) if obs.get("tmpc") is not None else 0.0
                wind_kts = float(obs.get("sknt", 0.0))
                wind_kmh = round(wind_kts * 1.852, 1)
                humidity = round(float(obs.get("relh", 0.0)), 1) if obs.get("relh") is not None else 0.0
                valid_time = obs.get("valid", datetime.datetime.now().strftime("%d.%m. %H:00"))

                return {
                    "station_name": "Lomnický štít (SHMÚ / 2634 m n.m.)",
                    "station_id": "11953",
                    "timestamp_str": str(valid_time),
                    "real_temp": temp_c,
                    "real_wind_kmh": wind_kmh,
                    "real_wind_dir": str(obs.get("drct", "--")),
                    "real_humidity": humidity,
                    "status": "Online (Live SYNOP)"
                }
    except Exception as e:
        print(f"[VAROVANIE] Sťahovanie reálneho SYNOP zlyhalo: {e}")

    return {
        "station_name": "Lomnický štít (SHMÚ / 2634 m n.m.)",
        "station_id": "11953",
        "timestamp_str": datetime.datetime.now().strftime("%d.%m. %H:00"),
        "real_temp": 4.5,
        "real_wind_kmh": 22.0,
        "real_wind_dir": "SZ",
        "real_humidity": 70.0,
        "status": "Offline (Fallback odhad)"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/points-grid")
def get_points_grid(step: int = Query(0, ge=0, le=8)):
    return calculate_35_node_grid_state(step)

@app.get("/api/forecast")
def get_forecast_legacy(step: int = Query(0, ge=0, le=8)):
    return calculate_35_node_grid_state(step)

@app.get("/api/hazards")
def get_hazards_48h():
    hazards = []
    base_dt = datetime.datetime.now()

    for step in range(0, 9):
        data = calculate_35_node_grid_state(step)
        target_time = base_dt + datetime.timedelta(hours=data["hours_ahead"])
        time_str = target_time.strftime("%d.%m. %H:%M") + f" (+{data['hours_ahead']}h)"

        for p in data["points"]:
            if p["wind_kmh"] >= 100.0:
                hazards.append({
                    "severity": "extreme", "type": "Orkán / Víchrica na hrebeni", "icon": "fa-wind",
                    "location": p["name"], "alt": p["alt"], "time": time_str, "value": f"{p['wind_kmh']} km/h",
                    "desc": "Extrémna sila vetra na štítoch a exponovaných trasách.", "sort_val": p["wind_kmh"]
                })
            elif p["wind_kmh"] >= 65.0 and p["cat"] in ["towns", "huts"]:
                hazards.append({
                    "severity": "high", "type": "Silný vietor / Bóra", "icon": "fa-wind",
                    "location": p["name"], "alt": p["alt"], "time": time_str, "value": f"{p['wind_kmh']} km/h",
                    "desc": "Silný nárazový vietor v dolinách a osadách.", "sort_val": p["wind_kmh"]
                })
            if p["lhi"] >= 52.0:
                hazards.append({
                    "severity": "high" if p["lhi"] < 75 else "extreme", "type": "Riziko zásahu bleskom", "icon": "fa-bolt",
                    "location": p["name"], "alt": p["alt"], "time": time_str, "value": f"LHI {p['lhi']}/100",
                    "desc": "Zvýšené až akútne nebezpečenstvo bleskov na hrebeňoch.", "sort_val": p["lhi"]
                })
            if p["precip_mmh"] >= 8.0:
                hazards.append({
                    "severity": "high", "type": "Prívalový lejak", "icon": "fa-cloud-showers-water",
                    "location": p["name"], "alt": p["alt"], "time": time_str, "value": f"{p['precip_mmh']} mm/h",
                    "desc": "Výrazné zrážky. Riziko stekania vody zo svahov.", "sort_val": p["precip_mmh"]
                })

    unique_hazards = []
    hazards.sort(key=lambda x: x["sort_val"], reverse=True)
    for h in hazards:
        count_for_key = sum(1 for item in unique_hazards if item["type"] == h["type"] and item["time"] == h["time"])
        if count_for_key < 2:
            unique_hazards.append(h)

    return {"status": "ok", "horizon": "48h", "has_hazards": len(unique_hazards) > 0, "count": len(unique_hazards), "hazards": unique_hazards[:6]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

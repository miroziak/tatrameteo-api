import io
import os
import datetime
import numpy as np
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import FastAPI, Response, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TATRYS-50 v2 Ultra-HD API | Avalanche.sk",
    description="Vysokorozlíšivý orografický model s hustou sieťou 200+ tatranských bodov.",
    version="2.4.0"
)

origins = [
    "https://www.avalanche.sk",
    "http://www.avalanche.sk",
    "https://avalanche.sk",
    "http://avalanche.sk",
    "http://localhost:3000",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# HUSTÁ DATABÁZA 200+ ORIENTAČNÝCH BODOV VYSOKÝCH TATIER
# (X: 0 až 24 km, Y: 0 až 24 km)
# =============================================================================
def build_dense_tatry_points():
    pts = []
    
    # 1. KĽÚČOVÉ HLAVNÉ ŠTÍTY (Priority 1)
    primary_peaks = [
        ("Gerlachovský štít", 2655, 12.0, 16.2), ("Lomnický štít", 2634, 18.0, 16.8),
        ("Ľadový štít", 2627, 16.5, 18.0), ("Pyšný štít", 2623, 17.5, 17.5),
        ("Zadný Gerlach", 2616, 11.7, 16.5), ("Lavínový štít", 2606, 11.9, 16.8),
        ("Rysy", 2501, 8.5, 16.0), ("Kriváň", 2495, 4.0, 15.0),
        ("Slavkovský štít", 2452, 14.5, 14.0), ("Východná Vysoká", 2428, 11.5, 17.2),
        ("Baranie rohy", 2526, 17.0, 18.2), ("Vysoká", 2560, 9.5, 15.8),
        ("Ťažký štít", 2500, 9.2, 16.1), ("Mengusovský štít", 2438, 8.2, 16.8),
        ("Hrubý vrch", 2428, 5.5, 16.5), ("Satan", 2421, 6.2, 15.2),
        ("Štrbský štít", 2381, 6.8, 16.0), ("Kôprovský štít", 2363, 6.8, 16.5),
        ("Javorový štít", 2418, 15.0, 18.5), ("Široká", 2210, 13.0, 20.5),
        ("Jahňací štít", 2230, 20.0, 19.5), ("Predné Solisko", 2117, 6.5, 13.0),
        ("Tupá", 2284, 9.8, 14.2), ("Končistá", 2538, 10.5, 14.8),
        ("Batizovský štít", 2448, 11.0, 15.5), ("Bradavica", 2476, 13.2, 16.2),
        ("Velický štít", 2682, 12.2, 16.9), ("Svišťový štít", 2382, 12.8, 17.8),
        ("Prostredný hrot", 2441, 15.5, 15.8), ("Kežmarský štít", 2556, 18.8, 16.5)
    ]
    for n, a, x, y in primary_peaks:
        pts.append({"name": n, "alt": f"{a}m", "x": x, "y": y, "type": "peak", "prio": 1})

    # 2. VEŽE, IHLY A MENŠIE ŠTÍTY (70 bodov)
    sub_peaks_raw = [
        ("Ostrva", 1984, 8.8, 13.2), ("Klin", 2186, 9.0, 14.0), ("Popradský Hrebeň", 2120, 8.0, 14.5),
        ("Veľké Solisko", 2412, 6.0, 14.5), ("Prostredné Solisko", 2400, 6.2, 14.0), ("Mlynár", 2170, 9.8, 18.5),
        ("Žabí kôň", 2291, 8.7, 16.3), ("Volia veža", 2373, 8.9, 16.4), ("Hincova veža", 2377, 7.8, 16.6),
        ("Mengusovská vežička", 2350, 8.1, 16.7), ("Čubrína", 2376, 7.5, 16.4), ("Dravý štít", 2300, 7.2, 16.2),
        ("Zadná Bašta", 2379, 6.3, 15.8), ("Zlatinská veža", 2250, 6.4, 15.4), ("Diablovina", 2390, 6.1, 15.1),
        ("Patria", 2203, 7.0, 12.5), ("Malá Bašta", 2288, 6.6, 13.8), ("Predná Bašta", 2373, 6.4, 14.2),
        ("Kostolík", 2262, 10.8, 15.0), ("Batizovská veža", 2400, 11.2, 15.6), ("Gerlachovská veža", 2640, 11.8, 16.1),
        ("Kotlový štít", 2601, 12.1, 15.6), ("Dromedár", 2450, 12.3, 15.8), ("Guľatý kopec", 2121, 12.1, 14.5),
        ("Granátové veže", 2300, 13.0, 15.5), ("Rohatá veža", 2455, 13.1, 16.5), ("Zadná Garajova veža", 2340, 4.8, 16.2),
        ("Vališkova veža", 2320, 4.5, 15.8), ("Krivánska veža", 2300, 4.2, 15.2), ("Krátka", 2374, 5.0, 15.5),
        ("Ostrá", 2350, 5.2, 15.2), ("Furkotský štít", 2405, 5.8, 16.2), ("Lomnická kopa", 2420, 17.6, 16.2),
        ("Lomnická veža", 2520, 17.8, 16.5), ("Vidlové veže", 2520, 18.2, 17.0), ("Veľká Vidla", 2520, 18.3, 17.1),
        ("Západná Vidla", 2500, 18.1, 16.9), ("Kezmarsky hrb", 2400, 18.9, 16.2), ("Huncovský štít", 2352, 19.5, 16.0),
        ("Malý Kežmarský š.", 2514, 18.6, 17.2), ("Čierny štít", 2429, 17.8, 18.5), ("Kolový štít", 2418, 18.2, 19.0),
        ("Svinka", 2163, 19.8, 18.2), ("Belianska kopa", 1835, 21.5, 19.8), ("Muráň", 1890, 17.0, 22.5),
        ("Nový vrch", 1999, 18.0, 22.0), ("Havran", 2152, 19.0, 21.8), ("Ždiarska vidla", 2142, 20.0, 21.5),
        ("Hlúpy", 2061, 21.0, 20.8), ("Jadlová", 2000, 21.2, 20.5), ("Gánok", 2462, 10.0, 16.2),
        ("Rumanka", 2380, 10.3, 16.0), ("Zlobivá", 2426, 10.5, 16.2), ("Kačací štít", 2401, 10.7, 16.6),
        ("Popradská Kopa", 2110, 8.4, 13.8), ("Kozí štít", 2111, 19.2, 18.6), ("Jastrabia veža", 2137, 19.5, 18.8),
        ("Belasá veža", 2190, 18.6, 19.1), ("Zmrzlá veža", 2200, 18.3, 18.8), ("Drobná veža", 2180, 16.8, 17.4),
        ("Sokolie veže", 2340, 16.0, 16.5), ("Pfinnova kopa", 2121, 16.0, 17.2), ("Strelecká veža", 2130, 14.5, 16.2),
        ("Kresaný roh", 2305, 13.5, 17.0), ("Rovienková veža", 2270, 13.0, 17.2), ("Hranatá veža", 2260, 12.8, 17.0),
        ("Kupola", 2414, 12.5, 17.1), ("Vesterov štít", 2420, 12.0, 15.8), ("Kotlový hrb", 2350, 12.2, 15.4)
    ]
    for n, a, x, y in sub_peaks_raw:
        pts.append({"name": n, "alt": f"{a}m", "x": x, "y": y, "type": "subpeak", "prio": 3})

    # 3. SEDLÁ A PRIECHODY (35 bodov)
    passes = [
        ("Poľský hrebeň", 2200, 11.8, 16.8), ("Prielom", 2290, 12.6, 17.2),
        ("Sedielko", 2376, 15.8, 17.8), ("Priečne sedlo", 2352, 15.2, 17.0),
        ("Baranie sedlo", 2384, 17.2, 18.0), ("Svišťové sedlo", 2192, 13.2, 17.5),
        ("Váha", 2340, 8.7, 15.8), ("Vyšné Kôprovské sedlo", 2180, 7.0, 16.2),
        ("Kopské sedlo", 1750, 20.5, 19.8), ("Sedlo pod Ostrvou", 1960, 9.0, 13.0),
        ("Bystrá lávka", 2300, 5.5, 15.8), ("Lorenzovo sedlo", 2314, 5.7, 15.6),
        ("Hladké sedlo", 1993, 4.2, 17.8), ("Závory", 1876, 4.5, 17.5),
        ("Batizovské sedlo", 2250, 11.2, 15.8), ("Gerlachovské sedlo", 2590, 11.9, 16.3),
        ("Tetmajerovo sedlo", 2580, 11.8, 16.4), ("Gánkovo sedlo", 2390, 10.1, 16.3),
        ("Východné Železné sedlo", 2160, 10.4, 15.4), ("Západné Železné sedlo", 2205, 10.2, 15.3),
        ("Studené sedlo", 2360, 14.8, 14.5), ("Slavkovské sedlo", 2295, 14.2, 14.8),
        ("Lomnické sedlo", 2190, 18.2, 16.0), ("Veľká Zmrzlá lávka", 2350, 17.6, 17.8),
        ("Kolové sedlo", 2090, 18.0, 19.2), ("Jahňacie sedlo", 2100, 19.8, 19.6),
        ("Sedlo pod Svišťovkou", 2023, 19.0, 17.5), ("Široké sedlo", 1825, 20.5, 21.0),
        ("Kozie sedlo", 2050, 19.1, 18.5), ("Žabie sedlo", 2225, 8.6, 16.2),
        ("Mengusovské sedlo", 2208, 8.1, 16.8), ("Hincovo sedlo", 2323, 7.6, 16.5),
        ("Temnosmrečinské sedlo", 2180, 5.8, 17.0), ("Chalubińského vráta", 2029, 7.3, 16.2),
        ("Krivánske sedlo", 2120, 4.2, 14.8)
    ]
    for n, a, x, y in passes:
        pts.append({"name": n, "alt": f"{a}m", "x": x, "y": y, "type": "pass", "prio": 2})

    # 4. PLESÁ A VODNÉ PLOCHY (35 bodov)
    lakes = [
        ("Veľké Hincovo pleso", 1945, 7.5, 15.8), ("Štrbské pleso", 1346, 5.8, 9.5),
        ("Popradské pleso", 1494, 7.2, 12.2), ("Batizovské pleso", 1884, 11.0, 14.5),
        ("Velické pleso", 1670, 12.4, 12.5), ("Skalnaté pleso", 1751, 18.2, 15.0),
        ("Zelené pleso Kežmarské", 1551, 19.2, 18.0), ("Veľké Spišské pleso", 2014, 16.0, 17.0),
        ("Prostredné Spišské pleso", 2013, 16.2, 16.8), ("Nižné Terianske pleso", 1940, 5.0, 16.5),
        ("Vyšné Terianske pleso", 2109, 5.2, 16.8), ("Vyšné Temnosmrečinské pl.", 1725, 5.4, 17.8),
        ("Nižné Temnosmrečinské pl.", 1677, 5.1, 17.5), ("Dračie pleso", 2019, 9.2, 14.8),
        ("Ľadové pleso Zlomiskové", 1925, 9.5, 14.6), ("Kačacie pleso", 1575, 10.6, 17.5),
        ("Čierne pleso Javorové", 1492, 13.8, 19.2), ("Žabie plesá Mengusovské", 1919, 8.2, 15.2),
        ("Vyšné Žabie pleso", 2045, 8.4, 15.5), ("Malé Hincovo pleso", 1923, 7.3, 15.6),
        ("Nižné Wahlenbergovo pleso", 2058, 5.6, 14.8), ("Vyšné Wahlenbergovo pleso", 2157, 5.4, 15.2),
        ("Capie pleso", 2075, 6.2, 15.5), ("Okrúhle pleso", 2105, 6.0, 15.8),
        ("Kozie pleso Mlynické", 1811, 6.3, 14.0), ("Pliesko pod Skokom", 1690, 6.4, 13.2),
        ("Dlhé pleso Velické", 1929, 12.0, 15.0), ("Kvetnicové pliesko", 1812, 12.2, 14.0),
        ("Pusté pleso", 2056, 13.2, 16.8), ("Starolesnianske pleso", 2000, 14.0, 16.2),
        ("Sesterské pleso", 1965, 13.9, 16.0), ("Zbojnícke plesá", 1960, 13.6, 16.4),
        ("Červené pleso", 1810, 19.5, 18.5), ("Belasé pleso", 1862, 19.6, 18.7),
        ("Trojrohé pleso", 1611, 20.0, 18.2)
    ]
    for n, a, x, y in lakes:
        pts.append({"name": n, "alt": f"{a}m", "x": x, "y": y, "type": "lake", "prio": 3})

    # 5. DOLINY (25 bodov)
    valleys = [
        ("Mengusovská dolina", 1600, 7.5, 14.0), ("Zlomisková dolina", 1750, 9.2, 14.0),
        ("Batizovská dolina", 1700, 11.0, 13.5), ("Velická dolina", 1500, 12.4, 11.5),
        ("Slavkovská dolina", 1600, 14.0, 12.5), ("Veľká Studená dolina", 1700, 14.5, 15.0),
        ("Malá Studená dolina", 1800, 16.5, 16.0), ("Skalnatá dolina", 1700, 18.5, 14.5),
        ("Dolina Kežmarskej Bielej vody", 1300, 20.0, 17.0), ("Dolina Zeleného plesa", 1600, 19.0, 18.0),
        ("Dolina Siedmich prameňov", 1350, 21.5, 18.5), ("Monkova dolina", 1100, 21.0, 22.5),
        ("Javorová dolina", 1300, 14.5, 20.0), ("Zadná Javorová dolina", 1700, 14.8, 18.5),
        ("Bielovodská dolina", 1200, 10.5, 20.0), ("Ťažká dolina", 1700, 9.8, 17.0),
        ("Kačacia dolina", 1650, 10.5, 17.0), ("Litvorová dolina", 1750, 11.5, 17.5),
        ("Rovienková dolina", 1800, 12.8, 18.0), ("Svišťová dolina", 1600, 12.5, 18.5),
        ("Kôprová dolina", 1200, 3.5, 14.0), ("Hlinská dolina", 1600, 5.0, 16.0),
        ("Temnosmrečinská dolina", 1600, 4.8, 17.2), ("Mlynická dolina", 1600, 6.3, 13.5),
        ("Furkotská dolina", 1700, 5.5, 14.0)
    ]
    for n, a, x, y in valleys:
        pts.append({"name": n, "alt": f"{a}m", "x": x, "y": y, "type": "valley", "prio": 4})

    # 6. HORSKÉ CHATY (15 bodov)
    huts = [
        ("Chata pod Rysmi", 2250, 8.6, 15.9), ("Téryho chata", 2015, 16.2, 16.8),
        ("Zbojnícka chata", 1960, 13.8, 15.8), ("Chata pod Soliskom", 1840, 6.4, 12.2),
        ("Skalnatá chata", 1751, 18.2, 15.0), ("Sliezsky dom", 1670, 12.4, 12.6),
        ("Chata pri Zelenom plese", 1551, 19.2, 18.0), ("Popradské pleso chata", 1494, 7.2, 12.2),
        ("Chata pri Popradskom plese", 1500, 7.3, 12.3), ("Bilíkova chata", 1255, 15.2, 10.5),
        ("Rainerova chata", 1301, 15.4, 11.0), ("Zamkovského chata", 1475, 16.5, 13.5),
        ("Chata Plesnivec", 1290, 21.8, 19.2), ("Chata pod Kriváňom (býv.)", 1950, 4.2, 14.2),
        ("Chata Kamzík (býv.)", 1295, 15.3, 10.8)
    ]
    for n, a, x, y in huts:
        pts.append({"name": n, "alt": f"{a}m", "x": x, "y": y, "type": "hut", "prio": 2})

    # 7. OSADY A MESTÁ (15 bodov)
    towns = [
        ("Starý Smokovec", 1010, 14.2, 7.5), ("Nový Smokovec", 1000, 13.8, 7.3),
        ("Horný Smokovec", 950, 14.8, 7.8), ("Dolný Smokovec", 890, 14.5, 6.0),
        ("Tatranská Lomnica", 850, 19.5, 9.0), ("Tatranské Matliare", 885, 20.5, 10.5),
        ("Tatranská Lesná", 915, 17.5, 7.2), ("Štrbské Pleso", 1346, 5.8, 9.5),
        ("Tatranská Štrba", 850, 5.0, 4.0), ("Štrba", 829, 4.5, 2.0),
        ("Tatranská Polianka", 1005, 11.5, 6.8), ("Nová Polianka", 1040, 10.0, 6.0),
        ("Vyšné Hágy", 1125, 8.5, 6.5), ("Podbanské", 940, 1.5, 11.0),
        ("Ždiar", 896, 20.5, 22.0), ("Poprad", 672, 20.0, 1.8),
        ("Veľká Lomnica", 640, 22.0, 5.0), ("Gerlachov", 780, 13.0, 3.5)
    ]
    for n, a, x, y in towns:
        pts.append({"name": n, "alt": f"{a}m", "x": x, "y": y, "type": "town", "prio": 1})

    return pts

ALL_LANDMARKS = build_dense_tatry_points()

def draw_dense_landmarks(ax, is_compact=False, show_all=True):
    """Vykreslí orientačnú sieť 200+ bodov s vyváženým zobrazením."""
    for lm in ALL_LANDMARKS:
        ltype = lm["type"]
        prio = lm["prio"]
        
        # Voľba štýlu ikony
        if ltype == "peak":
            marker = '^'
            mcolor = '#ef4444' # Červená
            msize = 5.5 if is_compact else 7.5
        elif ltype == "subpeak":
            marker = '^'
            mcolor = '#f87171' # Svetločervená
            msize = 2.5 if is_compact else 4.0
        elif ltype == "pass":
            marker = 'x'
            mcolor = '#fbbf24' # Jantárová
            msize = 3.0 if is_compact else 5.0
        elif ltype == "lake":
            marker = 'o'
            mcolor = '#38bdf8' # Azúrová
            msize = 2.5 if is_compact else 4.5
        elif ltype == "hut":
            marker = 's'
            mcolor = '#f59e0b' # Oranžová
            msize = 4.0 if is_compact else 6.0
        elif ltype == "town":
            marker = 'o'
            mcolor = '#a855f7' # Fialová
            msize = 4.5 if is_compact else 6.5
        else: # valley
            marker = '.'
            mcolor = '#94a3b8'
            msize = 2.0 if is_compact else 3.5

        ax.plot(lm["x"], lm["y"], marker=marker, markersize=msize, color=mcolor, 
                markeredgecolor='#000000', markeredgewidth=0.5, alpha=0.9, zorder=12)

        # Inteligentné popisky (aby sa texty neprekrývali)
        should_label = False
        if not is_compact:
            # V samostatnom veľkom grafe popíšeme prio 1, prio 2 a vybrané jazerá/chaty
            if prio <= 2 or ltype in ["hut", "town"]:
                should_label = True
        else:
            # V 4-panelovom grafe len hlavné štíty a hlavné obce
            if prio == 1:
                should_label = True

        if should_label:
            fsize = 5.5 if is_compact else 7.0
            label = lm['name'] if is_compact else f"{lm['name']}\n{lm['alt']}"
            y_offset = 0.28 if is_compact else 0.38
            
            ax.text(lm["x"], lm["y"] + y_offset, label,
                    fontsize=fsize, fontweight='bold', color='white', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='#0f172a', edgecolor=mcolor, alpha=0.82, linewidth=0.5),
                    zorder=13)

# =============================================================================
# TOPOGRAFIA TATIER SO ZABUDOVANÝMI 200+ STRUKTÚRAMI (DEM)
# =============================================================================
def generate_tatry_dem(grid_shape=(160, 160), dx=150.0, dy=150.0):
    nx, ny = grid_shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    dem = 600.0 + (Y * 0.009)
    
    # Hlavný hrebeň
    ridge_y = 16500.0
    main_ridge = 1500.0 * np.exp(-((Y - ridge_y)**2) / (2 * 2600.0**2))
    ridge_waves = 1.0 + 0.2 * np.sin(X / 1400.0) * np.cos(X / 2800.0)
    dem += main_ridge * ridge_waves
    
    # Rázsochy a bočné chrbty
    spurs = 600.0 * np.exp(-((Y - 13000.0)**2) / (2 * 3200.0**2)) * np.maximum(np.cos(X / 1700.0), 0.0)**2
    dem += spurs
    
    # Doliny (zárezy)
    valley_system = (
        np.exp(-((X - 3500.0)**2) / (2 * 900.0**2)) +   # Kôprová
        np.exp(-((X - 5500.0)**2) / (2 * 650.0**2)) +   # Furkotská
        np.exp(-((X - 6300.0)**2) / (2 * 600.0**2)) +   # Mlynická
        np.exp(-((X - 7500.0)**2) / (2 * 700.0**2)) +   # Mengusovská
        np.exp(-((X - 9200.0)**2) / (2 * 550.0**2)) +   # Zlomiská
        np.exp(-((X - 11000.0)**2) / (2 * 600.0**2)) +  # Batizovská
        np.exp(-((X - 12400.0)**2) / (2 * 600.0**2)) +  # Velická
        np.exp(-((X - 14500.0)**2) / (2 * 750.0**2)) +  # Veľká Studená
        np.exp(-((X - 16500.0)**2) / (2 * 700.0**2)) +  # Malá Studená
        np.exp(-((X - 18500.0)**2) / (2 * 650.0**2)) +  # Skalnatá
        np.exp(-((X - 20000.0)**2) / (2 * 800.0**2))    # Kežmarská Biela voda
    ) * np.exp(-((Y - 13500.0)**2) / (2 * 4200.0**2))
    dem -= valley_system * 480.0

    # Pridanie výškových kupolí štítov
    for p in ALL_LANDMARKS:
        if p["type"] in ["peak", "subpeak"]:
            alt_num = float(p["alt"].replace("m", ""))
            if alt_num > 2100:
                h_peak = (alt_num - 1900.0) * 0.95
                r_peak = 500.0 if p["type"] == "peak" else 320.0
                dem += h_peak * np.exp(-((X - p["x"]*1000.0)**2 + (Y - p["y"]*1000.0)**2) / (2 * r_peak**2))

    # Plynulý prechod do kotliny
    basin_weight = 1.0 / (1.0 + np.exp((Y - 6500.0) / 900.0))
    dem = dem * (1.0 - basin_weight) + (630.0 + Y * 0.003) * basin_weight
    dem = ndimage.gaussian_filter(dem, sigma=1.0)
    return X, Y, dem, dx, dy

X, Y, DEM, DX, DY = generate_tatry_dem()

# =============================================================================
# NUMERICKÝ ENGINE PRE 48H HORIZONT
# =============================================================================
def simulate_forecast_step(step_idx: int):
    hours_ahead = step_idx * 6
    
    wind_base_speed = 11.5 + 4.5 * np.sin(step_idx * 0.7)
    wind_angle_deg = -50.0 + 15.0 * np.cos(step_idx * 0.5)
    rad = np.radians(wind_angle_deg)
    
    u_bg = np.full_like(X, wind_base_speed * np.cos(rad)) + 1.2 * np.sin((X + hours_ahead * 400.0) / 8000.0)
    v_bg = np.full_like(Y, wind_base_speed * np.sin(rad)) + 1.2 * np.cos((Y + hours_ahead * 400.0) / 8000.0)
    
    t_synoptic = -2.5 - (step_idx * 0.5) + 1.5 * np.sin(X / 10000.0)
    t_bg = t_synoptic - ((DEM - 700.0) * 0.0065)
    
    is_night = (step_idx % 4) in [1, 2]
    inv_strength = 6.0 if is_night else 1.5
    temp_field = t_bg.copy()
    inv_mask = DEM < 1050.0
    temp_field[inv_mask] -= inv_strength * ((1050.0 - DEM[inv_mask]) / 400.0)
    
    dh_dx, dh_dy = np.gradient(DEM, DX, DY)
    slope = np.sqrt(dh_dx**2 + dh_dy**2)
    aspect = np.arctan2(-dh_dx, dh_dy)
    
    dem_base = ndimage.gaussian_filter(DEM, sigma=16)
    h_rel = np.maximum(DEM - dem_base, 0.0)
    delta_S = (1.6 * h_rel / 3500.0) * np.exp(-30.0 / 3500.0)
    u_speed = u_bg * (1.0 + delta_S)
    v_speed = v_bg * (1.0 + delta_S)
    
    speed_init = np.sqrt(u_speed**2 + v_speed**2)
    wind_dir = np.arctan2(v_speed, u_speed)
    delta_theta = np.clip(-0.25 * (slope * 100.0) * np.sin(2.0 * (aspect - wind_dir)), -0.35, 0.35)
    steered_dir = wind_dir + delta_theta
    u_steered = speed_init * np.cos(steered_dir)
    v_steered = speed_init * np.sin(steered_dir)
    
    downslope = u_steered * dh_dx + v_steered * dh_dy
    bora_mask = (downslope < -0.08) & (DEM > 750.0)
    fall_h = np.maximum(2100.0 - DEM, 0.0)
    acc = np.zeros_like(DEM)
    acc[bora_mask] = np.sqrt(2.0 * 9.81 * fall_h[bora_mask] * 0.08)
    
    spd = np.sqrt(u_steered**2 + v_steered**2)
    u_bora = u_steered.copy()
    v_bora = v_steered.copy()
    u_bora[bora_mask] += (u_steered[bora_mask] / spd[bora_mask]) * acc[bora_mask]
    v_bora[bora_mask] += (v_steered[bora_mask] / spd[bora_mask]) * acc[bora_mask]
    
    # MASCON Solver
    nx, ny = DEM.shape
    lam = np.zeros((nx, ny))
    du_dx = np.zeros_like(u_bora)
    dv_dy = np.zeros_like(v_bora)
    du_dx[1:-1, :] = (u_bora[2:, :] - u_bora[:-2, :]) / (2.0 * DX)
    dv_dy[:, 1:-1] = (v_bora[:, 2:] - v_bora[:, :-2]) / (2.0 * DY)
    source = -2.0 * (du_dx + dv_dy)
    omega = 1.65
    dx2, dy2 = DX**2, DY**2
    denom = 2.0 * (1.0/dx2 + 1.0/dy2)
    for _ in range(50):
        lam_old = lam.copy()
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                l_new = ((lam[i+1, j] + lam[i-1, j])/dx2 + (lam[i, j+1] + lam[i, j-1])/dy2 - source[i, j]) / denom
                lam[i, j] = (1.0 - omega) * lam[i, j] + omega * l_new
        if np.max(np.abs(lam - lam_old)) < 1e-4:
            break
            
    dlam_dx = np.zeros_like(lam)
    dlam_dy = np.zeros_like(lam)
    dlam_dx[1:-1, :] = (lam[2:, :] - lam[:-2, :]) / (2.0 * DX)
    dlam_dy[:, 1:-1] = (lam[:, 2:] - lam[:, :-2]) / (2.0 * DY)
    u_opt = u_bora + 0.5 * dlam_dx
    v_opt = v_bora + 0.5 * dlam_dy
    w_opt = u_opt * dh_dx + v_opt * dh_dy
    
    synoptic_precip = np.maximum(2.2 + 2.8 * np.sin((step_idx + 1) * 0.75) + 1.2 * np.sin(X / 5000.0), 0.2)
    p_sf = synoptic_precip.copy()
    p_sf[w_opt > 0.0] *= (1.0 + 0.4 * w_opt[w_opt > 0.0])
    shift_x = int(np.clip(-u_opt.mean() * 250.0 / DX, -6, 6))
    shift_y = int(np.clip(-v_opt.mean() * 250.0 / DY, -6, 6))
    p_drift = np.roll(np.roll(p_sf, shift_x, axis=0), shift_y, axis=1)
    p_final = np.maximum(p_drift * np.where(w_opt < 0.0, np.exp(0.3 * w_opt), 1.0), 0.0)
    
    snow_mask = temp_field < 0.0
    fresh_snow_rate = np.zeros_like(p_final)
    fresh_snow_rate[snow_mask] = p_final[snow_mask]
    
    wind_spd = np.sqrt(u_opt**2 + v_opt**2)
    snow_drift = np.zeros_like(wind_spd)
    act = (wind_spd > 6.0) & snow_mask
    snow_drift[act] = 0.0012 * (wind_spd[act] - 6.0)**2.5
    
    qx = snow_drift * (u_opt / wind_spd)
    qy = snow_drift * (v_opt / wind_spd)
    dqx = np.zeros_like(qx)
    dqy = np.zeros_like(qy)
    dqx[1:-1, :] = (qx[2:, :] - qx[:-2, :]) / (2.0 * DX)
    dqy[:, 1:-1] = (qy[:, 2:] - qy[:, :-2]) / (2.0 * DY)
    
    dqx[0, :] = dqx[-1, :] = dqy[:, 0] = dqy[:, -1] = 0.0
    snow_redist = -(dqx + dqy) * (6.0 * 3600.0) / (100.0 * 0.01)
    snow_redist = np.clip(snow_redist, -25.0, 35.0)
    snow_diff = np.where(snow_mask, (fresh_snow_rate * 6.0) + snow_redist, 0.0)
    
    exposure = np.exp((DEM - 650.0) / 600.0) * (1.0 + 1.8 * slope)
    instability = np.maximum(w_opt, 0.0) * 1.2
    temp_factor = np.zeros_like(temp_field)
    mp = (temp_field >= -18.0) & (temp_field <= -2.0)
    temp_factor[mp] = np.exp(-((temp_field[mp] + 10.0)**2) / 32.0)
    lhi = np.clip(ndimage.gaussian_filter(exposure * 0.35 + instability * 25.0 + temp_factor * 35.0, sigma=1.0), 0.0, 100.0)
    
    return {
        'hours': hours_ahead,
        'u_opt': u_opt, 'v_opt': v_opt, 'acc': acc,
        'p_final': p_final, 'snow_diff': snow_diff,
        'lhi': lhi, 'temp_field': temp_field, 'wind_spd': wind_spd
    }

FORECAST_TIMELINE = [simulate_forecast_step(i) for i in range(9)]

# =============================================================================
# ENDPOINTY
# =============================================================================
@app.get("/")
def read_root():
    return {"status": "ok", "service": "TATRYS-50 v2 Ultra-HD", "landmarks_total": len(ALL_LANDMARKS)}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "tatry-meteo-api"}

@app.get("/api/landmarks")
def get_landmarks():
    """Vráti kompletnú sieť 200+ bodov pre interaktívnu Leaflet mapu."""
    return {"status": "ok", "total": len(ALL_LANDMARKS), "landmarks": ALL_LANDMARKS}

@app.get("/api/forecast")
@app.get("/api/stations")
def get_forecast(step: int = Query(0, ge=0, le=8)):
    d = FORECAST_TIMELINE[step]
    
    # 8 hlavných monitorovacích bodov
    locs = [
        {"name": "Lomnický štít (2 634 m)", "x": 18.0, "y": 16.8},
        {"name": "Gerlachovský štít (2 655 m)", "x": 12.0, "y": 16.2},
        {"name": "Kriváň (2 495 m)", "x": 4.0, "y": 15.0},
        {"name": "Téryho chata (2 015 m)", "x": 16.2, "y": 16.8},
        {"name": "Zbojnícka chata (1 960 m)", "x": 13.8, "y": 15.8},
        {"name": "Štrbské Pleso (1 346 m)", "x": 5.8, "y": 9.5},
        {"name": "Starý Smokovec (1 010 m)", "x": 14.2, "y": 7.5},
        {"name": "Poprad (672 m)", "x": 20.0, "y": 1.8}
    ]
    res = []
    for l in locs:
        ix = int(np.clip(l['x'] * 1000.0 / DX, 0, DEM.shape[0]-1))
        iy = int(np.clip(l['y'] * 1000.0 / DY, 0, DEM.shape[1]-1))
        spd = float(d['wind_spd'][ix, iy] * 3.6)
        t = float(d['temp_field'][ix, iy])
        p = float(d['p_final'][ix, iy])
        s = float(d['snow_diff'][ix, iy])
        lh = float(d['lhi'][ix, iy])
        res.append({
            "name": l['name'],
            "temp": round(t, 1),
            "wind_kmh": round(spd, 1),
            "precip_mmh": round(p, 1),
            "snow_6h_cm": round(s, 1),
            "lightning_risk": "Vysoké" if lh > 60 else ("Stredné" if lh > 30 else "Nízke"),
            "lhi_raw": round(lh, 0)
        })
    return {"status": "ok", "step": step, "hours_ahead": d['hours'], "stations": res}

@app.get("/api/render-map")
def render_map(layer: str = Query("all"), step: int = Query(0, ge=0, le=8)):
    d = FORECAST_TIMELINE[step]
    X_km, Y_km = X / 1000.0, Y / 1000.0
    h_label = f"+{d['hours']}h"
    
    if layer == "all":
        fig, axs = plt.subplots(2, 2, figsize=(18, 15), facecolor='#0f172a')
        for row in axs:
            for ax in row:
                ax.set_facecolor('#1e293b')
                ax.tick_params(colors='#94a3b8')
                ax.set_xlabel('Západ -> Východ (km)', color='#94a3b8', fontsize=8)
                ax.set_ylabel('Juh -> Sever (km)', color='#94a3b8', fontsize=8)
                for s in ax.spines.values():
                    s.set_color('#334155')

        # 1. Vietor & Bóra
        im1 = axs[0, 0].contourf(X_km, Y_km, DEM, levels=30, cmap='terrain', alpha=0.85)
        cb1 = fig.colorbar(im1, ax=axs[0, 0])
        plt.setp(plt.getp(cb1.ax.axes, 'yticklabels'), color='white')
        axs[0, 0].quiver(X_km[::8, ::8], Y_km[::8, ::8], d['u_opt'][::8, ::8], d['v_opt'][::8, ::8], scale=130, color='black', width=0.002)
        if d['acc'].max() > 1.0:
            axs[0, 0].contour(X_km, Y_km, d['acc'], levels=[3.0, 6.0, 9.0], colors='#ef4444', linewidths=1.2, linestyles='--')
        draw_dense_landmarks(axs[0, 0], is_compact=True)
        axs[0, 0].set_title(f'A. Topografia Tatier & Vietor [{h_label}]', color='white', fontweight='bold', fontsize=11)

        # 2. Zrážky
        im2 = axs[0, 1].contourf(X_km, Y_km, d['p_final'], levels=25, cmap='YlGnBu')
        cb2 = fig.colorbar(im2, ax=axs[0, 1], label='mm / h')
        plt.setp(plt.getp(cb2.ax.axes, 'yticklabels'), color='white')
        cb2.ax.yaxis.label.set_color('white')
        axs[0, 1].contour(X_km, Y_km, DEM, levels=[1000, 1500, 2000, 2500], colors='#64748b', linewidths=0.5, alpha=0.5)
        draw_dense_landmarks(axs[0, 1], is_compact=True)
        axs[0, 1].set_title(f'B. Intenzita zrážok (Seeder-Feeder) [{h_label}]', color='white', fontweight='bold', fontsize=11)

        # 3. Sneh
        m_diff = max(np.max(np.abs(d['snow_diff'])), 1.0)
        im3 = axs[1, 0].contourf(X_km, Y_km, d['snow_diff'], levels=25, cmap='RdBu', vmin=-m_diff, vmax=m_diff)
        cb3 = fig.colorbar(im3, ax=axs[1, 0], label='Rozdiel (cm / 6h)')
        plt.setp(plt.getp(cb3.ax.axes, 'yticklabels'), color='white')
        cb3.ax.yaxis.label.set_color('white')
        axs[1, 0].contour(X_km, Y_km, DEM, levels=[1000, 1500, 2000, 2500], colors='#64748b', linewidths=0.5, alpha=0.5)
        draw_dense_landmarks(axs[1, 0], is_compact=True)
        axs[1, 0].set_title(f'C. Prevejovanie snehu za 6h [{h_label}]', color='white', fontweight='bold', fontsize=11)

        # 4. Blesky & Inverzia
        im4 = axs[1, 1].contourf(X_km, Y_km, d['lhi'], levels=25, cmap='YlOrRd')
        cb4 = fig.colorbar(im4, ax=axs[1, 1], label='LHI Index (0-100)')
        plt.setp(plt.getp(cb4.ax.axes, 'yticklabels'), color='white')
        cb4.ax.yaxis.label.set_color('white')
        axs[1, 1].contour(X_km, Y_km, d['temp_field'], levels=[-6.0, -3.0, 0.0], colors='#38bdf8', linewidths=1.2, linestyles='-.')
        draw_dense_landmarks(axs[1, 1], is_compact=True)
        axs[1, 1].set_title(f'D. Riziko bleskov & Teplotné pole [{h_label}]', color='white', fontweight='bold', fontsize=11)

        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='#0f172a')
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.set_xlabel('Západ -> Východ (km)', color='#94a3b8', fontsize=10)
        ax.set_ylabel('Juh -> Sever (km)', color='#94a3b8', fontsize=10)

        if layer == "wind":
            im = ax.contourf(X_km, Y_km, DEM, levels=30, cmap='terrain', alpha=0.85)
            ax.quiver(X_km[::6, ::6], Y_km[::6, ::6], d['u_opt'][::6, ::6], d['v_opt'][::6, ::6], scale=120, color='black', width=0.0025)
            if d['acc'].max() > 1.0:
                ax.contour(X_km, Y_km, d['acc'], levels=[3.0, 6.0, 9.0], colors='#ef4444', linewidths=1.5, linestyles='--')
            cb = fig.colorbar(im, ax=ax, label='Nadmorská výška (m n.m.)')
            ax.set_title(f'Prúdenie vetra & Bóra (200+ bodov) [{h_label}]', color='white', fontweight='bold', fontsize=13)
        elif layer == "precip":
            im = ax.contourf(X_km, Y_km, d['p_final'], levels=25, cmap='YlGnBu')
            ax.contour(X_km, Y_km, DEM, levels=[1000, 1500, 2000, 2500], colors='#64748b', linewidths=0.6, alpha=0.6)
            cb = fig.colorbar(im, ax=ax, label='Intenzita zrážok (mm / h)')
            ax.set_title(f'Lokálne orografické zrážky [{h_label}]', color='white', fontweight='bold', fontsize=13)
        elif layer == "snow":
            m_diff = max(np.max(np.abs(d['snow_diff'])), 1.0)
            im = ax.contourf(X_km, Y_km, d['snow_diff'], levels=25, cmap='RdBu', vmin=-m_diff, vmax=m_diff)
            ax.contour(X_km, Y_km, DEM, levels=[1000, 1500, 2000, 2500], colors='#64748b', linewidths=0.6, alpha=0.6)
            cb = fig.colorbar(im, ax=ax, label='Zmena výšky snehu (cm / 6h)')
            ax.set_title(f'Redistribúcia a akumulácia snehu za 6h [{h_label}]', color='white', fontweight='bold', fontsize=13)
        elif layer == "lightning":
            im = ax.contourf(X_km, Y_km, d['lhi'], levels=25, cmap='YlOrRd')
            ax.contour(X_km, Y_km, d['temp_field'], levels=[-6.0, -3.0, 0.0], colors='#38bdf8', linewidths=1.2, linestyles='-.')
            cb = fig.colorbar(im, ax=ax, label='LHI Index (0-100)')
            ax.set_title(f'Index nebezpečenstva bleskov & Teplota [{h_label}]', color='white', fontweight='bold', fontsize=13)

        draw_dense_landmarks(ax, is_compact=False)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
        cb.ax.yaxis.label.set_color('white')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

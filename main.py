import io
import os
import datetime
import urllib.request
import json
import numpy as np
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import FastAPI, Response, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TATRYS-50 v2 High-Density Grid API | Avalanche.sk",
    description="Numerický orografický model Vysokých Tatier s 200+ výpočtovými bodmi.",
    version="2.6.0"
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
# 1. GENERÁTOR TERÉNU VYSOKÝCH TATIER (200m DEM)
# =============================================================================
def generate_tatry_dem(grid_shape=(140, 140), dx=171.4, dy=171.4):
    nx, ny = grid_shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    dem = 620.0 + (Y * 0.008)
    ridge_y = 16500.0
    main_ridge = 1450.0 * np.exp(-((Y - ridge_y)**2) / (2 * 2800.0**2))
    ridge_waves = 1.0 + 0.18 * np.sin(X / 1600.0) * np.cos(X / 3200.0)
    dem += main_ridge * ridge_waves
    
    spurs = 550.0 * np.exp(-((Y - 13000.0)**2) / (2 * 3000.0**2)) * np.maximum(np.cos(X / 1800.0), 0.0)**2
    dem += spurs
    
    valleys = (
        np.exp(-((X - 7000.0)**2) / (2 * 700.0**2)) +   # Mengusovská
        np.exp(-((X - 12000.0)**2) / (2 * 650.0**2)) +  # Velická
        np.exp(-((X - 15000.0)**2) / (2 * 800.0**2)) +  # Studená
        np.exp(-((X - 3000.0)**2) / (2 * 900.0**2))     # Kôprová
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
    return X, Y, dem, dx, dy

X, Y, DEM, DX, DY = generate_tatry_dem()

# =============================================================================
# 2. GENERÁCIA 200+ VÝPOČTOVÝCH BODOV V PRIESTORE TATIER
# =============================================================================
def generate_200_calculation_points():
    """Generuje 210 pravidelných a geomorfologických výpočtových uzlov v doméne 24x24 km."""
    pts = []
    # Matica 15 x 14 bodov = 210 bodov
    gx = np.linspace(1.0, 23.0, 15)
    gy = np.linspace(1.5, 22.5, 14)
    point_id = 1
    for px in gx:
        for py in gy:
            ix = int(np.clip(px * 1000.0 / DX, 0, DEM.shape[0] - 1))
            iy = int(np.clip(py * 1000.0 / DY, 0, DEM.shape[1] - 1))
            ele = int(DEM[ix, iy])
            pts.append({
                "id": point_id,
                "name": f"Bod #{point_id} ({ele}m)",
                "x": round(float(px), 2),
                "y": round(float(py), 2),
                "ix": ix,
                "iy": iy,
                "ele": ele
            })
            point_id += 1
    return pts

CALC_POINTS_200 = generate_200_calculation_points()

# =============================================================================
# 3. ZÍSKANIE LIVE DWD MODELU A NUMERICKÁ SIMULÁCIA
# =============================================================================
def fetch_real_dwd_forecast():
    url = (
        "https://api.open-meteo.com/v1/dwd-icon?"
        "latitude=49.16&longitude=20.13&hourly=temperature_2m,precipitation,"
        "wind_speed_10m,wind_direction_10m,cape&forecast_days=3&timezone=auto"
    )
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'AvalancheTatryModel/2.6'})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())
            return data.get("hourly", {})
    except Exception as e:
        print(f"[VAROVANIE] DWD API offline: {e}. Používam záložné pole.")
        return None

RAW_DWD_DATA = fetch_real_dwd_forecast()

def simulate_forecast_step(step_idx: int):
    hours_ahead = step_idx * 6
    
    if RAW_DWD_DATA and "temperature_2m" in RAW_DWD_DATA:
        idx = min(hours_ahead, len(RAW_DWD_DATA["temperature_2m"]) - 1)
        base_temp_2m = RAW_DWD_DATA["temperature_2m"][idx]
        base_wind_spd = RAW_DWD_DATA["wind_speed_10m"][idx] / 3.6
        base_wind_dir = RAW_DWD_DATA["wind_direction_10m"][idx]
        base_precip = RAW_DWD_DATA["precipitation"][idx]
        base_cape = RAW_DWD_DATA.get("cape", [0])[idx] or 0.0
    else:
        base_temp_2m = 16.0 - (step_idx * 0.4)
        base_wind_spd = 5.5 + np.sin(step_idx)
        base_wind_dir = 310.0
        base_precip = 0.5
        base_cape = 150.0

    # Teplotné pole (gradient -0.65 °C / 100 m)
    t_ref_dem = 800.0
    temp_field = base_temp_2m - ((DEM - t_ref_dem) * 0.0065)
    
    is_night = (step_idx % 4) in [1, 2]
    if is_night and base_wind_spd < 6.0:
        inv_mask = DEM < 1000.0
        temp_field[inv_mask] -= 3.0 * ((1000.0 - DEM[inv_mask]) / 350.0)

    # Vektor vetra
    rad = np.radians(270.0 - base_wind_dir)
    u_bg = np.full_like(X, base_wind_spd * np.cos(rad))
    v_bg = np.full_like(Y, base_wind_spd * np.sin(rad))
    
    dh_dx, dh_dy = np.gradient(DEM, DX, DY)
    slope = np.sqrt(dh_dx**2 + dh_dy**2)
    aspect = np.arctan2(-dh_dx, dh_dy)
    
    dem_base = ndimage.gaussian_filter(DEM, sigma=16)
    h_rel = np.maximum(DEM - dem_base, 0.0)
    delta_S = np.clip((1.2 * h_rel / 3500.0), 0.0, 0.5)
    u_speed = u_bg * (1.0 + delta_S)
    v_speed = v_bg * (1.0 + delta_S)
    
    speed_init = np.sqrt(u_speed**2 + v_speed**2)
    wind_dir = np.arctan2(v_speed, u_speed)
    delta_theta = np.clip(-0.2 * (slope * 100.0) * np.sin(2.0 * (aspect - wind_dir)), -0.3, 0.3)
    steered_dir = wind_dir + delta_theta
    u_steered = speed_init * np.cos(steered_dir)
    v_steered = speed_init * np.sin(steered_dir)
    
    # Tatranská Bóra
    downslope = u_steered * dh_dx + v_steered * dh_dy
    bora_acc = np.zeros_like(DEM)
    if base_wind_spd > 8.0:
        bora_mask = (downslope < -0.1) & (DEM > 750.0)
        fall_h = np.maximum(2100.0 - DEM, 0.0)
        bora_acc[bora_mask] = np.sqrt(2.0 * 9.81 * fall_h[bora_mask] * 0.06)
        spd = np.sqrt(u_steered**2 + v_steered**2)
        u_steered[bora_mask] += (u_steered[bora_mask] / np.maximum(spd[bora_mask], 0.1)) * bora_acc[bora_mask]
        v_steered[bora_mask] += (v_steered[bora_mask] / np.maximum(spd[bora_mask], 0.1)) * bora_acc[bora_mask]

    u_opt = u_steered
    v_opt = v_steered
    w_opt = u_opt * dh_dx + v_opt * dh_dy
    wind_spd = np.sqrt(u_opt**2 + v_opt**2)

    # Zrážky
    p_final = np.maximum(base_precip * (1.0 + 0.35 * np.maximum(w_opt, 0.0)), 0.0)
    if base_precip == 0:
        p_final = np.zeros_like(DEM)

    # Sneh
    snow_mask = temp_field < 0.0
    fresh_snow_rate = np.zeros_like(p_final)
    fresh_snow_rate[snow_mask] = p_final[snow_mask] * 1.0
    
    snow_diff = np.zeros_like(DEM)
    if np.any(snow_mask) and np.any(fresh_snow_rate > 0):
        snow_diff = fresh_snow_rate * 6.0
        snow_diff[slope > 0.6] *= 0.7

    # LHI
    instability = np.maximum(w_opt, 0.0) * (base_cape / 400.0)
    exposure = np.clip((DEM - 650.0) / 40.0, 0.0, 40.0)
    temp_factor = np.where((temp_field >= -15.0) & (temp_field <= 2.0), 25.0, 0.0)
    lhi = np.clip(ndimage.gaussian_filter(exposure * 0.4 + instability * 20.0 + temp_factor, sigma=1.2), 0.0, 100.0)
    if base_cape < 50.0 and base_precip == 0:
        lhi = np.clip(lhi * 0.2, 0.0, 25.0)

    return {
        'hours': hours_ahead,
        'u_opt': u_opt, 'v_opt': v_opt, 'acc': bora_acc,
        'p_final': p_final, 'snow_diff': snow_diff,
        'lhi': lhi, 'temp_field': temp_field, 'wind_spd': wind_spd
    }

FORECAST_TIMELINE = [simulate_forecast_step(i) for i in range(9)]

# =============================================================================
# 4. KARTOGRAFIA A VYKRESLENIE 200 BODOV NA MAPU
# =============================================================================
PRIMARY_LANDMARKS = [
    {"name": "Gerlach", "alt": "2655m", "x": 12.0, "y": 16.2, "type": "peak"},
    {"name": "Lomnický štít", "alt": "2634m", "x": 18.0, "y": 16.8, "type": "peak"},
    {"name": "Kriváň", "alt": "2495m", "x": 4.0, "y": 15.0, "type": "peak"},
    {"name": "Rysy", "alt": "2501m", "x": 8.5, "y": 16.0, "type": "peak"},
    {"name": "Zbojnícka chata", "alt": "1960m", "x": 13.8, "y": 15.8, "type": "hut"},
    {"name": "Téryho chata", "alt": "2015m", "x": 16.2, "y": 16.8, "type": "hut"},
    {"name": "Štrbské Pleso", "alt": "1346m", "x": 5.8, "y": 9.5, "type": "town"},
    {"name": "Starý Smokovec", "alt": "1010m", "x": 14.2, "y": 7.5, "type": "town"},
    {"name": "Poprad", "alt": "672m", "x": 20.0, "y": 1.8, "type": "town"}
]

def draw_grid_and_landmarks(ax, is_compact=False):
    # 1. Vykreslenie siete 200 výpočtových bodov (jemné body)
    grid_x = [p["x"] for p in CALC_POINTS_200]
    grid_y = [p["y"] for p in CALC_POINTS_200]
    ax.scatter(grid_x, grid_y, s=4 if is_compact else 8, c='#94a3b8', marker='+', alpha=0.55, zorder=8)

    # 2. Vykreslenie hlavných dominant s popiskami[cite: 1]
    for lm in PRIMARY_LANDMARKS:
        is_peak = lm["type"] == "peak"
        marker = '^' if is_peak else ('s' if lm["type"] == "hut" else 'o')
        mcolor = '#ef4444' if is_peak else ('#f59e0b' if lm["type"] == "hut" else '#38bdf8')
        ax.plot(lm["x"], lm["y"], marker=marker, markersize=5 if is_compact else 7, color=mcolor, markeredgecolor='white', zorder=10)
        ax.text(lm["x"], lm["y"] + (0.35 if is_compact else 0.45), f"{lm['name']}\n{lm['alt']}",
                fontsize=5.5 if is_compact else 7, fontweight='bold', color='white', ha='center',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='#0f172a', edgecolor=mcolor, alpha=0.85),
                zorder=11)

# =============================================================================
# 5. FASTAPI ENDPOINTY
# =============================================================================
@app.get("/")
def read_root():
    return {"status": "ok", "service": "TATRYS-50 High-Density Grid", "calc_points_count": len(CALC_POINTS_200)}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/points-grid")
def get_points_grid(step: int = Query(0, ge=0, le=8)):
    """Vráti vypočítané meteorologické hodnoty pre všetkých 200+ bodov v danom kroku."""
    d = FORECAST_TIMELINE[step]
    results = []
    for p in CALC_POINTS_200:
        ix, iy = p["ix"], p["iy"]
        spd = float(d['wind_spd'][ix, iy] * 3.6)
        t = float(d['temp_field'][ix, iy])
        prec = float(d['p_final'][ix, iy])
        sn = float(d['snow_diff'][ix, iy])
        lh = float(d['lhi'][ix, iy])
        results.append({
            "id": p["id"],
            "name": p["name"],
            "x": p["x"],
            "y": p["y"],
            "ele": p["ele"],
            "temp": round(t, 1),
            "wind_kmh": round(spd, 1),
            "precip_mmh": round(prec, 1),
            "snow_6h_cm": round(sn, 1),
            "lhi": round(lh, 0)
        })
    return {"status": "ok", "step": step, "hours_ahead": d["hours"], "count": len(results), "points": results}

@app.get("/api/forecast")
@app.get("/api/stations")
def get_forecast(step: int = Query(0, ge=0, le=8)):
    d = FORECAST_TIMELINE[step]
    locs = [
        {"name": "Lomnický štít (2 634 m)", "x": 18.0, "y": 16.8},
        {"name": "Gerlachovský štít (2 655 m)", "x": 12.0, "y": 16.2},
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
        fig, axs = plt.subplots(2, 2, figsize=(16, 13), facecolor='#0f172a')
        for row in axs:
            for ax in row:
                ax.set_facecolor('#1e293b')
                ax.tick_params(colors='#94a3b8')
                for s in ax.spines.values():
                    s.set_color('#334155')

        # 1. Topografia & Vietor
        im1 = axs[0, 0].contourf(X_km, Y_km, DEM, levels=25, cmap='terrain', alpha=0.85)
        fig.colorbar(im1, ax=axs[0, 0])
        axs[0, 0].quiver(X_km[::7, ::7], Y_km[::7, ::7], d['u_opt'][::7, ::7], d['v_opt'][::7, ::7], scale=120, color='black')
        draw_grid_and_landmarks(axs[0, 0], is_compact=True)
        axs[0, 0].set_title(f'A. Topografia & Vietor (200+ bodov) [{h_label}]', color='white', fontweight='bold')

        # 2. Zrážky
        im2 = axs[0, 1].contourf(X_km, Y_km, d['p_final'], levels=20, cmap='YlGnBu')
        fig.colorbar(im2, ax=axs[0, 1], label='mm / h')
        draw_grid_and_landmarks(axs[0, 1], is_compact=True)
        axs[0, 1].set_title(f'B. Intenzita zrážok [{h_label}]', color='white', fontweight='bold')

        # 3. Sneh
        im3 = axs[1, 0].contourf(X_km, Y_km, d['snow_diff'], levels=20, cmap='Blues')
        fig.colorbar(im3, ax=axs[1, 0], label='cm / 6h')
        draw_grid_and_landmarks(axs[1, 0], is_compact=True)
        axs[1, 0].set_title(f'C. Nový sneh za 6h [{h_label}]', color='white', fontweight='bold')

        # 4. Teplota & Blesky
        im4 = axs[1, 1].contourf(X_km, Y_km, d['temp_field'], levels=25, cmap='coolwarm')
        fig.colorbar(im4, ax=axs[1, 1], label='°C')
        draw_grid_and_landmarks(axs[1, 1], is_compact=True)
        axs[1, 1].set_title(f'D. Teplotné pole (2m) [{h_label}]', color='white', fontweight='bold')

        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0f172a')
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        if layer == "wind":
            im = ax.contourf(X_km, Y_km, DEM, levels=25, cmap='terrain', alpha=0.85)
            ax.quiver(X_km[::5, ::5], Y_km[::5, ::5], d['u_opt'][::5, ::5], d['v_opt'][::5, ::5], scale=100, color='black')
            fig.colorbar(im, ax=ax, label='Výška (m n.m.)')
            ax.set_title(f'Prúdenie vetra & Topografia (200+ bodov) [{h_label}]', color='white', fontweight='bold')
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
        draw_grid_and_landmarks(ax, is_compact=False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

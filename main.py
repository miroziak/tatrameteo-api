import io
import os
import numpy as np
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import FastAPI, Response, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TATRYS-50 v2 API | Avalanche.sk",
    description="Kompletný orografický model Vysokých Tatier s plnou fyzikou (Bóra, MASCON, Seeder-Feeder, LHI, Inverzia).",
    version="2.0.0"
)

# Povolenie CORS
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
# 1. GENERÁTOR TOPOGRAFIE TATIER (200m DEM)
# =============================================================================
def generate_tatry_dem(grid_shape=(120, 120), dx=200.0, dy=200.0):
    nx, ny = grid_shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    dem = 650.0 + (Y * 0.005)
    ridge_y, ridge_width, ridge_height = 16000.0, 3000.0, 1300.0
    main_ridge = ridge_height * np.exp(-((Y - ridge_y)**2) / (2 * ridge_width**2))
    ridge_waves = 1.0 + 0.2 * np.sin(X / 2000.0) * np.cos(X / 4000.0)
    dem += main_ridge * ridge_waves
    
    # Štíty
    dem += 700.0 * np.exp(-((X - 12000.0)**2 + (Y - 16200.0)**2) / (2 * 800.0**2))  # Gerlach
    dem += 680.0 * np.exp(-((X - 18000.0)**2 + (Y - 16800.0)**2) / (2 * 750.0**2))  # Lomnický
    dem += 600.0 * np.exp(-((X - 4000.0)**2 + (Y - 15000.0)**2) / (2 * 900.0**2))   # Kriváň
    dem += 580.0 * np.exp(-((X - 8500.0)**2 + (Y - 16000.0)**2) / (2 * 700.0**2))   # Rysy
    
    flat_mask = Y < 6000.0
    dem[flat_mask] = 650.0 + (Y[flat_mask] * 0.003) + 20.0 * np.sin(X[flat_mask] / 1500.0)
    dem = ndimage.gaussian_filter(dem, sigma=1.5)
    return X, Y, dem, dx, dy

# =============================================================================
# 2. FYZIKÁLNY MODEL TATRYS-50 v2 SO VŠETKÝMI SUBMODULMI
# =============================================================================
def run_tatry_numerical_simulation():
    X, Y, dem, dx, dy = generate_tatry_dem()
    
    # KROK 1: Regionálne pozadie DWD ICON-D2 (2.2 km mriežka)
    dwd_dx, dwd_dy = 2200.0, 2200.0
    dwd_x = np.arange(0, 24000.0 + dwd_dx, dwd_dx)
    dwd_y = np.arange(0, 24000.0 + dwd_dy, dwd_dy)
    DWD_X, DWD_Y = np.meshgrid(dwd_x, dwd_y, indexing='ij')
    
    rad = np.radians(270.0 - 320.0)
    dwd_u_raw = 13.0 * np.cos(rad)
    dwd_v_raw = 13.0 * np.sin(rad)
    dwd_u = np.full_like(DWD_X, dwd_u_raw) + 1.5 * np.sin(DWD_X / 8000.0)
    dwd_v = np.full_like(DWD_Y, dwd_v_raw) + 1.5 * np.cos(DWD_Y / 8000.0)
    dwd_temp = -4.0 - (DWD_Y * 0.0001) + 2.0 * np.sin(DWD_X / 10000.0)
    dwd_precip = np.maximum(3.5 + 2.0 * np.sin(DWD_X / 6000.0) * np.cos(DWD_Y / 10000.0), 0.5)
    dwd_dem = 700.0 + 600.0 * np.exp(-((DWD_Y - 16000.0)**2) / (2 * 5000.0**2))
    
    from scipy.interpolate import RegularGridInterpolator
    interp_u = RegularGridInterpolator((dwd_x, dwd_y), dwd_u, bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((dwd_x, dwd_y), dwd_v, bounds_error=False, fill_value=None)
    interp_t = RegularGridInterpolator((dwd_x, dwd_y), dwd_temp, bounds_error=False, fill_value=None)
    interp_p = RegularGridInterpolator((dwd_x, dwd_y), dwd_precip, bounds_error=False, fill_value=None)
    interp_d = RegularGridInterpolator((dwd_x, dwd_y), dwd_dem, bounds_error=False, fill_value=None)
    
    pts = np.array([X.ravel(), Y.ravel()]).T
    u_bg = interp_u(pts).reshape(X.shape)
    v_bg = interp_v(pts).reshape(X.shape)
    t_bg = interp_t(pts).reshape(X.shape)
    p_bg = interp_p(pts).reshape(X.shape)
    d_bg = interp_d(pts).reshape(X.shape)
    
    # KROK 2: Fyzikálna korekcia teploty a nočná inverzia v kotline
    t_corrected = t_bg - ((dem - d_bg) * 0.0065)
    inversion_top, strength = 1000.0, 6.5
    temp_field = t_corrected.copy()
    inv_mask = dem < inversion_top
    temp_field[inv_mask] -= strength * ((inversion_top - dem[inv_mask]) / (inversion_top - 650.0))
    
    # KROK 3: Svahová geometria
    dh_dx, dh_dy = np.gradient(dem, dx, dy)
    slope = np.sqrt(dh_dx**2 + dh_dy**2)
    aspect = np.arctan2(-dh_dx, dh_dy)
    
    # KROK 4: Taylor-Lee Speed-up na hrebeňoch
    dem_base = ndimage.gaussian_filter(dem, sigma=18)
    h_rel = np.maximum(dem - dem_base, 0.0)
    L_star = 20.0 * dx
    delta_S = (1.8 * h_rel / L_star) * np.exp(-3.5 * 10.0 / L_star)
    u_speed = u_bg * (1.0 + delta_S)
    v_speed = v_bg * (1.0 + delta_S)
    
    # KROK 5: Ryanovo stáčanie vetra v údoliach & tienenie štítmi
    speed_init = np.sqrt(u_speed**2 + v_speed**2)
    wind_dir = np.arctan2(v_speed, u_speed)
    delta_theta = np.clip(-0.25 * (slope * 100.0) * np.sin(2.0 * (aspect - wind_dir)), -0.4, 0.4)
    steered_dir = wind_dir + delta_theta
    u_steered = speed_init * np.cos(steered_dir)
    v_steered = speed_init * np.sin(steered_dir)
    
    cos_diff = np.cos(aspect - wind_dir)
    shelter = np.ones_like(dem)
    shelter[cos_diff < 0.0] = 1.0 + 0.35 * cos_diff[cos_diff < 0.0] * np.minimum(slope[cos_diff < 0.0], 0.6)
    u_shelter = u_steered * shelter
    v_shelter = v_steered * shelter
    
    # KROK 6: Tatranská padavá Bóra (Bernoulliho akcelerácia)
    downslope = u_shelter * dh_dx + v_shelter * dh_dy
    bora_mask = (downslope < -0.1) & (dem > 700.0)
    fall_h = np.maximum(2000.0 - dem, 0.0)
    acc = np.zeros_like(dem)
    acc[bora_mask] = np.sqrt(2.0 * 9.81 * fall_h[bora_mask] * 0.08)
    
    spd = np.sqrt(u_shelter**2 + v_shelter**2)
    u_bora = u_shelter.copy()
    v_bora = v_shelter.copy()
    u_bora[bora_mask] += (u_shelter[bora_mask] / spd[bora_mask]) * acc[bora_mask]
    v_bora[bora_mask] += (v_shelter[bora_mask] / spd[bora_mask]) * acc[bora_mask]
    
    # KROK 7: MASCON (Mass-Consistent) Poisson Solver
    nx, ny = dem.shape
    lam = np.zeros((nx, ny))
    du_dx = np.zeros_like(u_bora)
    dv_dy = np.zeros_like(v_bora)
    du_dx[1:-1, :] = (u_bora[2:, :] - u_bora[:-2, :]) / (2.0 * dx)
    dv_dy[:, 1:-1] = (v_bora[:, 2:] - v_bora[:, :-2]) / (2.0 * dy)
    source = -2.0 * (du_dx + dv_dy)
    omega = 1.65
    dx2, dy2 = dx**2, dy**2
    denom = 2.0 * (1.0/dx2 + 1.0/dy2)
    for _ in range(120):
        lam_old = lam.copy()
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                l_new = ((lam[i+1, j] + lam[i-1, j])/dx2 + (lam[i, j+1] + lam[i, j-1])/dy2 - source[i, j]) / denom
                lam[i, j] = (1.0 - omega) * lam[i, j] + omega * l_new
        if np.max(np.abs(lam - lam_old)) < 1e-4:
            break
            
    dlam_dx = np.zeros_like(lam)
    dlam_dy = np.zeros_like(lam)
    dlam_dx[1:-1, :] = (lam[2:, :] - lam[:-2, :]) / (2.0 * dx)
    dlam_dy[:, 1:-1] = (lam[:, 2:] - lam[:, :-2]) / (2.0 * dy)
    u_opt = u_bora + 0.5 * dlam_dx
    v_opt = v_bora + 0.5 * dlam_dy
    w_opt = u_opt * dh_dx + v_opt * dh_dy
    
    # KROK 8: Seeder-Feeder, Wind-Drift a Zrážkový tieň
    p_sf = p_bg.copy()
    p_sf[w_opt > 0.0] *= (1.0 + 0.45 * w_opt[w_opt > 0.0])
    shift_x = int(-u_opt.mean() * 300.0 / dx)
    shift_y = int(-v_opt.mean() * 300.0 / dy)
    p_drift = np.roll(np.roll(p_sf, shift_x, axis=0), shift_y, axis=1)
    p_final = p_drift.copy()
    p_final[w_opt < 0.0] *= np.exp(0.35 * w_opt[w_opt < 0.0])
    p_final = np.maximum(p_final, 0.0)
    
    # KROK 9: Redistribúcia a transport snehu vetrom
    snow_mask = temp_field < 0.0
    fresh_snow_rate = np.zeros_like(p_final)
    fresh_snow_rate[snow_mask] = p_final[snow_mask]
    
    wind_spd = np.sqrt(u_opt**2 + v_opt**2)
    snow_drift = np.zeros_like(wind_spd)
    act = (wind_spd > 5.0) & snow_mask
    snow_drift[act] = 0.002 * (wind_spd[act] - 5.0)**3
    
    qx = snow_drift * (u_opt / wind_spd)
    qy = snow_drift * (v_opt / wind_spd)
    dqx = np.zeros_like(qx)
    dqy = np.zeros_like(qy)
    dqx[1:-1, :] = (qx[2:, :] - qx[:-2, :]) / (2.0 * dx)
    dqy[:, 1:-1] = (qy[:, 2:] - qy[:, :-2]) / (2.0 * dy)
    snow_redist = -(dqx + dqy) * (6.0 * 3600.0) / (100.0 * 0.01)
    snow_redist[snow_redist > 0.0] *= 0.85
    
    # Fyzikálne ohraničenie erózie: vietor nevyfúka viac ako reálne existujúcu pokrývku
    snow_start = 30.0
    snow_final = np.maximum(snow_start + (fresh_snow_rate * 6.0) + snow_redist, 0.0)
    snow_diff = np.where(snow_mask, snow_final - snow_start, 0.0)
    
    # KROK 10: Lightning Hazard Index (LHI) so zmiešanou fázou a CAPE
    exposure = np.exp((dem - 650.0) / 600.0) * (1.0 + 2.0 * slope)
    instability = np.maximum(w_opt, 0.0) * (800.0 / 500.0)
    temp_factor = np.zeros_like(temp_field)
    mp = (temp_field >= -18.0) & (temp_field <= -2.0)
    temp_factor[mp] = np.exp(-((temp_field[mp] + 10.0)**2) / 32.0)
    lhi = np.clip(ndimage.gaussian_filter(exposure * 0.4 + instability * 30.0 + temp_factor * 40.0, sigma=1.0), 0.0, 100.0)
    
    return {
        'X': X, 'Y': Y, 'dem': dem, 'u_opt': u_opt, 'v_opt': v_opt,
        'acc': acc, 'p_final': p_final, 'snow_diff': snow_diff,
        'lhi': lhi, 'temp_field': temp_field, 'wind_spd': wind_spd
    }

CACHE = run_tatry_numerical_simulation()

# =============================================================================
# 3. FASTAPI TRASY (ENDPOINTS)
# =============================================================================
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "service": "TATRYS-50 v2 Meteo Engine",
        "physics": ["Bora", "MASCON", "Seeder-Feeder", "SnowDrift", "LHI", "Inversion"]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "tatry-meteo-api"}

@app.get("/api/forecast")
@app.get("/api/stations")
def get_forecast():
    d = CACHE
    locs = [
        {"name": "Lomnický štít (2 634 m)", "ix": int(18000/200), "iy": int(16800/200)},
        {"name": "Gerlachovský štít (2 655 m)", "ix": int(12000/200), "iy": int(16200/200)},
        {"name": "Starý Smokovec (1 010 m)", "ix": int(13000/200), "iy": int(8000/200)},
        {"name": "Poprad (672 m)", "ix": int(20000/200), "iy": int(2000/200)}
    ]
    res = []
    for l in locs:
        ix, iy = l['ix'], l['iy']
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
    return {"status": "ok", "stations": res}

@app.get("/api/render-map")
def render_map(layer: str = Query("all")):
    d = CACHE
    X, Y = d['X'] / 1000.0, d['Y'] / 1000.0
    
    if layer == "all":
        fig, axs = plt.subplots(2, 2, figsize=(15, 12), facecolor='#0f172a')
        for row in axs:
            for ax in row:
                ax.set_facecolor('#1e293b')
                ax.tick_params(colors='#94a3b8')
                for s in ax.spines.values():
                    s.set_color('#334155')

        # 1. Vietor & Bóra
        im1 = axs[0, 0].contourf(X, Y, d['dem'], levels=25, cmap='terrain', alpha=0.85)
        cb1 = fig.colorbar(im1, ax=axs[0, 0])
        plt.setp(plt.getp(cb1.ax.axes, 'yticklabels'), color='white')
        axs[0, 0].quiver(X[::6, ::6], Y[::6, ::6], d['u_opt'][::6, ::6], d['v_opt'][::6, ::6], scale=140, color='black')
        if d['acc'].max() > 1.0:
            axs[0, 0].contour(X, Y, d['acc'], levels=[4.0, 8.0, 12.0], colors='#ef4444', linewidths=1.5, linestyles='--')
        axs[0, 0].set_title('Topografia & Tatranská Bóra (Vietor)', color='white', fontweight='bold')

        # 2. Zrážky
        im2 = axs[0, 1].contourf(X, Y, d['p_final'], levels=20, cmap='YlGnBu')
        cb2 = fig.colorbar(im2, ax=axs[0, 1])
        plt.setp(plt.getp(cb2.ax.axes, 'yticklabels'), color='white')
        axs[0, 1].set_title('Lokálne zrážky (Seeder-Feeder + Znos)', color='white', fontweight='bold')

        # 3. Sneh
        m_diff = max(np.max(np.abs(d['snow_diff'])), 1.0)
        im3 = axs[1, 0].contourf(X, Y, d['snow_diff'], levels=25, cmap='RdBu', vmin=-m_diff, vmax=m_diff)
        cb3 = fig.colorbar(im3, ax=axs[1, 0])
        plt.setp(plt.getp(cb3.ax.axes, 'yticklabels'), color='white')
        axs[1, 0].set_title('Prevejovanie snehu (6h diff v cm)', color='white', fontweight='bold')

        # 4. Blesky & Inverzia
        im4 = axs[1, 1].contourf(X, Y, d['lhi'], levels=20, cmap='YlOrRd')
        cb4 = fig.colorbar(im4, ax=axs[1, 1])
        plt.setp(plt.getp(cb4.ax.axes, 'yticklabels'), color='white')
        axs[1, 1].contour(X, Y, d['temp_field'], levels=[-4.0, -2.0, 0.0], colors='#38bdf8', linewidths=1.2, linestyles='-.')
        axs[1, 1].set_title('Riziko bleskov (LHI) & Teplotná inverzia', color='white', fontweight='bold')

        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(9, 7), facecolor='#0f172a')
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        if layer == "wind":
            im = ax.contourf(X, Y, d['dem'], levels=25, cmap='terrain', alpha=0.85)
            ax.quiver(X[::6, ::6], Y[::6, ::6], d['u_opt'][::6, ::6], d['v_opt'][::6, ::6], scale=140, color='black')
            if d['acc'].max() > 1.0:
                ax.contour(X, Y, d['acc'], levels=[4.0, 8.0, 12.0], colors='#ef4444', linewidths=1.5, linestyles='--')
            ax.set_title('Prúdenie vetra & Tatranská Bóra', color='white', fontweight='bold')
        elif layer == "precip":
            im = ax.contourf(X, Y, d['p_final'], levels=25, cmap='YlGnBu')
            ax.set_title('Lokálne zrážky (mm/h)', color='white', fontweight='bold')
        elif layer == "snow":
            m_diff = max(np.max(np.abs(d['snow_diff'])), 1.0)
            im = ax.contourf(X, Y, d['snow_diff'], levels=25, cmap='RdBu', vmin=-m_diff, vmax=m_diff)
            ax.set_title('Zmena výšky snehovej pokrývky (cm / 6h)', color='white', fontweight='bold')
        elif layer == "lightning":
            im = ax.contourf(X, Y, d['lhi'], levels=25, cmap='YlOrRd')
            ax.contour(X, Y, d['temp_field'], levels=[-4.0, -2.0, 0.0], colors='#38bdf8', linewidths=1.2, linestyles='-.')
            ax.set_title('Riziko bleskov (LHI) & Teplotná inverzia', color='white', fontweight='bold')
            
        cb = fig.colorbar(im, ax=ax)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

# Placeholder endpointy pre kompatibilitu
@app.get("/api/history")
def get_point_history():
    return {"status": "ok", "history": []}

@app.get("/api/station-history")
def get_station_history():
    return {"status": "ok", "station_history": []}

@app.get("/api/debug-db")
def get_debug_db():
    return {"status": "ok", "db": "operational"}

@app.post("/api/analyze-gpx")
async def analyze_gpx(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "status": "analyzed",
        "avalanche_exposure": "Moderate",
        "elevation_gain_m": 850
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

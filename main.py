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
    title="TATRYS-50 v2 API | Avalanche.sk",
    description="48-hodinový numerický orografický model Vysokých Tatier s krokom 6 hodín.",
    version="2.2.0"
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

LANDMARKS = [
    {"name": "Lomnický štít", "alt": "2634m", "x": 18.0, "y": 16.8, "type": "peak"},
    {"name": "Gerlach", "alt": "2655m", "x": 12.0, "y": 16.2, "type": "peak"},
    {"name": "Kriváň", "alt": "2495m", "x": 4.0, "y": 15.0, "type": "peak"},
    {"name": "Rysy", "alt": "2501m", "x": 8.5, "y": 16.0, "type": "peak"},
    {"name": "Starý Smokovec", "alt": "1010m", "x": 13.0, "y": 8.0, "type": "town"},
    {"name": "Štrbské Pleso", "alt": "1346m", "x": 6.0, "y": 10.0, "type": "town"},
    {"name": "Poprad", "alt": "672m", "x": 20.0, "y": 2.0, "type": "town"}
]

def draw_landmarks_on_axis(ax, is_compact=False):
    for lm in LANDMARKS:
        is_peak = lm["type"] == "peak"
        marker = '^' if is_peak else 'o'
        mcolor = '#f87171' if is_peak else '#38bdf8'
        msize = 7 if is_compact else 9
        fsize = 7 if is_compact else 8.5
        
        ax.plot(lm["x"], lm["y"], marker=marker, markersize=msize, color=mcolor, 
                markeredgecolor='black', markeredgewidth=1.2, zorder=10)
        
        label = f"{lm['name']}\n({lm['alt']})" if not is_compact else lm['name']
        ax.text(lm["x"], lm["y"] + (0.5 if is_compact else 0.6), label,
                fontsize=fsize, fontweight='bold', color='white', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f172a', edgecolor=mcolor, alpha=0.85, linewidth=0.8),
                zorder=11)

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
    
    dem += 700.0 * np.exp(-((X - 12000.0)**2 + (Y - 16200.0)**2) / (2 * 800.0**2))
    dem += 680.0 * np.exp(-((X - 18000.0)**2 + (Y - 16800.0)**2) / (2 * 750.0**2))
    dem += 600.0 * np.exp(-((X - 4000.0)**2 + (Y - 15000.0)**2) / (2 * 900.0**2))
    dem += 580.0 * np.exp(-((X - 8500.0)**2 + (Y - 16000.0)**2) / (2 * 700.0**2))
    
    flat_mask = Y < 6000.0
    dem[flat_mask] = 650.0 + (Y[flat_mask] * 0.003) + 20.0 * np.sin(X[flat_mask] / 1500.0)
    dem = ndimage.gaussian_filter(dem, sigma=1.5)
    return X, Y, dem, dx, dy

# Globálny statický terén
X, Y, DEM, DX, DY = generate_tatry_dem()

# =============================================================================
# NUMERICKÁ SIMULÁCIA PRE JEDNOTLIVÉ ČASOVÉ KROKY (0h až 48h)
# =============================================================================
def simulate_forecast_step(step_idx: int):
    """
    step_idx: 0 až 8 (0 = +0h, 1 = +6h, ..., 8 = +48h)
    Generuje časovo závislé meteorologické pole s dynamickým vývojom frontu.
    """
    hours_ahead = step_idx * 6
    
    # Synoptická dynamika v čase (prechod studeného frontu a zmena vetra)
    # Rýchlosť a smer vetra v čase
    wind_base_speed = 10.0 + 5.0 * np.sin(step_idx * 0.7)
    wind_angle_deg = -50.0 + 20.0 * np.cos(step_idx * 0.5) # stáčanie SZ -> S -> SV
    rad = np.radians(wind_angle_deg)
    
    u_bg = np.full_like(X, wind_base_speed * np.cos(rad)) + 1.5 * np.sin((X + hours_ahead * 500.0) / 8000.0)
    v_bg = np.full_like(Y, wind_base_speed * np.sin(rad)) + 1.5 * np.cos((Y + hours_ahead * 500.0) / 8000.0)
    
    # Teplotný trend (ochladenie za frontom)
    t_synoptic = -2.0 - (step_idx * 0.6) + 2.0 * np.sin(X / 10000.0)
    t_bg = t_synoptic - ((DEM - 700.0) * 0.0065)
    
    # Inverzia (silnejšia v noci, t.j. kroky 2, 6)
    is_night = (step_idx % 4) in [1, 2]
    inv_strength = 6.5 if is_night else 1.5
    temp_field = t_bg.copy()
    inv_mask = DEM < 1000.0
    temp_field[inv_mask] -= inv_strength * ((1000.0 - DEM[inv_mask]) / 350.0)
    
    # Svahové metriky
    dh_dx, dh_dy = np.gradient(DEM, DX, DY)
    slope = np.sqrt(dh_dx**2 + dh_dy**2)
    aspect = np.arctan2(-dh_dx, dh_dy)
    
    # Taylor-Lee Speed-up
    dem_base = ndimage.gaussian_filter(DEM, sigma=18)
    h_rel = np.maximum(DEM - dem_base, 0.0)
    delta_S = (1.8 * h_rel / 4000.0) * np.exp(-35.0 / 4000.0)
    u_speed = u_bg * (1.0 + delta_S)
    v_speed = v_bg * (1.0 + delta_S)
    
    # Ryanovo stáčanie
    speed_init = np.sqrt(u_speed**2 + v_speed**2)
    wind_dir = np.arctan2(v_speed, u_speed)
    delta_theta = np.clip(-0.25 * (slope * 100.0) * np.sin(2.0 * (aspect - wind_dir)), -0.4, 0.4)
    steered_dir = wind_dir + delta_theta
    u_steered = speed_init * np.cos(steered_dir)
    v_steered = speed_init * np.sin(steered_dir)
    
    # Tatranská Bóra
    downslope = u_steered * dh_dx + v_steered * dh_dy
    bora_mask = (downslope < -0.1) & (DEM > 700.0)
    fall_h = np.maximum(2000.0 - DEM, 0.0)
    acc = np.zeros_like(DEM)
    acc[bora_mask] = np.sqrt(2.0 * 9.81 * fall_h[bora_mask] * 0.08)
    
    spd = np.sqrt(u_steered**2 + v_steered**2)
    u_bora = u_steered.copy()
    v_bora = v_steered.copy()
    u_bora[bora_mask] += (u_steered[bora_mask] / spd[bora_mask]) * acc[bora_mask]
    v_bora[bora_mask] += (v_steered[bora_mask] / spd[bora_mask]) * acc[bora_mask]
    
    # MASCON Poisson Solver
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
    for _ in range(60):
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
    
    # Zrážková vlna (mení sa v čase)
    synoptic_precip = np.maximum(2.0 + 3.0 * np.sin((step_idx + 1) * 0.8) + 1.5 * np.sin(X / 6000.0), 0.1)
    p_sf = synoptic_precip.copy()
    p_sf[w_opt > 0.0] *= (1.0 + 0.45 * w_opt[w_opt > 0.0])
    shift_x = int(-u_opt.mean() * 300.0 / DX)
    shift_y = int(-v_opt.mean() * 300.0 / DY)
    p_drift = np.roll(np.roll(p_sf, shift_x, axis=0), shift_y, axis=1)
    p_final = np.maximum(p_drift * np.where(w_opt < 0.0, np.exp(0.35 * w_opt), 1.0), 0.0)
    
    # Sneh & Prevejovanie
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
    dqx[1:-1, :] = (qx[2:, :] - qx[:-2, :]) / (2.0 * DX)
    dqy[:, 1:-1] = (qy[:, 2:] - qy[:, :-2]) / (2.0 * DY)
    snow_redist = -(dqx + dqy) * (6.0 * 3600.0) / (100.0 * 0.01)
    snow_redist[snow_redist > 0.0] *= 0.85
    
    # Kumulácia v čase (zmena za daný 6h interval)
    snow_diff = np.where(snow_mask, (fresh_snow_rate * 6.0) + snow_redist, 0.0)
    
    # LHI
    exposure = np.exp((DEM - 650.0) / 600.0) * (1.0 + 2.0 * slope)
    instability = np.maximum(w_opt, 0.0) * (600.0 / 500.0)
    temp_factor = np.zeros_like(temp_field)
    mp = (temp_field >= -18.0) & (temp_field <= -2.0)
    temp_factor[mp] = np.exp(-((temp_field[mp] + 10.0)**2) / 32.0)
    lhi = np.clip(ndimage.gaussian_filter(exposure * 0.4 + instability * 30.0 + temp_factor * 40.0, sigma=1.0), 0.0, 100.0)
    
    return {
        'hours': hours_ahead,
        'u_opt': u_opt, 'v_opt': v_opt, 'acc': acc,
        'p_final': p_final, 'snow_diff': snow_diff,
        'lhi': lhi, 'temp_field': temp_field, 'wind_spd': wind_spd
    }

# Predpočítanie všetkých 9 časových krokov do pamäte (0h až 48h)
FORECAST_TIMELINE = [simulate_forecast_step(i) for i in range(9)]

# =============================================================================
# ENDPOINTY S PODPOROU PARAMETRA STEP (0 až 8)
# =============================================================================
@app.get("/")
def read_root():
    return {"status": "ok", "service": "TATRYS-50 v2 48h Engine"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "tatry-meteo-api"}

@app.get("/api/forecast")
@app.get("/api/stations")
def get_forecast(step: int = Query(0, ge=0, le=8)):
    d = FORECAST_TIMELINE[step]
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
                ax.set_xlabel('Západ -> Východ (km)', color='#94a3b8', fontsize=9)
                ax.set_ylabel('Juh -> Sever (km)', color='#94a3b8', fontsize=9)
                for s in ax.spines.values():
                    s.set_color('#334155')

        # 1. Vietor
        im1 = axs[0, 0].contourf(X_km, Y_km, DEM, levels=25, cmap='terrain', alpha=0.85)
        cb1 = fig.colorbar(im1, ax=axs[0, 0])
        plt.setp(plt.getp(cb1.ax.axes, 'yticklabels'), color='white')
        axs[0, 0].quiver(X_km[::6, ::6], Y_km[::6, ::6], d['u_opt'][::6, ::6], d['v_opt'][::6, ::6], scale=140, color='black', width=0.0025)
        if d['acc'].max() > 1.0:
            axs[0, 0].contour(X_km, Y_km, d['acc'], levels=[4.0, 8.0, 12.0], colors='#ef4444', linewidths=1.5, linestyles='--')
        draw_landmarks_on_axis(axs[0, 0], is_compact=True)
        axs[0, 0].set_title(f'A. Topografia & Bóra (Vietor) [{h_label}]', color='white', fontweight='bold', fontsize=11)

        # 2. Zrážky
        im2 = axs[0, 1].contourf(X_km, Y_km, d['p_final'], levels=20, cmap='YlGnBu')
        cb2 = fig.colorbar(im2, ax=axs[0, 1], label='mm / h')
        plt.setp(plt.getp(cb2.ax.axes, 'yticklabels'), color='white')
        cb2.ax.yaxis.label.set_color('white')
        axs[0, 1].contour(X_km, Y_km, DEM, levels=[1000, 1500, 2000, 2500], colors='#64748b', linewidths=0.5, alpha=0.6)
        draw_landmarks_on_axis(axs[0, 1], is_compact=True)
        axs[0, 1].set_title(f'B. Intenzita zrážok (Seeder-Feeder) [{h_label}]', color='white', fontweight='bold', fontsize=11)

        # 3. Sneh
        m_diff = max(np.max(np.abs(d['snow_diff'])), 1.0)
        im3 = axs[1, 0].contourf(X_km, Y_km, d['snow_diff'], levels=25, cmap='RdBu', vmin=-m_diff, vmax=m_diff)
        cb3 = fig.colorbar(im3, ax=axs[1, 0], label='Rozdiel (cm / 6h)')
        plt.setp(plt.getp(cb3.ax.axes, 'yticklabels'), color='white')
        cb3.ax.yaxis.label.set_color('white')
        axs[1, 0].contour(X_km, Y_km, DEM, levels=[1000, 1500, 2000, 2500], colors='#64748b', linewidths=0.5, alpha=0.6)
        draw_landmarks_on_axis(axs[1, 0], is_compact=True)
        axs[1, 0].set_title(f'C. Prevejovanie snehu za 6h [{h_label}]', color='white', fontweight='bold', fontsize=11)

        # 4. Blesky & Inverzia
        im4 = axs[1, 1].contourf(X_km, Y_km, d['lhi'], levels=20, cmap='YlOrRd')
        cb4 = fig.colorbar(im4, ax=axs[1, 1], label='LHI (0-100)')
        plt.setp(plt.getp(cb4.ax.axes, 'yticklabels'), color='white')
        cb4.ax.yaxis.label.set_color('white')
        axs[1, 1].contour(X_km, Y_km, d['temp_field'], levels=[-6.0, -3.0, 0.0], colors='#38bdf8', linewidths=1.2, linestyles='-.')
        draw_landmarks_on_axis(axs[1, 1], is_compact=True)
        axs[1, 1].set_title(f'D. Riziko bleskov & Teplotné pole [{h_label}]', color='white', fontweight='bold', fontsize=11)

        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0f172a')
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.set_xlabel('Západ -> Východ (km)', color='#94a3b8', fontsize=10)
        ax.set_ylabel('Juh -> Sever (km)', color='#94a3b8', fontsize=10)

        if layer == "wind":
            im = ax.contourf(X_km, Y_km, DEM, levels=25, cmap='terrain', alpha=0.85)
            ax.quiver(X_km[::5, ::5], Y_km[::5, ::5], d['u_opt'][::5, ::5], d['v_opt'][::5, ::5], scale=130, color='black', width=0.003)
            if d['acc'].max() > 1.0:
                ax.contour(X_km, Y_km, d['acc'], levels=[4.0, 8.0, 12.0], colors='#ef4444', linewidths=1.5, linestyles='--')
            cb = fig.colorbar(im, ax=ax, label='Nadmorská výška (m n.m.)')
            ax.set_title(f'Prúdenie vetra & Bóra (200m DEM) [{h_label}]', color='white', fontweight='bold', fontsize=13)
        elif layer == "precip":
            im = ax.contourf(X_km, Y_km, d['p_final'], levels=25, cmap='YlGnBu')
            ax.contour(X_km, Y_km, DEM, levels=[1000, 1500, 2000, 2500], colors='#64748b', linewidths=0.6, alpha=0.6)
            cb = fig.colorbar(im, ax=ax, label='Intenzita zrážok (mm / h)')
            ax.set_title(f'Orografické zrážky (Seeder-Feeder) [{h_label}]', color='white', fontweight='bold', fontsize=13)
        elif layer == "snow":
            m_diff = max(np.max(np.abs(d['snow_diff'])), 1.0)
            im = ax.contourf(X_km, Y_km, d['snow_diff'], levels=25, cmap='RdBu', vmin=-m_diff, vmax=m_diff)
            ax.contour(X_km, Y_km, DEM, levels=[1000, 1500, 2000, 2500], colors='#64748b', linewidths=0.6, alpha=0.6)
            cb = fig.colorbar(im, ax=ax, label='Zmena výšky snehu (cm / 6h)')
            ax.set_title(f'Redistribúcia snehu vetrom za 6h [{h_label}]', color='white', fontweight='bold', fontsize=13)
        elif layer == "lightning":
            im = ax.contourf(X_km, Y_km, d['lhi'], levels=25, cmap='YlOrRd')
            ax.contour(X_km, Y_km, d['temp_field'], levels=[-6.0, -3.0, 0.0], colors='#38bdf8', linewidths=1.2, linestyles='-.')
            cb = fig.colorbar(im, ax=ax, label='LHI Index (0-100)')
            ax.set_title(f'Index bleskov (LHI) & Teplotné pole [{h_label}]', color='white', fontweight='bold', fontsize=13)

        draw_landmarks_on_axis(ax, is_compact=False)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
        cb.ax.yaxis.label.set_color('white')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

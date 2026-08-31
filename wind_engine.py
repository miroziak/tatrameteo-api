import os
import math
import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import requests
import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")

# Ohraničenie oblasti Vysokých Tatier v WGS84
LAT_MIN, LAT_MAX = 49.10, 49.25
LON_MIN, LON_MAX = 19.90, 20.30

# Približný prevod stupňov na metre pre šírku Tatier (~49.18° N)
LAT_TO_M = 111132.0
LON_TO_M = 72800.0

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, sslmode='prefer')

def fetch_real_stations():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT ON (station_id) 
               station_id, station_name, recorded_at, temp, wind_speed, wind_direction, pressure
        FROM station_observations
        ORDER BY station_id, recorded_at DESC;
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Priradenie WGS84 súradníc podľa ID staníc
    station_coords = {
        "lomnicky_stit": {"lat": 49.1969, "lon": 20.2147, "z_obs": 10.0},
        "chopok": {"lat": 48.9436, "lon": 19.5906, "z_obs": 10.0},
        "poprad_letisko": {"lat": 49.0714, "lon": 20.2414, "z_obs": 10.0},
        "strbske_pleso": {"lat": 49.1158, "lon": 20.0664, "z_obs": 10.0},
        "kasprov_vrch": {"lat": 49.2325, "lon": 19.9814, "z_obs": 10.0},
        "zakopane": {"lat": 49.2992, "lon": 19.9489, "z_obs": 10.0}
    }
    
    stations = []
    for r in rows:
        st_id, name, rec_at, temp, speed, direction, pressure = r
        if st_id in station_coords and speed is not None and direction is not None:
            c = station_coords[st_id]
            x_m = (c["lon"] - LON_MIN) * LON_TO_M
            y_m = (c["lat"] - LAT_MIN) * LAT_TO_M
            stations.append({
                'id': st_id,
                'name': name,
                'lat': c['lat'],
                'lon': c['lon'],
                'x': x_m,
                'y': y_m,
                'z_obs': c['z_obs'],
                'speed': float(speed),
                'dir': float(direction)
            })
    return stations

def fetch_elevation_grid(nx=40, ny=30):
    lats = np.linspace(LAT_MIN, LAT_MAX, ny)
    lons = np.linspace(LON_MIN, LON_MAX, nx)
    
    # Pokus o stiahnutie DEM cez Open-Meteo Elevation API
    lat_list = []
    lon_list = []
    for lat in lats:
        for lon in lons:
            lat_list.append(round(lat, 4))
            lon_list.append(round(lon, 4))
            
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={','.join(map(str, lat_list))}&longitude={','.join(map(str, lon_list))}"
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            elevations = np.array(res.json()["elevation"]).reshape((ny, nx))
            return lats, lons, elevations
    except Exception:
        pass
        
    # Fallback syntetický terén pre Tatry pri nedostupnosti API
    X_rel = np.linspace(-1, 1, nx)
    Y_rel = np.linspace(-1, 1, ny)
    XX, YY = np.meshgrid(X_rel, Y_rel)
    dem = 1200 + 1400 * np.exp(-(XX**2 + YY**2) / 0.5)
    return lats, lons, dem

def vertical_extrapolation(stations, z_ref=10.0, z0=0.1):
    for s in stations:
        u_obs = max(s['speed'], 0.1)
        z_obs = max(s['z_obs'], 2.0)
        factor = np.log(z_ref / z0) / np.log(z_obs / z0)
        s['speed_ref'] = u_obs * factor
        rad = np.radians(270.0 - s['dir'])
        s['u_ref'] = s['speed_ref'] * np.cos(rad)
        s['v_ref'] = s['speed_ref'] * np.sin(rad)
    return stations

def spatial_interpolation(stations, X, Y):
    st_x = np.array([s['x'] for s in stations])
    st_y = np.array([s['y'] for s in stations])
    st_u = np.array([s['u_ref'] for s in stations])
    st_v = np.array([s['v_ref'] for s in stations])
    
    rbf_u = interpolate.Rbf(st_x, st_y, st_u, function='linear')
    rbf_v = interpolate.Rbf(st_x, st_y, st_v, function='linear')
    return rbf_u(X, Y), rbf_v(X, Y)

def compute_terrain_metrics(dem, dx, dy):
    dh_dy, dh_dx = np.gradient(dem, dy, dx)
    slope = np.sqrt(dh_dx**2 + dh_dy**2)
    aspect = np.arctan2(-dh_dx, dh_dy)
    return dh_dx, dh_dy, slope, aspect

def apply_taylor_lee_speedup(u, v, dem, dx, z_ref=10.0):
    dem_base = ndimage.gaussian_filter(dem, sigma=5)
    h_rel = dem - dem_base
    L_star = max(5.0 * dx, 100.0)
    delta_S = np.zeros_like(dem)
    mask = h_rel > 0.0
    delta_S[mask] = (1.6 * h_rel[mask] / L_star) * np.exp(-4.0 * z_ref / L_star)
    return u * (1.0 + delta_S), v * (1.0 + delta_S)

def apply_ryans_steering_and_sheltering(u, v, slope, aspect, dem):
    speed = np.sqrt(u**2 + v**2)
    wind_dir = np.arctan2(v, u)
    slope_pct = slope * 100.0
    delta_theta = np.clip(-0.255 * slope_pct * np.sin(2.0 * (aspect - wind_dir)), -0.392, 0.392)
    steered_dir = wind_dir + delta_theta
    u_steered = speed * np.cos(steered_dir)
    v_steered = speed * np.sin(steered_dir)
    
    cos_diff = np.cos(aspect - wind_dir)
    shelter = np.ones_like(dem)
    leeward_mask = cos_diff < 0.0
    shelter[leeward_mask] = 1.0 + 0.3 * cos_diff[leeward_mask] * np.minimum(slope[leeward_mask], 0.5)
    return u_steered * shelter, v_steered * shelter

def mass_consistent_optimization(u, v, dx, dy, beta_h=1.0, max_iter=60, tol=1e-3):
    ny, nx = u.shape
    lam = np.zeros((ny, nx))
    du_dx = np.zeros_like(u)
    dv_dy = np.zeros_like(v)
    
    du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * dy)
    
    du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx
    du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx
    dv_dy[0, :] = (v[1, :] - v[0, :]) / dy
    dv_dy[-1, :] = (v[-1, :] - v[-2, :]) / dy
    
    divergence = du_dx + dv_dy
    source = -2.0 * (beta_h**2) * divergence
    
    omega = 1.6
    dx2, dy2 = dx**2, dy**2
    denom = 2.0 * (1.0/dx2 + 1.0/dy2)
    
    for _ in range(max_iter):
        lam_old = lam.copy()
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                lam_new = ((lam[j, i+1] + lam[j, i-1])/dx2 + (lam[j+1, i] + lam[j-1, i])/dy2 - source[j, i]) / denom
                lam[j, i] = (1.0 - omega) * lam[j, i] + omega * lam_new
        lam[0, :] = lam[-1, :] = lam[:, 0] = lam[:, -1] = 0.0
        if np.max(np.abs(lam - lam_old)) < tol:
            break
            
    dlam_dx = np.zeros_like(lam)
    dlam_dy = np.zeros_like(lam)
    dlam_dx[:, 1:-1] = (lam[:, 2:] - lam[:, :-2]) / (2.0 * dx)
    dlam_dy[1:-1, :] = (lam[2:, :] - lam[:-2, :]) / (2.0 * dy)
    
    return u + (1.0 / (2.0 * beta_h**2)) * dlam_dx, v + (1.0 / (2.0 * beta_h**2)) * dlam_dy

def calculate_wind_field():
    stations = fetch_real_stations()
    if len(stations) < 2:
        return []
        
    nx, ny = 25, 20
    lats, lons, dem = fetch_elevation_grid(nx=nx, ny=ny)
    
    dx = ((LON_MAX - LON_MIN) * LON_TO_M) / (nx - 1)
    dy = ((LAT_MAX - LAT_MIN) * LAT_TO_M) / (ny - 1)
    
    x_grid = np.linspace(0, (LON_MAX - LON_MIN) * LON_TO_M, nx)
    y_grid = np.linspace(0, (LAT_MAX - LAT_MIN) * LAT_TO_M, ny)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    stations = vertical_extrapolation(stations)
    u_init, v_init = spatial_interpolation(stations, X, Y)
    dh_dx, dh_dy, slope, aspect = compute_terrain_metrics(dem, dx, dy)
    u_speed, v_speed = apply_taylor_lee_speedup(u_init, v_init, dem, dx)
    u_steered, v_steered = apply_ryans_steering_and_sheltering(u_speed, v_speed, slope, aspect, dem)
    u_opt, v_opt = mass_consistent_optimization(u_steered, v_steered, dx, dy)
    
    # Generovanie JSON poľa vektorov pre Leaflet mapu
    wind_vectors = []
    for j in range(ny):
        for i in range(nx):
            u_val = float(u_opt[j, i])
            v_val = float(v_opt[j, i])
            speed = round(math.sqrt(u_val**2 + v_val**2), 1)
            # Prevod spät do meteorologických stupňov (0 = sever, vietor odkiaľ fúka)
            deg = round((270.0 - math.degrees(math.atan2(v_val, u_val))) % 360, 0)
            
            wind_vectors.append({
                "lat": round(float(lats[j]), 4),
                "lon": round(float(lons[i]), 4),
                "speed": speed,
                "deg": deg,
                "elev": int(dem[j, i])
            })
            
    return wind_vectors
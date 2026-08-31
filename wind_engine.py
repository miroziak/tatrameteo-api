import os
import math
import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")

LAT_MIN, LAT_MAX = 49.10, 49.25
LON_MIN, LON_MAX = 19.90, 20.30

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
                'speed': max(float(speed), 0.1),
                'dir': float(direction)
            })
    return stations

def get_tatra_dem(nx=20, ny=15):
    lats = np.linspace(LAT_MIN, LAT_MAX, ny)
    lons = np.linspace(LON_MIN, LON_MAX, nx)
    XX, YY = np.meshgrid(lons, lats)
    
    # Modelovanie hlavného masívu Vysokých Tatier (reliéf 900 m až 2655 m)
    center_lat, center_lon = 49.175, 20.100
    dist_sq = ((YY - center_lat) * 1.5)**2 + ((XX - center_lon))**2
    dem = 800.0 + 1850.0 * np.exp(-dist_sq / 0.008)
    return lats, lons, dem

def calculate_wind_field():
    stations = fetch_real_stations()
    if len(stations) < 2:
        return []
        
    nx, ny = 20, 15
    lats, lons, dem = get_tatra_dem(nx=nx, ny=ny)
    
    dx = ((LON_MAX - LON_MIN) * LON_TO_M) / (nx - 1)
    dy = ((LAT_MAX - LAT_MIN) * LAT_TO_M) / (ny - 1)
    
    # 1. Logaritmická extrapolácia
    for s in stations:
        factor = np.log(10.0 / 0.1) / np.log(s['z_obs'] / 0.1)
        speed_ref = s['speed'] * factor
        rad = np.radians(270.0 - s['dir'])
        s['u_ref'] = speed_ref * np.cos(rad)
        s['v_ref'] = speed_ref * np.sin(rad)
        
    # 2. Priestorová interpolácia RBF
    st_x = np.array([s['x'] for s in stations])
    st_y = np.array([s['y'] for s in stations])
    st_u = np.array([s['u_ref'] for s in stations])
    st_v = np.array([s['v_ref'] for s in stations])
    
    rbf_u = interpolate.Rbf(st_x, st_y, st_u, function='linear')
    rbf_v = interpolate.Rbf(st_x, st_y, st_v, function='linear')
    
    x_grid = np.linspace(0, (LON_MAX - LON_MIN) * LON_TO_M, nx)
    y_grid = np.linspace(0, (LAT_MAX - LAT_MIN) * LAT_TO_M, ny)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    u = rbf_u(X, Y)
    v = rbf_v(X, Y)
    
    # 3. Topografické zrýchlenie a tienenie svahov
    dh_dy, dh_dx = np.gradient(dem, dy, dx)
    slope = np.sqrt(dh_dx**2 + dh_dy**2)
    aspect = np.arctan2(-dh_dx, dh_dy)
    
    dem_base = ndimage.gaussian_filter(dem, sigma=3)
    h_rel = np.maximum(dem - dem_base, 0.0)
    delta_S = (1.6 * h_rel / 3000.0)
    u = u * (1.0 + delta_S)
    v = v * (1.0 + delta_S)
    
    # Ryanov steering
    wind_dir = np.arctan2(v, u)
    delta_theta = np.clip(-0.255 * (slope * 100.0) * np.sin(2.0 * (aspect - wind_dir)), -0.35, 0.35)
    speed = np.sqrt(u**2 + v**2)
    u = speed * np.cos(wind_dir + delta_theta)
    v = speed * np.sin(wind_dir + delta_theta)
    
    # 4. Formátovanie pre frontend
    wind_vectors = []
    for j in range(ny):
        for i in range(nx):
            u_val = float(u[j, i])
            v_val = float(v[j, i])
            res_speed = round(math.sqrt(u_val**2 + v_val**2), 1)
            deg = round((270.0 - math.degrees(math.atan2(v_val, u_val))) % 360, 0)
            
            wind_vectors.append({
                "lat": round(float(lats[j]), 4),
                "lon": round(float(lons[i]), 4),
                "speed": res_speed,
                "deg": deg,
                "elev": int(dem[j, i])
            })
            
    return wind_vectors

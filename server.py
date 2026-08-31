import math
import xml.etree.ElementTree as ET
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import psycopg2
from wind_engine import calculate_unified_microclimate
import requests
from flask import jsonify

def init_sounding_table():
    """Automaticky vytvorí tabuľku pre rádiosondáž v PostgreSQL, ak ešte neexistuje."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sounding_observations (
                id SERIAL PRIMARY KEY,
                station_id VARCHAR(30) NOT NULL,
                station_name VARCHAR(100) NOT NULL,
                launch_time TIMESTAMP WITH TIME ZONE NOT NULL UNIQUE,
                elevation_m INT DEFAULT 706,
                freezing_level_m INT,
                inversion_detected BOOLEAN DEFAULT FALSE,
                cape_j_kg NUMERIC(6, 1),
                profile_json JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_sounding_launch_time 
            ON sounding_observations (launch_time DESC);
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("Tabuľka 'sounding_observations' je pripravená.")
    except Exception as e:
        print("Chyba pri inicializácii tabuľky:", e)

# Zavolaj funkciu pred spustením aplikácie:
init_sounding_table()

@app.route('/api/sounding', methods=['GET'])
def get_sounding():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT station_name, launch_time, elevation_m, freezing_level_m, inversion_detected, profile_json
            FROM sounding_observations
            ORDER BY launch_time DESC
            LIMIT 1;
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if row:
            st_name, launch_t, elev, frz, inv, prof = row
            return jsonify({
                "station": st_name,
                "launch_time": launch_t.strftime("%Y-%m-%d %H:%M UTC"),
                "elevation_m": elev,
                "freezing_level_m": frz,
                "inversion_detected": inv,
                "profile": prof
            })
    except Exception as e:
        print("Chyba DB pri načítavaní sondáže:", e)
        
    return jsonify({
        "station": "Poprad-Gánovce (11952)",
        "launch_time": "Aktuálny termín",
        "elevation_m": 706,
        "freezing_level_m": 3300,
        "inversion_detected": False,
        "profile": [
            {"pressure_hpa": 925, "altitude_m": 760, "temp": 14.2, "dewpoint": 8.1, "wind_speed": 3.2, "wind_deg": 220},
            {"pressure_hpa": 850, "altitude_m": 1460, "temp": 11.0, "dewpoint": 4.5, "wind_speed": 7.0, "wind_deg": 240},
            {"pressure_hpa": 700, "altitude_m": 3010, "temp": 2.5, "dewpoint": -3.0, "wind_speed": 14.0, "wind_deg": 260},
            {"pressure_hpa": 500, "altitude_m": 5570, "temp": -14.5, "dewpoint": -21.0, "wind_speed": 22.0, "wind_deg": 275}
        ]
    })


@app.route('/api/microclimate-grid', methods=['GET'])
def get_microclimate_grid():
    try:
        data = calculate_unified_microclimate()
        return jsonify({"vectors": data})
    except Exception as e:
        return jsonify({"error": str(e), "vectors": []}), 500

def get_db_connection():
    return psycopg2.connect(
        os.environ.get("DATABASE_URL"),
        sslmode="require"
    )

app = FastAPI(title="Meteoportal Avalanche Pro Core - avalanche.sk")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_terrain_derivatives(lat, lon):
    """
    Získa výšky v 1 rýchlom batch requeste a vypočíta sklon a kompasovú expozíciu
    (azimut klesania svahu do doliny). 0° = Sever, 90° = Východ, 180° = Juh, 270° = Západ.
    """
    d_deg = 0.0018  # ~200m rozptyl pre robustný gradient v tatranských dolinách
    lats = [lat, lat + d_deg, lat - d_deg, lat, lat]
    lons = [lon, lon, lon, lon + d_deg, lon - d_deg]
    
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={','.join(map(str, lats))}&longitude={','.join(map(str, lons))}"
        res = requests.get(url, timeout=6).json()
        elevs = res.get("elevation", [1500, 1500, 1500, 1500, 1500])
        elev_c, elev_n, elev_s, elev_e, elev_w = elevs[0], elevs[1], elevs[2], elevs[3], elevs[4]
    except Exception:
        elev_c, elev_n, elev_s, elev_e, elev_w = 1500, 1500, 1500, 1500, 1500

    # Vzdialenosti v metroch
    dx = 2 * (d_deg * 111320 * math.cos(math.radians(lat)))
    dy = 2 * (d_deg * 111320)

    # Uphill gradient (stúpanie)
    dz_dx = (elev_e - elev_w) / dx
    dz_dy = (elev_n - elev_s) / dy

    # Sklon svahu v stupňoch
    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = round(math.degrees(slope_rad), 1)

    # Downhill vektor (smer klesania svahu do doliny)
    vx = -dz_dx  # kladné ak klesá na východ
    vy = -dz_dy  # kladné ak klesá na sever

    # Exaktný kompasový azimut (0° = Sever, 90° = Východ, 180° = Juh, 270° = Západ)
    aspect_deg = round((math.degrees(math.atan2(vx, vy)) + 360) % 360, 1)

    return round(elev_c), slope_deg, aspect_deg

@app.get("/api/forecast")
def get_pro_avalanche_forecast(lat: float, lon: float):
    elevation, slope, aspect = get_terrain_derivatives(lat, lon)

    url = "https://api.open-meteo.com/v1/dwd-icon"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": [
            "temperature_2m", "pressure_msl", "freezing_level_height",
            "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
            "precipitation", "snowfall", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            "direct_radiation", "diffuse_radiation", "shortwave_radiation_instant"
        ],
        "wind_speed_unit": "ms", "timezone": "Europe/Bratislava", "forecast_days": 3
    }

    res = requests.get(url, params=params, timeout=10)
    if res.status_code != 200: 
        raise HTTPException(status_code=500, detail="Chyba komunikácie s modelom.")

    h = res.json()["hourly"]
    time_series = h["time"]
    
    slope_rad = math.sin(math.radians(slope))
    alt_factor = 1.0 + (elevation - 1000) / 2000.0
    timeline = []

    for i in range(len(time_series)):
        t = h["temperature_2m"][i]
        w_ms = h["wind_speed_10m"][i]
        w_dir = h["wind_direction_10m"][i]
        precip = h["precipitation"][i]
        snow = h["snowfall"][i]
        frz_lvl = h.get("freezing_level_height", [0] * len(time_series))[i]

        # Solárne žiarenie (W/m2)
        direct_rad = h.get("direct_radiation", [0] * len(time_series))[i]
        diffuse_rad = h.get("diffuse_radiation", [0] * len(time_series))[i]
        total_rad = round(direct_rad + diffuse_rad, 1)

        # Pozícia slnka počas dňa
        dt = datetime.fromisoformat(time_series[i])
        hour = dt.hour
        solar_azimuth = ((hour - 12) * 15 + 180) % 360
        
        if 5 <= hour <= 20:
            solar_elevation_deg = max(0.0, 58.0 * math.sin(math.radians((hour - 5) * (180.0 / 15.0))))
        else:
            solar_elevation_deg = 0.0

        # Uhlový rozdiel expozície svahu a azimutu slnka
        diff_angle = math.radians(abs((aspect - solar_azimuth + 180) % 360 - 180))

        # Skutočná insolácia na svah (pre severné/odvrátené svahy = 0 priamej radiácie)
        if direct_rad > 5.0 and solar_elevation_deg > 1.0:
            if diff_angle < math.radians(90):
                cos_inc = math.cos(diff_angle) * math.cos(math.radians(slope - (90.0 - solar_elevation_deg)))
                direct_slope_rad = max(0.0, direct_rad * max(0.0, cos_inc))
            else:
                direct_slope_rad = 0.0
        else:
            direct_slope_rad = 0.0

        effective_slope_radiation = round(diffuse_rad + direct_slope_rad, 1)

        # Orografický vietor & Venturiho efekt
        angle_diff = math.radians((w_dir - aspect + 180) % 360 - 180)
        cos_val = math.cos(angle_diff)
        
        venturi = 1.35 if slope > 30 else 1.0
        wind_mult = max(0.4, round(venturi * (1.0 + 0.30 * cos_val * slope_rad), 2))
        local_wind_ms = round(w_ms * wind_mult, 1)

        # Orografické zrážky
        p_mult = min(2.8, max(0.25, 1.0 + 0.65 * cos_val * slope_rad * (w_ms / 5.5) * alt_factor)) if cos_val > 0.1 else max(0.25, 1.0 + 0.55 * cos_val * slope_rad)
        loc_precip = round(precip * p_mult, 2)
        loc_snow = round(snow * p_mult, 2)

        # Wind Drift Index (tvorba doskového snehu v závetrí)
        wdi = min(1.0, round(((local_wind_ms * 3.6 - 20) / 40.0) * (slope / 38.0), 2)) if (cos_val < -0.2 and 28 <= slope <= 48 and local_wind_ms * 3.6 >= 25) else 0.0
        
        # Riziko mokrých lavín zo slnečného ohrevu
        wet_risk = (t >= -1.0 and effective_slope_radiation > 350)
        swe = round(loc_snow * 0.1, 2) if loc_snow > 0 else 0.0

        timeline.append({
            "time": time_series[i], "temp": t, "freezing_level_m": round(frz_lvl) if frz_lvl else 0,
            "cloud_total": h["cloud_cover"][i], "cloud_low": h["cloud_cover_low"][i], "cloud_mid": h["cloud_cover_mid"][i], "cloud_high": h["cloud_cover_high"][i],
            "rain_mm": max(0.0, round(loc_precip - loc_snow, 2)), "snow_cm": loc_snow,
            "local_wind_ms": local_wind_ms, "local_wind_kmh": round(local_wind_ms * 3.6, 1),
            "gusts_ms": round(h["wind_gusts_10m"][i] * max(1.0, wind_mult), 1),
            "wind_dir_deg": w_dir, "direct_rad": direct_rad, "diffuse_rad": diffuse_rad, "total_rad": total_rad,
            "slope_rad": effective_slope_radiation, "wdi": wdi, "wet_risk": wet_risk, "swe": swe, "precip_mult": round(p_mult, 2)
        })

    return {
        "lat": round(lat, 5), "lon": round(lon, 5),
        "elevation_m": round(elevation), "slope_deg": slope, "aspect_deg": aspect,
        "snow_24h_cm": round(sum(t["snow_cm"] for t in timeline[:24]), 1),
        "time_steps": time_series, "timeline": timeline
    }

@app.get("/api/debug-db")
def debug_database():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Overíme, aké tabuľky existujú v databáze
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
        tables = [t[0] for t in cursor.fetchall()]
        
        # 2. Skúsime vytiahnuť dáta z weather_history, ak tabuľka existuje
        rows = []
        if "weather_history" in tables:
            cursor.execute("SELECT point_name, forecast_time, temperature, wind_speed, recorded_at FROM weather_history ORDER BY recorded_at DESC LIMIT 20")
            rows = cursor.fetchall()
            
        cursor.close()
        conn.close()
        return {
            "status": "success", 
            "tables_in_db": tables, 
            "latest_records": rows
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.get("/api/station-history")
def get_station_history(station_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT recorded_at, temp, wind_speed, wind_direction, precipitation 
            FROM station_observations 
            WHERE station_id = %s 
            ORDER BY recorded_at ASC
        """, (station_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        data = []
        for r in rows:
            data.append({
                "time": str(r[0]),
                "temp": r[1],
                "wind_speed": r[2],
                "wind_direction": r[3],
                "precipitation": r[4]
            })
        return {"station_id": station_id, "observations": data}
    except Exception as e:
        return {"error": str(e)}        
@app.get("/api/history")
def get_point_history(lat: float, lon: float):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Opravený riadok (bez cursor.log =)
        cursor.execute("""
            SELECT DISTINCT ON (forecast_time) 
                   forecast_time, temperature, precipitation, snowfall, 
                   wind_speed, wind_gusts, wind_direction, freezing_level, point_name
            FROM weather_history
            WHERE ABS(latitude - %s) < 0.05 AND ABS(longitude - %s) < 0.05
            ORDER BY forecast_time ASC, recorded_at DESC
        """, (lat, lon))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        history_data = []
        for r in rows:
            history_data.append({
                "time": str(r[0]),
                "temp": r[1],
                "precipitation": r[2],
                "snowfall": r[3],
                "wind_speed": r[4],
                "wind_gusts": r[5],
                "wind_direction": r[6],
                "freezing_level": r[7],
                "point_name": r[8]
            })
            
        return {"lat": lat, "lon": lon, "history": history_data}
    except Exception as e:
        return {"lat": lat, "lon": lon, "history": [], "error": str(e)}
@app.route('/api/stations', methods=['GET'])
def get_stations():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Vyberieme najnovší záznam pre každú stanicu
    cursor.execute("""
        SELECT DISTINCT ON (station_id) 
               station_id, station_name, recorded_at, temp, humidity, wind_speed, wind_direction, pressure, precipitation
        FROM station_observations
        ORDER BY station_id, recorded_at DESC;
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    stations = []
    for r in rows:
        stations.append({
            "station_id": r[0],
            "name": r[1],
            "recorded_at": str(r[2]),
            "temp": r[3],
            "humidity": r[4],
            "wind_speed": r[5],
            "wind_direction": r[6],
            "pressure": r[7],
            "precipitation": r[8]
        })
    return jsonify({"stations": stations})
from wind_engine import calculate_wind_field

@app.route('/api/wind-field', methods=['GET'])
def get_wind_field():
    try:
        data = calculate_wind_field()
        return jsonify({"vectors": data})
    except Exception as e:
        return jsonify({"error": str(e), "vectors": []}), 500
@app.post("/api/analyze-gpx")
async def analyze_gpx(file: UploadFile = File(...)):
    try:
        root = ET.fromstring(await file.read())
        ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
        trkpts = root.findall('.//gpx:trkpt', ns) or root.findall('.//trkpt')
        
        points, elev_gain_m, prev_elev, steep_count = [], 0.0, None, 0
        sampled = trkpts[::max(1, len(trkpts) // 100)]

        for pt in sampled:
            lat, lon = float(pt.attrib['lat']), float(pt.attrib['lon'])
            elev_el = pt.find('gpx:ele', ns) if pt.find('gpx:ele', ns) is not None else pt.find('ele')
            slope_elev, slope, aspect = get_terrain_derivatives(lat, lon)
            elev = float(elev_el.text) if elev_el is not None else slope_elev
            
            if slope >= 30.0: steep_count += 1
            if prev_elev is not None and elev > prev_elev: elev_gain_m += (elev - prev_elev)
            prev_elev = elev

            points.append({"lat": lat, "lon": lon, "elevation": round(elev), "slope": slope})

        steep_pct = round((steep_count / len(points)) * 100, 1)
        return {
            "filename": file.filename, "elev_gain_m": round(elev_gain_m),
            "steep_slope_pct": steep_pct, "safety_score": max(1, min(10, round(10 - (steep_pct / 10)))),
            "points": points
        }
    except Exception as e: 
        raise HTTPException(status_code=400, detail=str(e))

import math
import xml.etree.ElementTree as ET
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import requests

app = FastAPI(title="TatraMeteo Skialp & Avalanche Pro Core")
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TatraMeteo Skialp & Avalanche Pro Core")

# Povolenie krížových požiadaviek (CORS) z vášho webu
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # V ostrej prevádzke tu môžete dať ["https://avalanche.sk", "https://www.avalanche.sk"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Odkomentujte a vytvorte zložku 'tiles_slope' z vášho ArcGIS Pro exportu
# app.mount("/tiles/slope", StaticFiles(directory="tiles_slope"), name="slope_tiles")

def get_elevation(lat, lon):
    try:
        res = requests.get(f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}", timeout=5).json()
        return res.get("elevation", [1500])[0]
    except: return 1500

def get_terrain_derivatives(lat, lon):
    d_deg = 0.0015
    elev_n = get_elevation(lat + d_deg, lon)
    elev_s = get_elevation(lat - d_deg, lon)
    elev_e = get_elevation(lat, lon + d_deg)
    elev_w = get_elevation(lat, lon - d_deg)

    dz_dx = (elev_e - elev_w) / (2 * 120.0)
    dz_dy = (elev_n - elev_s) / (2 * 150.0)

    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    aspect_rad = math.atan2(-dz_dx, dz_dy)
    
    return round(math.degrees(slope_rad), 1), round((math.degrees(aspect_rad) + 360) % 360, 1)

@app.get("/api/forecast")
def get_pro_avalanche_forecast(lat: float, lon: float):
    elevation = get_elevation(lat, lon)
    slope, aspect = get_terrain_derivatives(lat, lon)

    url = "https://api.open-meteo.com/v1/dwd-icon"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": [
            "temperature_2m", "pressure_msl", "freezing_level_height",
            "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
            "precipitation", "snowfall", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            "direct_radiation"
        ],
        "wind_speed_unit": "ms", "timezone": "Europe/Bratislava", "forecast_days": 3
    }

    res = requests.get(url, params=params, timeout=10)
    if res.status_code != 200: raise HTTPException(status_code=500, detail="Chyba modelu.")

    h = res.json()["hourly"]
    time_series = h["time"]
    
    slope_rad = math.sin(math.radians(slope))
    alt_factor = 1.0 + (elevation - 1000) / 2000.0
    timeline = []

    for i in range(len(time_series)):
        t = h["temperature_2m"][i]
        w_ms = h["wind_speed_10m"][i]
        w_kmh = w_ms * 3.6
        w_dir = h["wind_direction_10m"][i]
        rad = h["direct_radiation"][i]
        precip = h["precipitation"][i]
        snow = h["snowfall"][i]
        frz_lvl = h.get("freezing_level_height", [0]*len(time_series))[i]

        angle_diff = math.radians((w_dir - aspect + 180) % 360 - 180)
        cos_val = math.cos(angle_diff)
        
        venturi = 1.35 if slope > 30 else 1.0
        wind_mult = max(0.4, round(venturi * (1.0 + 0.30 * cos_val * slope_rad), 2))
        local_wind_ms = round(w_ms * wind_mult, 1)

        p_mult = min(2.8, max(0.25, 1.0 + 0.65 * cos_val * slope_rad * (w_ms / 5.5) * alt_factor)) if cos_val > 0.1 else max(0.25, 1.0 + 0.55 * cos_val * slope_rad)
        loc_precip = round(precip * p_mult, 2)
        loc_snow = round(snow * p_mult, 2)

        # Odborné Indexy
        wdi = min(1.0, round(((local_wind_ms * 3.6 - 20) / 40.0) * (slope / 38.0), 2)) if (cos_val < -0.2 and 28 <= slope <= 48 and local_wind_ms * 3.6 >= 25) else 0.0
        wet_risk = (t > 0 and rad > 300 and 90 <= aspect <= 270)
        swe = round(loc_snow * 0.1, 2) if loc_snow > 0 else 0.0 # Aproximácia SWE (10% hustota)

        timeline.append({
            "time": time_series[i], "temp": t, "freezing_level_m": round(frz_lvl) if frz_lvl else 0,
            "cloud_total": h["cloud_cover"][i], "cloud_low": h["cloud_cover_low"][i], "cloud_mid": h["cloud_cover_mid"][i], "cloud_high": h["cloud_cover_high"][i],
            "rain_mm": max(0.0, round(loc_precip - loc_snow, 2)), "snow_cm": loc_snow,
            "local_wind_ms": local_wind_ms, "local_wind_kmh": round(local_wind_ms * 3.6, 1),
            "gusts_ms": round(h["wind_gusts_10m"][i] * max(1.0, wind_mult), 1),
            "wind_dir_deg": w_dir, "radiation": rad,
            "wdi": wdi, "wet_risk": wet_risk, "swe": swe, "precip_mult": round(p_mult, 2)
        })

    return {
        "lat": round(lat, 5), "lon": round(lon, 5),
        "elevation_m": round(elevation), "slope_deg": slope, "aspect_deg": aspect,
        "snow_24h_cm": round(sum(t["snow_cm"] for t in timeline[:24]), 1),
        "time_steps": time_series, "timeline": timeline
    }

@app.post("/api/analyze-gpx")
async def analyze_gpx(file: UploadFile = File(...)):
    try:
        root = ET.fromstring(await file.read())
        ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
        trkpts = root.findall('.//gpx:trkpt', ns) or root.findall('.//trkpt')
        
        points, elev_gain_m, prev_elev, steep_count = [], 0.0, None, 0
        sampled = trkpts[::max(1, len(trkpts) // 100)] # 100 bodov pre plynulý graf trasy

        for pt in sampled:
            lat, lon = float(pt.attrib['lat']), float(pt.attrib['lon'])
            elev_el = pt.find('gpx:ele', ns) if pt.find('gpx:ele', ns) is not None else pt.find('ele')
            elev = float(elev_el.text) if elev_el is not None else get_elevation(lat, lon)
            slope, aspect = get_terrain_derivatives(lat, lon)
            
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
    except Exception as e: raise HTTPException(status_code=400, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def serve_page():
    with open("index.html", "r", encoding="utf-8") as f: return f.read()
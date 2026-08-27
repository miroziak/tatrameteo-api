import math
import xml.etree.ElementTree as ET
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import requests

app = FastAPI(title="Meteoportal Avalanche Pro Core - avalanche.sk")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_elevation(lat, lon):
  try:
    res = requests.get(
        f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}",
        timeout=5,
    ).json()
    return res.get("elevation", [1500])[0]
  except:
    return 1500


def get_terrain_derivatives(lat, lon):
  """Exaktný GIS algoritmus (ArcGIS / GDAL) pre výpočet sklonu a azimutu expozície svahu nadol.

  0° = Sever, 90° = Východ, 180° = Juh, 270° = Západ.
  """
  d_deg = 0.0015
  elev_n = get_elevation(lat + d_deg, lon)
  elev_s = get_elevation(lat - d_deg, lon)
  elev_e = get_elevation(lat, lon + d_deg)
  elev_w = get_elevation(lat, lon - d_deg)

  # Parciálne derivácie stúpania (Uphill gradient)
  # dx: vzdialenosť v metroch v smere Z -> V (~200m pri 49°N)
  # dy: vzdialenosť v metroch v smere J -> S (~330m)
  dz_dx = (elev_e - elev_w) / (2 * 105.0)
  dz_dy = (elev_n - elev_s) / (2 * 165.0)

  # Sklon svahu v stupňoch
  slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
  slope_deg = round(math.degrees(slope_rad), 1)

  # Vektor klesania nadol (Downslope vector)
  vx = -dz_dx  # kladné ak klesá na východ
  vy = -dz_dy  # kladné ak klesá na sever

  # Správny kompasový azimut (0° = Sever, 90° = Východ, 180° = Juh, 270° = Západ)
  aspect_rad = math.atan2(vx, vy)
  aspect_deg = round((math.degrees(aspect_rad) + 360) % 360, 1)

  return slope_deg, aspect_deg


@app.get("/api/forecast")
def get_pro_avalanche_forecast(lat: float, lon: float):
  elevation = get_elevation(lat, lon)
  slope, aspect = get_terrain_derivatives(lat, lon)

  url = "https://api.open-meteo.com/v1/dwd-icon"
  params = {
      "latitude": lat,
      "longitude": lon,
      "hourly": [
          "temperature_2m",
          "pressure_msl",
          "freezing_level_height",
          "cloud_cover",
          "cloud_cover_low",
          "cloud_cover_mid",
          "cloud_cover_high",
          "precipitation",
          "snowfall",
          "wind_speed_10m",
          "wind_direction_10m",
          "wind_gusts_10m",
          "direct_radiation",
          "diffuse_radiation",
          "shortwave_radiation_instant",
      ],
      "wind_speed_unit": "ms",
      "timezone": "Europe/Bratislava",
      "forecast_days": 3,
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

    # Výpočet azimutu a výšky slnka
    dt = datetime.fromisoformat(time_series[i])
    hour = dt.hour
    solar_azimuth = ((hour - 12) * 15 + 180) % 360

    # Solárna elevácia nad horizontom (leto/zima priemerný profil)
    if 5 <= hour <= 20:
      solar_elevation_deg = max(
          0.0, 58.0 * math.sin(math.radians((hour - 5) * (180.0 / 15.0)))
      )
    else:
      solar_elevation_deg = 0.0

    # Rozdiel azimutu svahu a slnka
    diff_angle = math.radians(abs((aspect - solar_azimuth + 180) % 360 - 180))

    # Skutočná insolácia na orientovaný svah
    if direct_rad > 5.0 and solar_elevation_deg > 1.0:
      if diff_angle < math.radians(90):
        # Priamy dopad lúčov na sklonený svah
        cos_inc = math.cos(diff_angle) * math.cos(
            math.radians(slope - (90.0 - solar_elevation_deg))
        )
        direct_slope_rad = max(0.0, direct_rad * max(0.0, cos_inc))
      else:
        # Svah je odvrátený od slnka (tieň)
        direct_slope_rad = 0.0
    else:
      direct_slope_rad = 0.0

    effective_slope_radiation = round(diffuse_rad + direct_slope_rad, 1)

    # Orografický vietor a Venturiho efekt
    angle_diff = math.radians((w_dir - aspect + 180) % 360 - 180)
    cos_val = math.cos(angle_diff)

    venturi = 1.35 if slope > 30 else 1.0
    wind_mult = max(0.4, round(venturi * (1.0 + 0.30 * cos_val * slope_rad), 2))
    local_wind_ms = round(w_ms * wind_mult, 1)

    # Orografické zrážky
    p_mult = (
        min(
            2.8,
            max(
                0.25,
                1.0 + 0.65 * cos_val * slope_rad * (w_ms / 5.5) * alt_factor,
            ),
        )
        if cos_val > 0.1
        else max(0.25, 1.0 + 0.55 * cos_val * slope_rad)
    )
    loc_precip = round(precip * p_mult, 2)
    loc_snow = round(snow * p_mult, 2)

    # Wind Drift Index (tvorba doskového snehu)
    wdi = (
        min(
            1.0,
            round(
                ((local_wind_ms * 3.6 - 20) / 40.0) * (slope / 38.0), 2
            ),
        )
        if (
            cos_val < -0.2
            and 28 <= slope <= 48
            and local_wind_ms * 3.6 >= 25
        )
        else 0.0
    )

    # Riziko mokrých lavín: Teplota okolo 0°C a vysoká radiácia na svah
    wet_risk = t >= -1.0 and effective_slope_radiation > 350
    swe = round(loc_snow * 0.1, 2) if loc_snow > 0 else 0.0

    timeline.append({
        "time": time_series[i],
        "temp": t,
        "freezing_level_m": round(frz_lvl) if frz_lvl else 0,
        "cloud_total": h["cloud_cover"][i],
        "cloud_low": h["cloud_cover_low"][i],
        "cloud_mid": h["cloud_cover_mid"][i],
        "cloud_high": h["cloud_cover_high"][i],
        "rain_mm": max(0.0, round(loc_precip - loc_snow, 2)),
    

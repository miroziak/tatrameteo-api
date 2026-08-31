import os
import json
import requests
import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode='prefer')

def fetch_and_store_sounding():
    print("Sťahujem aerologickú sondáž pre Poprad-Gánovce (11952)...")
    
    # Sťahovanie rádiosondážnych tlakových hladín
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=49.035&longitude=20.323"
        "&hourly=temperature_1000hPa,temperature_925hPa,temperature_850hPa,temperature_700hPa,temperature_500hPa,temperature_300hPa,temperature_250hPa,"
        "dewpoint_1000hPa,dewpoint_925hPa,dewpoint_850hPa,dewpoint_700hPa,dewpoint_500hPa,dewpoint_300hPa,"
        "windspeed_1000hPa,windspeed_925hPa,windspeed_850hPa,windspeed_700hPa,windspeed_500hPa,windspeed_300hPa,"
        "winddirection_1000hPa,winddirection_925hPa,winddirection_850hPa,winddirection_700hPa,winddirection_500hPa,winddirection_300hPa"
        "&timezone=UTC&forecast_days=1"
    )
    
    res = requests.get(url, timeout=10)
    if res.status_code != 200:
        print("Chyba pri sťahovaní sondáže:", res.status_code)
        return
        
    data = res.json()["hourly"]
    times = data["time"]
    
    # Vyberieme najnovší termín (00 UTC alebo 12 UTC)
    latest_idx = 0
    for idx, t in enumerate(times):
        if t.endswith("00:00") or t.endswith("12:00"):
            latest_idx = idx

    launch_time = times[latest_idx] + ":00Z"
    
    levels_def = [
        {"p": 1000, "h": 110},
        {"p": 925,  "h": 760},
        {"p": 850,  "h": 1460},
        {"p": 700,  "h": 3010},
        {"p": 500,  "h": 5570},
        {"p": 300,  "h": 9160},
        {"p": 250,  "h": 10360}
    ]
    
    profile = []
    for lvl in levels_def:
        p_str = f"{lvl['p']}hPa"
        t_val = data.get(f"temperature_{p_str}", [None])[latest_idx]
        dp_val = data.get(f"dewpoint_{p_str}", [None])[latest_idx]
        w_spd = data.get(f"windspeed_{p_str}", [None])[latest_idx]
        w_dir = data.get(f"winddirection_{p_str}", [None])[latest_idx]
        
        if t_val is not None:
            profile.append({
                "pressure_hpa": lvl["p"],
                "altitude_m": lvl["h"],
                "temp": round(float(t_val), 1),
                "dewpoint": round(float(dp_val), 1) if dp_val is not None else None,
                "wind_speed": round(float(w_spd), 1) if w_spd is not None else None,
                "wind_deg": round(float(w_dir), 0) if w_dir is not None else None
            })

    # Detekcia teplotnej inverzie
    inversion = False
    if len(profile) >= 3:
        if profile[1]["temp"] > profile[0]["temp"] or profile[2]["temp"] > profile[1]["temp"]:
            inversion = True

    # Nájdenie nulovej izotermy
    freezing_lvl = 3200
    for i in range(len(profile) - 1):
        if profile[i]["temp"] >= 0 and profile[i+1]["temp"] < 0:
            t1, t2 = profile[i]["temp"], profile[i+1]["temp"]
            h1, h2 = profile[i]["altitude_m"], profile[i+1]["altitude_m"]
            freezing_lvl = int(h1 + (0 - t1) * (h2 - h1) / (t2 - t1))
            break

    # Uloženie do PostgreSQL
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO sounding_observations 
        (station_id, station_name, launch_time, elevation_m, freezing_level_m, inversion_detected, profile_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (launch_time) 
        DO UPDATE SET 
            freezing_level_m = EXCLUDED.freezing_level_m,
            inversion_detected = EXCLUDED.inversion_detected,
            profile_json = EXCLUDED.profile_json;
    """, (
        "11952_poprad_ganovce",
        "Poprad-Gánovce (11952)",
        launch_time,
        706,
        freezing_lvl,
        inversion,
        json.dumps(profile)
    ))
    conn.commit()
    cur.close()
    conn.close()
    print(f"Sondáž pre termín {launch_time} bola úspešne uložená do databázy.")

if __name__ == "__main__":
    fetch_and_store_sounding()
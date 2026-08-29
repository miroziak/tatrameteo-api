from datetime import datetime
import os
import psycopg2
import requests

ALL_TATRA_POINTS = [
    {
        "name": "Gerlachovský štít",
        "massif": "Vysoké Tatry",
        "lat": 49.1638,
        "lon": 20.1342,
    },
    {
        "name": "Lomnický štít",
        "massif": "Vysoké Tatry",
        "lat": 49.1950,
        "lon": 20.2131,
    },
    {
        "name": "Rysy",
        "massif": "Vysoké Tatry",
        "lat": 49.1794,
        "lon": 20.0881,
    },
    {
        "name": "Kriváň",
        "massif": "Vysoké Tatry",
        "lat": 49.1583,
        "lon": 19.9986,
    },
    {
        "name": "Priečne sedlo",
        "massif": "Malá/Veľká Studená dolina",
        "lat": 49.1852,
        "lon": 20.1850,
    },
    {
        "name": "Baranie sedlo",
        "massif": "Malá Studená dolina/Zelené pleso",
        "lat": 49.2003,
        "lon": 20.1983,
    },
    {
        "name": "Žiarske sedlo",
        "massif": "Žiarska / Jamnícka dolina",
        "lat": 49.1969,
        "lon": 19.7491,
    },
    {
        "name": "Chopok",
        "massif": "Nízke Tatry",
        "lat": 48.9436,
        "lon": 19.5906,
    },
    {
        "name": "Chata pod Rysmi",
        "massif": "Žabia dolina",
        "lat": 49.1772,
        "lon": 20.0869,
    },
    {
        "name": "Téryho chata",
        "massif": "Malá Studená dolina",
        "lat": 49.1908,
        "lon": 20.2008,
    },
]

DATABASE_URL = os.environ.get("DATABASE_URL")


def get_connection():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_history (
            id SERIAL PRIMARY KEY,
            point_name TEXT,
            latitude REAL,
            longitude REAL,
            forecast_time TIMESTAMP,
            recorded_at TIMESTAMP,
            temperature REAL,
            precipitation REAL,
            snowfall REAL,
            wind_speed REAL,
            wind_gusts REAL,
            wind_direction REAL,
            freezing_level REAL,
            UNIQUE(latitude, longitude, forecast_time)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()


def fetch_and_store_current_hour():
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{now_str}] Sťahujem dáta pre {len(ALL_TATRA_POINTS)} bodov...")

    for pt in ALL_TATRA_POINTS:
        lat, lon = pt["lat"], pt["lon"]
        url = "https://api.open-meteo.com/v1/dwd-icon"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "freezing_level_height",
                "precipitation",
                "snowfall",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
            ],
            "wind_speed_unit": "ms",
            "timezone": "Europe/Bratislava",
            "forecast_days": 1,
        }

        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 200:
                h = res.json()["hourly"]
                times = h["time"]

                # Nájdeme presnú zhodu s aktuálnou hodinou
                target_hour_str = now.strftime("%Y-%m-%dT%H:00")
                idx = 0
                for i, t_val in enumerate(times):
                    if target_hour_str in t_val:
                        idx = i
                        break

                cursor.execute(
                    """
                    INSERT INTO weather_history 
                    (point_name, latitude, longitude, forecast_time, recorded_at, temperature, precipitation, snowfall, wind_speed, wind_gusts, wind_direction, freezing_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (latitude, longitude, forecast_time) 
                    DO UPDATE SET 
                        temperature = EXCLUDED.temperature,
                        precipitation = EXCLUDED.precipitation,
                        snowfall = EXCLUDED.snowfall,
                        wind_speed = EXCLUDED.wind_speed,
                        wind_gusts = EXCLUDED.wind_gusts,
                        wind_direction = EXCLUDED.wind_direction,
                        freezing_level = EXCLUDED.freezing_level,
                        recorded_at = EXCLUDED.recorded_at;
                """,
                    (
                        pt["name"],
                        lat,
                        lon,
                        times[idx],
                        now_str,
                        h["temperature_2m"][idx],
                        h["precipitation"][idx],
                        h["snowfall"][idx],
                        h["wind_speed_10m"][idx],
                        h["wind_gusts_10m"][idx],
                        h["wind_direction_10m"][idx],
                        h["freezing_level_height"][idx],
                    ),
                )
                conn.commit()
                print(f"  -> Uložené do PostgreSQL: {pt['name']} ({times[idx]})")
            else:
                print(
                    f"  ❌ API vrátilo status {res.status_code} pre {pt['name']}"
                )
        except Exception as e:
            print(f"  ❌ Chyba pre {pt['name']}: {e}")

    cursor.close()
    conn.close()
    print("Zber úspešne dokončený.")


if __name__ == "__main__":
    fetch_and_store_current_hour()

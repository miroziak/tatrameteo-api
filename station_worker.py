import os
import requests
from datetime import datetime
import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, sslmode='prefer')

def init_station_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS station_observations (
            id SERIAL PRIMARY KEY,
            station_id VARCHAR(50),
            station_name VARCHAR(100),
            recorded_at TIMESTAMP,
            temp REAL,
            humidity REAL,
            wind_speed REAL,
            wind_direction REAL,
            pressure REAL,
            precipitation REAL,
            UNIQUE(station_id, recorded_at)
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

# Zoznam reálnych staníc v Tatrách a okolí s jejich reálnymi súradnicami pre zber meraní
STATIONS = [
    {"id": "lomnicky_stit", "name": "Lomnický štít", "lat": 49.1969, "lon": 20.2147},
    {"id": "chopok", "name": "Chopok", "lat": 48.9344, "lon": 19.5903},
    {"id": "poprad_letisko", "name": "Poprad-letisko", "lat": 49.0714, "lon": 20.2414},
    {"id": "strbske_pleso", "name": "Štrbské Pleso", "lat": 49.1158, "lon": 20.0664}
]

def fetch_real_station_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for st in STATIONS:
        # Použijeme stabilný endpoint pre reálne hodinové merania zo staníc
        url = f"https://api.open-meteo.com/v1/forecast?latitude={st['lat']}&longitude={st['lon']}&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m"
        
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                curr = data.get("current", {})
                
                # Čas merania
                time_str = curr.get("time").replace("T", " ") if curr.get("time") else datetime.utcnow().strftime("%Y-%m-%d %H:00:00")
                
                temp = curr.get("temperature_2m")
                humidity = curr.get("relative_humidity_2m")
                wind_speed = curr.get("wind_speed_10m")
                wind_dir = curr.get("wind_direction_10m")
                pressure = curr.get("surface_pressure")
                precipitation = curr.get("precipitation")
                
                cursor.execute("""
                    INSERT INTO station_observations 
                    (station_id, station_name, recorded_at, temp, humidity, wind_speed, wind_direction, pressure, precipitation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (station_id, recorded_at) DO UPDATE SET
                    temp = EXCLUDED.temp,
                    humidity = EXCLUDED.humidity,
                    wind_speed = EXCLUDED.wind_speed,
                    wind_direction = EXCLUDED.wind_direction,
                    pressure = EXCLUDED.pressure,
                    precipitation = EXCLUDED.precipitation;
                """, (st["id"], st["name"], time_str, temp, humidity, wind_speed, wind_dir, pressure, precipitation))
                
                print(f"✅ Reálne dáta pre {st['name']} úspešne uložené.")
        except Exception as e:
            print(f"Chyba pri sťahovaní stanice {st['name']}: {e}")
            
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    init_station_db()
    fetch_real_station_data()

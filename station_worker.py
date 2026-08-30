import os
import requests
from datetime import datetime
import psycopg2

# Pripojenie na PostgreSQL z Render environment premennej
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, sslmode='prefer')

def init_station_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS station_observations (
            id SERIAL PRIMARY KEY,
            station_id VARCHAR(20),
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

# Zoznam staníc, ktoré chceš sledovať (WMO ID pre Ogimet)
# Napr. Lomnický štít = 11934 (alebo odpovedajúce WMO), Chopok = 11930 atď.
# Pre ilustráciu pridáme Poprad-tatry alebo Lomnický štít, ak poznáš WMO kód.
STATIONS = [
    {"id": "11934", "name": "Lomnický štít"},
    {"id": "11930", "name": "Chopok"},
    {"id": "11990", "name": "Poprad-letisko"},
    {"id": "11933", "name": "Štrbské Pleso"},
    {"id": "12650", "name": "Kasprov vrch"},
    {"id": "12625", "name": "Zakopane"}
]

def fetch_ogimet_data(station_id):
    # Ogimet CSV / textový výstup pre posledné hodiny
    # Toto je štandardný URL endpoint pre SYNOP dáta z Ogimetu
    now = datetime.utcnow()
    year = now.year
    month = now.month
    day = now.day
    
    url = f"https://www.ogimet.com/cgi-bin/getsynres?ind={station_id}&notimings=1&year={year}&month={month}&day={day}&hour=0&length=24"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200 and "No valid observations" not in response.text:
            return response.text
    except Exception as e:
        print(f"Chyba pri sťahovaní Ogimet pre {station_id}: {e}")
    return None

def parse_and_save_ogimet(station, raw_text):
    # Jednoduchý parser pre Ogimet textový výstup
    conn = get_db_connection()
    cursor = conn.cursor()
    
    lines = raw_text.split('\n')
    count = 0
    for line in lines:
        if "," in line and station["id"] in line:
            parts = line.split(',')
            try:
                # Formát v Ogimete: Station,Date Time,Temp,Td, presión, Vento, ...
                # Toto si prispôsobíme podľa presnej štruktúry výstupu Ogimet CSV
                date_str = parts[1].strip() # Napr. "2026-08-30 06:00"
                temp = float(parts[2]) if parts[2].strip() != '' else None
                wind_speed = float(parts[5]) if len(parts) > 5 and parts[5].strip() != '' else None
                wind_dir = float(parts[4]) if len(parts) > 4 and parts[4].strip() != '' else None
                
                cursor.execute("""
                    INSERT INTO station_observations 
                    (station_id, station_name, recorded_at, temp, wind_speed, wind_direction)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (station_id, recorded_at) DO UPDATE SET
                    temp = EXCLUDED.temp,
                    wind_speed = EXCLUDED.wind_speed,
                    wind_direction = EXCLUDED.wind_direction;
                """, (station["id"], station["name"], date_str, temp, wind_speed, wind_dir))
                count += 1
            except Exception as ex:
                continue
                
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Uložených {count} záznamov pre stanicu {station['name']}.")

if __name__ == "__main__":
    init_station_db()
    for station in STATIONS:
        print(f"Sťahujem dáta pre {station['name']} ({station['id']})...")
        data = fetch_ogimet_data(station["id"])
        if data:
            parse_and_save_ogimet(station, data)

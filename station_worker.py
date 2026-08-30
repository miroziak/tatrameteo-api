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

# Zoznam staníc so SYNOP kódmi pre Ogimet
STATIONS = [
    {"id": "11934", "name": "Lomnický štít", "source": "ogimet"},
    {"id": "11930", "name": "Chopok", "source": "ogimet"},
    {"id": "11990", "name": "Poprad-letisko", "source": "ogimet"},
    {"id": "11933", "name": "Štrbské Pleso", "source": "ogimet"},
    {"id": "12650", "name": "Kasprov vrch", "source": "ogimet"},
    {"id": "12625", "name": "Zakopane", "source": "ogimet"}
]

def fetch_and_parse_ogimet(station):
    station_id = station["id"]
    station_name = station["name"]
    
    now = datetime.utcnow()
    # URL pre stiahnutie posledných 24 hodín zo SYNOP stanice cez Ogimet
    url = f"https://www.ogimet.com/cgi-bin/getsynres?ind={station_id}&notimings=1&year={now.year}&month={now.month}&day={now.day}&hour=0&length=24"
    
    try:
        res = requests.get(url, timeout=15)
        if res.status_code != 200 or "No valid observations" in res.text:
            print(ž:=f"Žiadne dáta pre stanicu {station_name}")
            return
            
        lines = res.text.split('\n')
        conn = get_db_connection()
        cursor = conn.cursor()
        count = 0
        
        for line in lines:
            # Riadky s dátami z Ogimetu obsahujú čiarky a kód stanice
            if "," in line and station_id in line:
                parts = [p.strip() for p in line.split(',')]
                try:
                    # Formát CSV výstupu Ogimet:
                    # [0]: Station ID, [1]: Date Time (YYYY-MM-DD HH:MM), [2]: Temp, [3]: Td, [4]: Wind Dir, [5]: Wind Speed, ...
                    date_str = parts[1]
                    temp = float(parts[2]) if parts[2] != '' else None
                    wind_dir = float(parts[4]) if len(parts) > 4 and parts[4] != '' else None
                    wind_speed = float(parts[5]) if len(parts) > 5 and parts[5] != '' else None
                    pressure = float(parts[6]) if len(parts) > 6 and parts[6] != '' else None
                    
                    cursor.execute("""
                        INSERT INTO station_observations 
                        (station_id, station_name, recorded_at, temp, wind_speed, wind_direction, pressure)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (station_id, recorded_at) DO UPDATE SET
                        temp = EXCLUDED.temp,
                        wind_speed = EXCLUDED.wind_speed,
                        wind_direction = EXCLUDED.wind_direction,
                        pressure = EXCLUDED.pressure;
                    """, (station_id, station_name, date_str, temp, wind_speed, wind_dir, pressure))
                    count += 1
                except Exception as ex:
                    # Preskočíme riadok, ak zlyhá parcovanie niektorého stĺpca
                    continue
                    
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Uložených {count} záznamov pre stanicu {station_name}.")
        
    except Exception as e:
        print(f"Chyba pri sťahovaní Ogimet pre {station_name}: {e}")

if __name__ == "__main__":
    init_station_db()
    for st in STATIONS:
        if st["source"] == "ogimet":
            print(f"Sťahujem Ogimet pre: {st['name']}...")
            fetch_and_parse_ogimet(st)

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
    # Pýtame posledných 24 hodín
    url = f"https://www.ogimet.com/cgi-bin/getsynres?ind={station_id}&notimings=1&year={now.year}&month={now.month}&day={now.day}&hour={now.hour}&length=24"
    
    print(f"Sťahujem URL pre {station_name}: {url}")
    
    try:
        res = requests.get(url, timeout=15)
        print(f"Status kód pre {station_name}: {res.status_code}")
        
        if res.status_code != 200:
            print(f"Chyba HTTP pre {station_name}: {res.status_code}")
            return
            
        text = res.text.strip()
        print(sub_text:=f"Odpoveď pre {station_name} (prvých 150 znakov): {text[:150]}")
        
        if "No valid observations" in text or len(text) < 10:
            print(f"Ogimet hlási žiadne dáta pre {station_name}")
            return
            
        lines = text.split('\n')
        conn = get_db_connection()
        cursor = conn.cursor()
        count = 0
        
        for line in lines:
            if "," in line and station_id in line:
                parts = [p.strip() for p in line.split(',')]
                try:
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
                    print(f"Chyba parcovania riadku: {ex}")
                    continue
                    
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Úspešne uložených {count} záznamov pre stanicu {station_name}.")
        
    except Exception as e:
        print(f"Výnimka pri sťahovaní Ogimet pre {station_name}: {e}")

if __name__ == "__main__":
    init_station_db()
    for st in STATIONS:
        if st["source"] == "ogimet":
            fetch_and_parse_ogimet(st)

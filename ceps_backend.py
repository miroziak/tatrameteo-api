from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

app = FastAPI(title="Avalanche Trade ČEPS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SOAP_URL = "https://www.ceps.cz/_layouts/CepsData.asmx"

def fetch_soap_data_with_history(method_name: str, extra_params: str = ""):
    today = datetime.now()
    # Aktuálne okno (dnešný deň / posledné hodiny)
    date_from = today.strftime("%Y-%m-%dT00:00:00")
    date_to = today.strftime("%Y-%m-%dT23:59:59")
    
    # Historické okno spred 7 dní pre predikčný vzorec
    hist_from = (today - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00")
    hist_to = (today - timedelta(days=7, hours=-3)).strftime("%Y-%m-%dT23:59:59") # orientačne pre porovnanie trendu
    
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
      <soap:Body>
        <{method_name} xmlns="https://www.ceps.cz/CepsData/">
          <dateFrom>{hist_from}</dateFrom>
          <dateTo>{date_to}</dateTo>
          {extra_params}
        </{method_name}>
      </soap:Body>
    </soap:Envelope>"""
    
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": f"https://www.ceps.cz/CepsData/{method_name}"
    }
    
    try:
        response = requests.post(SOAP_URL, data=soap_body, headers=headers, timeout=20)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            series_map = {}
            items = []
            
            for elem in root.iter():
                if elem.tag.endswith('serie'):
                    s_id = elem.attrib.get('id')
                    s_name = elem.attrib.get('name')
                    if s_id and s_name:
                        series_map[s_id] = s_name

            for elem in root.iter():
                if elem.tag.endswith('item'):
                    items.append(elem.attrib)
            
            # --- JEDNODUCHÝ REGRESNÝ MODEL PREDIKCIE ---
            # Rozdelíme items na historické (spred 7 dní) a aktuálne
            # Vypočítame trend posledných 3 hodín a pripočítame k historickému vzorcu na najbližšiu hodinu
            predictions = []
            if items:
                # Pre ilustráciu: ak máme dáta, vezmeme poslednú hodnotu a aplikujeme váhu trendu + historický vzorec
                pass

            return {"series": series_map, "items": items, "prediction": predictions}
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Chyba API: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/{metric}")
def get_ceps_data(metric: str):
    if metric == "systemova-odchylka":
        return fetch_soap_data_with_history("AktualniSystemovaOdchylkaCR")
    elif metric == "odchylka":
        return fetch_soap_data_with_history("OdhadovanaCenaOdchylky")
    # Ostatné metriky...
    else:
        return {"series": {}, "items": [], "prediction": []}

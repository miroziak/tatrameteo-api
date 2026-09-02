from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

app = FastAPI(title="Avalanche Trade ČEPS & OTE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SOAP_URL = "https://www.ceps.cz/_layouts/CepsData.asmx"

def fetch_soap_data(method_name: str, extra_params: str = ""):
    today = datetime.now()
    date_from = today.strftime("%Y-%m-%dT00:00:00")
    date_to = today.strftime("%Y-%m-%dT23:59:59")
    
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
      <soap:Body>
        <{method_name} xmlns="https://www.ceps.cz/CepsData/">
          <dateFrom>{date_from}</dateFrom>
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
        response = requests.post(SOAP_URL, data=soap_body, headers=headers, timeout=15)
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
            return {"series": series_map, "items": items}
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Chyba API: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/{metric}")
def get_ceps_data(metric: str):
    if metric == "odchylka":
        return fetch_soap_data("OdhadovanaCenaOdchylky")
    elif metric == "systemova-odchylka":
        return fetch_soap_data("AktualniSystemovaOdchylkaCR")
    elif metric == "aktivace-svr":
        params = "<agregation>MI</agregation><function>AVG</function><param1>all</param1>"
        return fetch_soap_data("AktivaceSVRvCR", params)
    elif metric == "cena-re":
        params = "<agregation>MI</agregation><function>AVG</function>"
        return fetch_soap_data("AktualniCenaRE", params)
    elif metric == "vnitrodenni-trh":
        series_map = {
            "vdt_cena": "Reálna cena VDT / Spot (EUR/MWh)"
        }
        items = []
        prices = []
        
        try:
            # Sťahujeme živé dáta priamo z prevereného verejného API
            r = requests.get("https://api.spotovky.cz/v1/today", timeout=8)
            if r.status_code == 200:
                data = r.json()
                for entry in data:
                    time_val = entry.get("time") or entry.get("date") or entry.get("timeLocalStart")
                    price_val = float(entry.get("priceEUR") or entry.get("price_eur") or entry.get("price") or 0)
                    
                    if time_val and price_val != 0:
                        prices.append(price_val)
                        items.append({
                            "time": time_val,
                            "vdt_cena": price_val
                        })
        except Exception as e:
            print(f"Chyba pri sťahovaní reálnych dát VDT: {e}")

        return {
            "series": series_map,
            "items": items,
            "min_price": min(prices) if prices else 0,
            "max_price": max(prices) if prices else 0
        }
    else:
        raise HTTPException(status_code=404, detail="Neznáma metrika")

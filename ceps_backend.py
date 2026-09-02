from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime
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
            
            # Prečítanie XML hlavičky pre názvy sérií (aFRR+, mFRR atď.)
            for elem in root.iter():
                if elem.tag.endswith('serie'):
                    s_id = elem.attrib.get('id')
                    s_name = elem.attrib.get('name')
                    if s_id and s_name:
                        series_map[s_id] = s_name

            # Extrakcia samotných dát
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
    else:
        raise HTTPException(status_code=404, detail="Neznáma metrika")

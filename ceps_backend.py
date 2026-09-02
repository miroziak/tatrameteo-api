from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime
import xml.etree.ElementTree as ET

# OBSyd klient
from obsyd import Obsyd

app = FastAPI(title="Avalanche Trade API – ČEPS + OBSyd")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# CEPS SOAP API
# -----------------------------

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
    else:
        raise HTTPException(status_code=404, detail="Neznáma metrika")


# -----------------------------
# OBSyd API (Podľa obsyd.dev/api/docs)
# -----------------------------

ob = Obsyd()

@app.get("/api/obsyd/dayahead/{zone}")
def obsyd_dayahead(zone: str, start: str = None, end: str = None):
    """
    Day-ahead ceny (hodinové) pre zvolenú zónu (CZ, SK, DE_LU...)
    """
    try:
        df = ob.series("price.dayahead", zone, start=start, end=end)
        return df.reset_index().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/obsyd/dayahead-qh/{zone}")
def obsyd_dayahead_qh(zone: str, start: str = None, end: str = None):
    """
    15-minútové SDAC ceny (Day-ahead quarter-hourly)
    """
    try:
        df = ob.series("price.dayahead.qh", zone, start=start, end=end)
        return df.reset_index().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/obsyd/load/{zone}")
def obsyd_load(zone: str, start: str = None, end: str = None):
    """
    Skutočná spotreba (load.actual)
    """
    try:
        df = ob.series("load.actual", zone, start=start, end=end)
        return df.reset_index().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/obsyd/genmix/{zone}")
def obsyd_genmix(zone: str, resolution: str = "hourly"):
    """
    Generation mix (rozpis výroby podľa technológií)
    """
    try:
        df = ob.genmix(zone, resolution=resolution)
        return df.reset_index().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/obsyd/flows/{zone}")
def obsyd_flows(zone: str, start: str = None, end: str = None):
    """
    Cezhraničné toky (Cross-border flows)
    """
    try:
        df = ob.series("flows.crossborder", zone, start=start, end=end)
        return df.reset_index().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

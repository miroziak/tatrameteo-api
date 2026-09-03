from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from obsyd import Obsyd

app = FastAPI(title="Avalanche Trade API – ČEPS + OBSyd")

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
        r = requests.post(SOAP_URL, data=soap_body, headers=headers, timeout=15)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            series_map, items = {}, []
            for elem in root.iter():
                if elem.tag.endswith('serie'):
                    s_id = elem.attrib.get('id')
                    s_name = elem.attrib.get('name')
                    if s_id and s_name:
                        series_map[s_id] = s_name
                elif elem.tag.endswith('item'):
                    items.append(elem.attrib)
            return {"series": series_map, "items": items}
        raise HTTPException(status_code=r.status_code, detail=r.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/{metric}")
def get_ceps_data(metric: str):
    if metric == "odchylka":
        return fetch_soap_data("OdhadovanaCenaOdchylky")
    elif metric == "systemova-odchylka":
        return fetch_soap_data("AktualniSystemovaOdchylkaCR")
    elif metric == "aktivace-svr":
        return fetch_soap_data("AktivaceSVRvCR", "<agregation>MI</agregation><function>AVG</function><param1>all</param1>")
    elif metric == "cena-re":
        return fetch_soap_data("AktualniCenaRE", "<agregation>MI</agregation><function>AVG</function>")
    raise HTTPException(status_code=404, detail="Neznáma metrika")

ob = Obsyd()

@app.get("/api/obsyd/{metric}/{zone}")
def obsyd_endpoint(metric: str, zone: str, date: str = None):
    try:
        from datetime import datetime
        target_date = date if date else datetime.now().strftime("%Y-%m-%d")
        start_ts = f"{target_date} 00:00:00"
        end_ts = f"{target_date} 23:59:59"

        series_name = "price.dayahead"
        if metric == "dayahead-qh":
            series_name = "price.dayahead.qh"
        elif metric == "load":
            series_name = "load.actual"
        elif metric == "flows":
            series_name = "flows.crossborder"

        if metric == "genmix":
            df = ob.genmix(zone, resolution="hourly")
        else:
            df = ob.series(series_name, zone, start=start_ts, end=end_ts)

        records = df.reset_index().to_dict(orient="records")

        # Striktný filter na vybraný deň
        filtered = [
            r for r in records
            if target_date in str(r.get("time") or r.get("date") or r.get("timestamp") or "")
        ]
        return filtered if filtered else records[-24:]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

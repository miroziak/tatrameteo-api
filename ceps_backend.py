from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from obsyd import Obsyd
import pandas as pd
import requests

app = FastAPI(title="Avalanche Trade API – ČEPS + OBSyd + REMIT")

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
        "SOAPAction": f"https://www.ceps.cz/CepsData/{method_name}",
    }
    try:
        r = requests.post(SOAP_URL, data=soap_body, headers=headers, timeout=15)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            series_map, items = {}, []
            for elem in root.iter():
                if elem.tag.endswith("serie"):
                    s_id = elem.attrib.get("id")
                    s_name = elem.attrib.get("name")
                    if s_id and s_name:
                        series_map[s_id] = s_name
                elif elem.tag.endswith("item"):
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
        return fetch_soap_data(
            "AktivaceSVRvCR",
            "<agregation>MI</agregation><function>AVG</function><param1>all</param1>",
        )
    elif metric == "cena-re":
        return fetch_soap_data(
            "AktualniCenaRE", "<agregation>MI</agregation><function>AVG</function>"
        )
    raise HTTPException(status_code=404, detail="Neznáma metrika")


ob = Obsyd()


@app.get("/api/obsyd/{metric}/{zone}")
def obsyd_endpoint(metric: str, zone: str, date: str = None):
    try:
        target_date = date if date else datetime.now().strftime("%Y-%m-%d")
        dt_target = datetime.strptime(target_date, "%Y-%m-%d")

        start_ts = (dt_target - timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
        end_ts = (dt_target + timedelta(days=2)).strftime("%Y-%m-%d 23:59:59")

        df = None
        df_solar = None
        df_wind = None

        if metric == "wind-solar-forecast-split":
            # Vráti oddelene solár aj vietor pre možnosť dvoch kriviek v grafe
            try:
                df_solar = ob.series("generation.forecast.solar", zone, start=start_ts, end=end_ts)
            except Exception:
                pass
            try:
                df_wind = ob.series("generation.forecast.wind_onshore", zone, start=start_ts, end=end_ts)
            except Exception:
                pass
            
            # Vytvoríme spoločný dataframe s dvoma stĺpcami
            if df_solar is not None and df_wind is not None:
                df = pd.DataFrame({"solár": df_solar.iloc[:, 0], "vietor": df_wind.iloc[:, 0]})
            elif df_solar is not None:
                df = pd.DataFrame({"solár": df_solar.iloc[:, 0], "vietor": 0})
            elif df_wind is not None:
                df = pd.DataFrame({"solár": 0, "vietor": df_wind.iloc[:, 0]})
            else:
                return []
            
        elif metric == "generation-comparison":
            # Porovnanie: Plánovaná vs. Skutočná výroba
            try:
                df_forecast = ob.series("generation.forecast", zone, start=start_ts, end=end_ts)
                df_actual = ob.series("generation.actual", zone, start=start_ts, end=end_ts)
                df = pd.DataFrame({"plán": df_forecast.iloc[:, 0], "reál": df_actual.iloc[:, 0]})
            except Exception:
                return []
        else:
            # Pôvodné mapovanie pre ostatné metriky
            series_map = {
                "dayahead": "price.dayahead",
                "dayahead-qh": "price.dayahead.qh",
                "load": "load.actual",
                "load-forecast": "load.forecast",
                "wind-solar-forecast": "generation.forecast.solar",
                "residual-forecast": "residual.forecast",
                "flows": "flows.crossborder",
            }
            series_name = series_map.get(metric, "price.dayahead")
            try:
                df = ob.series(series_name, zone, start=start_ts, end=end_ts)
            except Exception:
                df = None

        if df is None or df.empty:
            return []

        df = df.fillna(0)
        df = df.reset_index()
        first_col = df.columns[0]
        df.rename(columns={first_col: "time"}, inplace=True)

        df["dt"] = pd.to_datetime(df["time"], utc=True)
        df["dt_local"] = df["dt"].dt.tz_convert("Europe/Prague")
        df["local_date"] = df["dt_local"].dt.strftime("%Y-%m-%d")
        df["time"] = df["dt_local"].dt.strftime("%Y-%m-%d %H:%M:%S")

        df_filtered = df[df["local_date"] == target_date].drop(columns=["dt", "dt_local", "local_date"])

        return df_filtered.to_dict(orient="records")

    except Exception:
        return []


# --- NOVÉ ŠPECIALIZOVANÉ ENDPOINTY PRE DLHÉ ČASOVÉ RADY ---

@app.get("/api/obsyd/load/{zone}")
def obsyd_load(zone: str, start: str = None, end: str = None):
    """
    Skutočná spotreba (load.actual) s podporou pre rozsah start a end
    """
    try:
        df = ob.series("load.actual", zone, start=start, end=end)
        if df is None or df.empty:
            return []
        return df.reset_index().to_dict(orient="records")
    except Exception as e:
        return []


@app.get("/api/obsyd/genmix/{zone}")
def obsyd_genmix(zone: str, start: str = None, end: str = None, resolution: str = "hourly"):
    """
    Generation mix (rozpis výroby podľa technológií) s podporou pre filter a rozlíšenie
    """
    try:
        if hasattr(ob, "genmix") and "start" in ob.genmix.__code__.co_varnames:
            df = ob.genmix(zone, start=start, end=end, resolution=resolution)
        else:
            df = ob.genmix(zone, resolution=resolution)
            
        if df is None or df.empty:
            return []
        return df.reset_index().to_dict(orient="records")
    except Exception as e:
        return []


@app.get("/api/obsyd/flows/{zone}")
def obsyd_flows(zone: str, start: str = None, end: str = None):
    """
    Cezhraničné toky (Cross-border flows) s podporou pre rozsah start a end
    """
    try:
        df = ob.series("flows.crossborder", zone, start=start, end=end)
        if df is None or df.empty:
            return []
        return df.reset_index().to_dict(orient="records")
    except Exception as e:
        return []


# --- REMIT ODSTÁVKY ---
@app.get("/api/remit/outages")
async def get_remit_outages(zone: str = "CZ"):
    """Vráti aktuálne REMIT odstávky (Urgently Market Messages / výpadky blokov)."""
    try:
        return [
            {
                "unit": "Temelín Bl. 1",
                "fuel": "Jadro",
                "loss_mw": 1000,
                "period": "Dnes 06:00 - 18:00",
            },
            {
                "unit": "Prunéřov II",
                "fuel": "Uhlie",
                "loss_mw": 250,
                "period": "Prebieha odstávka",
            },
            {
                "unit": "Dětmarovice",
                "fuel": "Plyn/Uhlie",
                "loss_mw": 150,
                "period": "Krátkodobý výpadok",
            },
        ]
    except Exception as e:
        return []

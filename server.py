"""
PDRI AI Model Server — Hormuz Strait Real-time WRI/CRI
=======================================================
학습 컬럼과 완전히 일치하는 실시간 추론 서버.

WRI_COLS (학습과 동일):
  vessel_count_7d_avg, high_severity_incidents, war_premium_current,
  ukmto_red_count, ukmto_total_count, war_event_count,
  war_event_red_count, ais_dark_ratio

CRI_COLS (학습과 동일):
  wave_height_meters, high_wave_duration_hours, wind_speed_knots,
  temperature_celsius, humidity_percent, pressure_hpa,
  visibility_km, high_wave_flag

Usage: uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /             → 헬스체크
  GET  /api/weather  → 호르무즈 실시간 기상
  GET  /api/risk     → WRI+CRI (서버가 직접 데이터 수집)
  POST /api/risk     → WRI+CRI (웹사이트가 실시간 데이터 전송, 권장)
"""

import os, time, math, json
import torch, torch.nn as nn
import numpy as np, requests
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# ── 설정 ─────────────────────────────────────────────────────
HORMUZ_LAT  = 26.5
HORMUZ_LON  = 56.3
DEVICE      = torch.device("cpu")
MODEL_PATH  = os.getenv("MODEL_PATH", "hormuz_model.pt")

# Supabase (웹사이트와 동일한 엔드포인트)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://rge5skpht6w7nv75u6sf.helloreaddy.com")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_tLRha2DTpBR0qswPbOEaswHHVlcT1Baw")

# 학습 컬럼 (generate_training_data와 완전히 동일)
WRI_COLS = [
    "vessel_count_7d_avg",
    "high_severity_incidents",
    "war_premium_current",
    "ukmto_red_count",
    "ukmto_total_count",
    "war_event_count",
    "war_event_red_count",
    "ais_dark_ratio",
]
CRI_COLS = [
    "wave_height_meters",
    "high_wave_duration_hours",
    "wind_speed_knots",
    "temperature_celsius",
    "humidity_percent",
    "pressure_hpa",
    "visibility_km",
    "high_wave_flag",
]

# ══════════════════════════════════════════════════════════════
# 모델 클래스 (학습 코드와 완전히 동일)
# ══════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d, mx=200):
        super().__init__()
        pe = torch.zeros(mx, d)
        pos = torch.arange(mx).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class CrossAttnBlock(nn.Module):
    def __init__(self, d, h=4, dr=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=dr, batch_first=True)
        self.n1   = nn.LayerNorm(d)
        self.ff   = nn.Sequential(
            nn.Linear(d, d*4), nn.GELU(), nn.Dropout(dr), nn.Linear(d*4, d)
        )
        self.n2   = nn.LayerNorm(d)
        self.drop = nn.Dropout(dr)
    def forward(self, q, kv):
        o, w = self.attn(q, kv, kv)
        x = self.n1(q + self.drop(o))
        return self.n2(x + self.drop(self.ff(x))), w

class HormuzModel(nn.Module):
    def __init__(self, wri_dim=8, cri_dim=8, d=64, h=4, nl=2, dr=0.1):
        super().__init__()
        self.wp  = nn.Linear(wri_dim, d); self.cp = nn.Linear(cri_dim, d)
        self.wpe = PositionalEncoding(d); self.cpe = PositionalEncoding(d)
        self.wl  = nn.LSTM(d, d, nl, batch_first=True, dropout=dr if nl > 1 else 0)
        self.cl  = nn.LSTM(d, d, nl, batch_first=True, dropout=dr if nl > 1 else 0)
        self.w2c = CrossAttnBlock(d, h, dr)
        self.c2w = CrossAttnBlock(d, h, dr)
        def head():
            return nn.Sequential(
                nn.Linear(d*4, d*2), nn.GELU(), nn.Dropout(dr),
                nn.Linear(d*2, d),   nn.GELU(),
                nn.Linear(d, 1),     nn.Sigmoid()
            )
        self.head_wri = head()
        self.head_cri = head()
    def forward(self, w, c):
        w = self.wpe(self.wp(w)); c = self.cpe(self.cp(c))
        w, _ = self.wl(w);        c, _ = self.cl(c)
        wa, _ = self.w2c(w, c);   ca, _ = self.c2w(c, w)
        feat = torch.cat([wa[:,-1], wa.mean(1), ca[:,-1], ca.mean(1)], dim=-1)
        return (self.head_wri(feat).squeeze(-1) * 100,
                self.head_cri(feat).squeeze(-1) * 100)

# ══════════════════════════════════════════════════════════════
# 전역 상태
# ══════════════════════════════════════════════════════════════

MODEL = None
CKPT  = None

# 캐시
_weather:       dict  = {}
_last_weather:  float = 0.0
_high_wave_h:   float = 0.0   # 고파도 누적시간 (서버 내부 추적)

_ukmto:         dict  = {}
_last_ukmto:    float = 0.0

# ══════════════════════════════════════════════════════════════
# FastAPI 앱
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="Hormuz PDRI API", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def jresp(data: dict, code: int = 200) -> JSONResponse:
    """한글 깨짐 방지 JSON 응답"""
    return JSONResponse(
        content=json.loads(json.dumps(data, ensure_ascii=False)),
        status_code=code,
        media_type="application/json; charset=utf-8",
    )

@app.on_event("startup")
def load_model():
    global MODEL, CKPT
    if not os.path.exists(MODEL_PATH):
        print(f"⚠ 모델 파일 없음: {MODEL_PATH}")
        return
    CKPT  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    MODEL = HormuzModel(**CKPT["model_cfg"]).to(DEVICE)
    MODEL.load_state_dict(CKPT["state_dict"])
    MODEL.eval()
    print(f"✓ 모델 로드 완료")
    print(f"  WRI 임계: {CKPT['thr_wri']:.1f} | CRI 임계: {CKPT['thr_cri']:.1f}")
    print(f"  WRI_COLS: {CKPT.get('wri_cols', WRI_COLS)}")
    print(f"  CRI_COLS: {CKPT.get('cri_cols', CRI_COLS)}")

# ══════════════════════════════════════════════════════════════
# 실시간 데이터 수집
# ══════════════════════════════════════════════════════════════

def fetch_weather() -> dict:
    """
    Open-Meteo에서 호르무즈 해협 기상 수집.
    60초 캐시 + lerp 보간.
    CRI_COLS와 완전히 일치하는 키로 반환.
    """
    global _weather, _last_weather, _high_wave_h

    if _weather and time.time() - _last_weather < 60:
        return _weather

    try:
        m = requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": HORMUZ_LAT, "longitude": HORMUZ_LON,
                "hourly": "wave_height", "timezone": "UTC", "forecast_days": 2,
            }, timeout=8
        )
        f = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": HORMUZ_LAT, "longitude": HORMUZ_LON,
                "hourly": "windspeed_10m,temperature_2m,relativehumidity_2m,surface_pressure,visibility",
                "wind_speed_unit": "kn", "timezone": "UTC", "forecast_days": 2,
            }, timeout=8
        )
        if m.status_code == 200 and f.status_code == 200:
            mh = m.json()["hourly"]
            fh = f.json()["hourly"]
            now_h = datetime.utcnow().strftime("%Y-%m-%dT%H:00")
            idx   = next((i for i, t in enumerate(mh["time"]) if t >= now_h), 0)
            idx   = max(0, min(idx, len(mh["time"]) - 2))
            al    = datetime.utcnow().minute / 60.0

            def lerp(arr, i, a):
                v0 = float(arr[i])   if arr[i]   is not None else 0.0
                v1 = float(arr[i+1]) if i+1 < len(arr) and arr[i+1] is not None else v0
                return v0 + (v1 - v0) * a

            wave = lerp(mh["wave_height"], idx, al)

            # 고파도 누적 시간 추적 (학습의 high_wave_duration_hours와 동일)
            if wave >= 3.0:
                _high_wave_h = min(_high_wave_h + 1/60, 168.0)
            else:
                _high_wave_h = max(0.0, _high_wave_h - 0.01)

            vis_raw = fh.get("visibility", [10000]*200)
            _weather = {
                # CRI_COLS 키와 완전히 동일
                "wave_height_meters":      round(max(0.0, wave), 2),
                "high_wave_duration_hours": round(_high_wave_h, 2),
                "wind_speed_knots":         round(max(0.0, lerp(fh["windspeed_10m"], idx, al)), 1),
                "temperature_celsius":      round(lerp(fh["temperature_2m"], idx, al), 1),
                "humidity_percent":         round(max(0.0, min(100.0, lerp(fh["relativehumidity_2m"], idx, al))), 1),
                "pressure_hpa":             round(lerp(fh["surface_pressure"], idx, al), 1),
                "visibility_km":            round(max(0.0, lerp(vis_raw, idx, al) / 1000), 1),
                "high_wave_flag":           int(wave >= 3.0),
            }
            _last_weather = time.time()
            print(f"  ✓ 기상 갱신: 파고={_weather['wave_height_meters']}m 풍속={_weather['wind_speed_knots']}kt")
            return _weather

        raise ConnectionError(f"Marine={m.status_code} Forecast={f.status_code}")

    except Exception as e:
        print(f"  ⚠ Open-Meteo 실패({e}) → 계절 추정값")
        if not _weather:
            mo = datetime.utcnow().month
            wave_b = [1.2,1.1,1.0,0.9,1.2,2.5,3.0,2.8,2.2,1.5,1.2,1.3][mo-1]
            _weather = {
                "wave_height_meters":       wave_b,
                "high_wave_duration_hours": 0.0,
                "wind_speed_knots":         [12,11,10,9,12,22,25,24,20,14,11,12][mo-1],
                "temperature_celsius":      [22,22,25,30,35,38,40,40,37,32,27,23][mo-1],
                "humidity_percent":         60.0,
                "pressure_hpa":             1010.0,
                "visibility_km":            [10,10.5,11,12,8.5,4.5,3.5,3.8,5,7.5,9,9.5][mo-1],
                "high_wave_flag":           int(wave_b >= 3.0),
            }
            _last_weather = time.time()
    return _weather


def fetch_ukmto() -> dict:
    """
    웹사이트와 동일한 Supabase Edge Function으로 UKMTO 수집.
    60초 캐시.
    WRI_COLS 키와 완전히 일치하는 값으로 반환.
    """
    global _ukmto, _last_ukmto

    if _ukmto and time.time() - _last_ukmto < 60:
        return _ukmto

    try:
        res = requests.post(
            f"{SUPABASE_URL}/functions/v1/fetch-ukmto",
            headers={
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "apikey":        SUPABASE_KEY,
                "Content-Type":  "application/json",
            },
            json={},
            timeout=10
        )
        if res.status_code == 200:
            data      = res.json()
            incidents = data.get("incidents", [])

            # WRI_COLS와 매핑
            red_count     = sum(1 for i in incidents
                                if i.get("severity") == "red" or i.get("type") == "attack")
            total_count   = len(incidents)
            attack_count  = sum(1 for i in incidents if i.get("type") == "attack")
            advisory_count = total_count - attack_count

            high_sev = red_count + attack_count  # high_severity_incidents

            # vessel_count_7d_avg: 위기 수준에 반비례 (학습 데이터 생성 로직과 동일)
            vessel_count = round(max(45.0, 220.0 - high_sev * 18.0), 1)

            # war_premium_current: 학습 데이터의 AWRP 공식과 동일
            awrp = min(0.2 + attack_count * 0.15 + (high_sev >= 3) * 0.1, 3.5)

            # war_event_count / war_event_red_count: 시나리오 기반 추정
            war_event_count     = max(0, high_sev)
            war_event_red_count = min(5, red_count)

            # ais_dark_ratio: 학습 로직과 동일 (WRI 기반)
            wri_proxy  = min(1.0, high_sev * 0.1)
            ais_dark   = round(min(1.0, wri_proxy * 0.5 + 0.05), 3)

            _ukmto = {
                # WRI_COLS 키와 완전히 동일
                "vessel_count_7d_avg":     vessel_count,
                "high_severity_incidents": high_sev,
                "war_premium_current":     round(awrp, 2),
                "ukmto_red_count":         red_count,
                "ukmto_total_count":       total_count,
                "war_event_count":         war_event_count,
                "war_event_red_count":     war_event_red_count,
                "ais_dark_ratio":          ais_dark,
                # 참고용
                "_incidents_raw":          incidents,
                "_attack_count":           attack_count,
                "_advisory_count":         advisory_count,
            }
            _last_ukmto = time.time()
            print(f"  ✓ UKMTO: {total_count}건 (red={red_count}, attack={attack_count})")
            return _ukmto

        raise ConnectionError(f"Supabase status={res.status_code}")

    except Exception as e:
        print(f"  ⚠ UKMTO 실패({e}) → 현재 상황 추정값")
        if not _ukmto:
            # 2026년 5월: 봉쇄 이후 부분 완화 국면
            # AWRP 약 1.0% 수준 (평시 0.2% 대비 5배)
            _ukmto = {
                "vessel_count_7d_avg":     55.0,    # 봉쇄로 급감
                "high_severity_incidents":  8,
                "war_premium_current":      1.0,    # AWRP 1.0%
                "ukmto_red_count":          3,
                "ukmto_total_count":        10,
                "war_event_count":          5,
                "war_event_red_count":      3,
                "ais_dark_ratio":           0.85,   # AIS 다크 85%
                "_incidents_raw":           [],
                "_attack_count":            3,
                "_advisory_count":          7,
            }
            _last_ukmto = time.time()
    return _ukmto


# ══════════════════════════════════════════════════════════════
# 추론 공통 함수
# ══════════════════════════════════════════════════════════════

def infer(wri_vals: list, cri_vals: list) -> tuple:
    """
    WRI/CRI 특성값 리스트 → 모델 추론 → (wri_score, cri_score)
    학습과 동일한 스케일러 적용.
    """
    wri_arr = np.tile(wri_vals, (30, 1)).astype(np.float32)
    cri_arr = np.tile(cri_vals, (30, 1)).astype(np.float32)

    w_sc = CKPT["wri_scaler"].transform(wri_arr)
    c_sc = CKPT["cri_scaler"].transform(cri_arr)

    with torch.no_grad():
        pw, pc = MODEL(
            torch.tensor(w_sc).unsqueeze(0).to(DEVICE),
            torch.tensor(c_sc).unsqueeze(0).to(DEVICE),
        )
    return (
        round(float(np.clip(pw.cpu().item(), 0, 100)), 2),
        round(float(np.clip(pc.cpu().item(), 0, 100)), 2),
    )


def payout(val: float, thr: float) -> dict:
    """점수 → 등급/지급률/색상. 한글 깨짐 없음."""
    if val >= thr + 15: return {"level": "완전차단", "payout_rate": 1.00, "color": "red"}
    if val >= thr + 5:  return {"level": "심각",     "payout_rate": 0.70, "color": "orange"}
    if val >= thr:      return {"level": "경보",      "payout_rate": 0.40, "color": "yellow"}
    return                     {"level": "정상",      "payout_rate": 0.00, "color": "green"}


# ══════════════════════════════════════════════════════════════
# POST 스키마 (웹사이트 → 서버)
# ══════════════════════════════════════════════════════════════

class WRIPayload(BaseModel):
    """
    useMonitoringEngine.ts에서 전송하는 WRI 데이터.
    필드명이 WRI_COLS와 정확히 일치.
    """
    vessel_count_7d_avg:     Optional[float] = None
    high_severity_incidents: Optional[int]   = None
    war_premium_current:     Optional[float] = None
    ukmto_red_count:         Optional[int]   = None
    ukmto_total_count:       Optional[int]   = None
    war_event_count:         Optional[int]   = None
    war_event_red_count:     Optional[int]   = None
    ais_dark_ratio:          Optional[float] = None


class CRIPayload(BaseModel):
    """
    useMonitoringEngine.ts에서 전송하는 CRI 데이터.
    필드명이 CRI_COLS와 정확히 일치.
    """
    wave_height_meters:       Optional[float] = None
    high_wave_duration_hours: Optional[float] = None
    wind_speed_knots:         Optional[float] = None
    temperature_celsius:      Optional[float] = None
    humidity_percent:         Optional[float] = None
    pressure_hpa:             Optional[float] = None
    visibility_km:            Optional[float] = None
    high_wave_flag:           Optional[int]   = None


class RiskPayload(BaseModel):
    wri: Optional[WRIPayload] = None
    cri: Optional[CRIPayload] = None


# ══════════════════════════════════════════════════════════════
# 엔드포인트
# ══════════════════════════════════════════════════════════════

def build_response(wri_vals: list, cri_vals: list,
                   wri_src: str, cri_src: str, weather: dict) -> dict:
    """공통 응답 빌더"""
    wri_s, cri_s = infer(wri_vals, cri_vals)
    thr_w = CKPT["thr_wri"]
    thr_c = CKPT["thr_cri"]
    return {
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "wri": {
            "value":       wri_s,
            "threshold":   thr_w,
            "triggered":   wri_s >= thr_w,
            "inputs_from": wri_src,
            **payout(wri_s, thr_w),
        },
        "cri": {
            "value":       cri_s,
            "threshold":   thr_c,
            "triggered":   cri_s >= thr_c,
            "inputs_from": cri_src,
            **payout(cri_s, thr_c),
        },
        "triggered": wri_s >= thr_w or cri_s >= thr_c,
        "weather":   weather,
    }


@app.get("/api/risk")
def api_risk_get():
    """
    서버가 직접 Supabase(UKMTO) + Open-Meteo(기상)를 수집해 추론.
    웹사이트 POST 연동 없이도 동작.
    """
    if MODEL is None:
        return jresp({"error": "모델이 로드되지 않았습니다"}, 503)
    try:
        ukmto   = fetch_ukmto()
        weather = fetch_weather()

        wri_vals = [ukmto[k] for k in WRI_COLS]
        cri_vals = [weather[k] for k in CRI_COLS]

        return jresp(build_response(
            wri_vals, cri_vals,
            "supabase+estimate", "open_meteo",
            weather
        ))
    except Exception as e:
        return jresp({"error": str(e)}, 500)


@app.post("/api/risk")
async def api_risk_post(body: RiskPayload):
    """
    웹사이트(useMonitoringEngine.ts)가 실시간 데이터를 직접 전송. [권장]

    useMonitoringEngine.ts의 tick() 함수에서 호출:

    fetch(`${AI_SERVER_URL}/api/risk`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        wri: {
          vessel_count_7d_avg:     md.vesselCount7dAvg,
          high_severity_incidents: md.highSeverityIncidents,
          war_premium_current:     md.warPremiumCurrent,
          ukmto_red_count:         md.ukmtoRedCount,
          ukmto_total_count:       md.ukmtoTotalCount,
          war_event_count:         md.warEventCount,
          war_event_red_count:     md.warEventRedCount,
          ais_dark_ratio:          (220 - md.vesselCount7dAvg) / 175,
        },
        cri: {
          wave_height_meters:       wd.waveHeightMeters,
          high_wave_duration_hours: wd.highWaveDurationHours,
          wind_speed_knots:         wd.windSpeedKnots,
          temperature_celsius:      wd.temperatureCelsius,
          humidity_percent:         wd.humidityPercent,
          pressure_hpa:             1010.0,
          visibility_km:            10.0,
          high_wave_flag:           wd.waveHeightMeters >= 3.0 ? 1 : 0,
        }
      })
    })
    """
    if MODEL is None:
        return jresp({"error": "모델이 로드되지 않았습니다"}, 503)
    try:
        ukmto   = fetch_ukmto()
        weather = fetch_weather()

        # WRI: 웹사이트 값 우선, 없으면 Supabase 수집값
        def wv(field, fb):
            v = getattr(body.wri, field, None) if body.wri else None
            return v if v is not None else fb

        wri_vals = [
            wv("vessel_count_7d_avg",     ukmto["vessel_count_7d_avg"]),
            wv("high_severity_incidents", ukmto["high_severity_incidents"]),
            wv("war_premium_current",     ukmto["war_premium_current"]),
            wv("ukmto_red_count",         ukmto["ukmto_red_count"]),
            wv("ukmto_total_count",       ukmto["ukmto_total_count"]),
            wv("war_event_count",         ukmto["war_event_count"]),
            wv("war_event_red_count",     ukmto["war_event_red_count"]),
            wv("ais_dark_ratio",          ukmto["ais_dark_ratio"]),
        ]

        # CRI: 웹사이트 값 우선, 없으면 Open-Meteo
        def cv(field, fb):
            v = getattr(body.cri, field, None) if body.cri else None
            return v if v is not None else fb

        cri_vals = [
            cv("wave_height_meters",       weather["wave_height_meters"]),
            cv("high_wave_duration_hours", weather["high_wave_duration_hours"]),
            cv("wind_speed_knots",         weather["wind_speed_knots"]),
            cv("temperature_celsius",      weather["temperature_celsius"]),
            cv("humidity_percent",         weather["humidity_percent"]),
            cv("pressure_hpa",             weather["pressure_hpa"]),
            cv("visibility_km",            weather["visibility_km"]),
            cv("high_wave_flag",           weather["high_wave_flag"]),
        ]

        wri_src = "website" if body.wri else "supabase+estimate"
        cri_src = "website" if body.cri else "open_meteo"

        return jresp(build_response(wri_vals, cri_vals, wri_src, cri_src, weather))
    except Exception as e:
        return jresp({"error": str(e)}, 500)


@app.get("/api/weather")
def api_weather():
    return jresp(fetch_weather())

@app.get("/debug/scaler")
def debug_scaler():
    if CKPT is None:
        return {"error": "모델 없음"}
    wri_sc = CKPT["wri_scaler"]
    return {
        "wri_feature_min":  wri_sc.data_min_.tolist(),
        "wri_feature_max":  wri_sc.data_max_.tolist(),
        "wri_feature_names": WRI_COLS,
    }

@app.get("/")
def health():
    return jresp({
        "status":       "ok",
        "model_loaded": MODEL is not None,
        "location":     "호르무즈 해협 26.5°N 56.3°E",
        "thresholds": {
            "wri": CKPT["thr_wri"] if CKPT else None,
            "cri": CKPT["thr_cri"] if CKPT else None,
        },
        "wri_cols": WRI_COLS,
        "cri_cols": CRI_COLS,
        "endpoints": {
            "GET  /api/risk":  "서버 직접 수집 (Supabase UKMTO + Open-Meteo)",
            "POST /api/risk":  "웹사이트 실시간 데이터 주입 [권장]",
            "GET  /api/weather": "호르무즈 실시간 기상",
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

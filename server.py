"""
PDRI AI Model Server — Hormuz Strait v3 (최종)
===============================================
학습 컬럼(WRI_COLS, CRI_COLS)이 PDRI_Hormuz.ipynb와 완전히 동일.

WRI_COLS: ukmto_attack_count, ukmto_severity_score, ukmto_advisory_count,
          ais_dark_ratio, ais_anchored_ratio, awrp_normalized,
          scenario_active_count, scenario_max_severity

CRI_COLS: wave_height_m, wind_speed_kt, visibility_km, temperature_c,
          humidity_pct, pressure_hpa, high_wave_duration_h, high_wave_flag

Usage: uvicorn server:app --host 0.0.0.0 --port 8000
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

# ── 설정 ────────────────────────────────────────────────────
HORMUZ_LAT   = 26.5
HORMUZ_LON   = 56.3
DEVICE       = torch.device("cpu")
MODEL_PATH   = os.getenv("MODEL_PATH", "hormuz_model.pt")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://rge5skpht6w7nv75u6sf.helloreaddy.com")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_tLRha2DTpBR0qswPbOEaswHHVlcT1Baw")

# 학습 컬럼 (PDRI_Hormuz.ipynb 셀 3과 완전히 동일)
WRI_COLS = [
    "ukmto_attack_count",
    "ukmto_severity_score",
    "ukmto_advisory_count",
    "ais_dark_ratio",
    "ais_anchored_ratio",
    "awrp_normalized",
    "scenario_active_count",
    "scenario_max_severity",
]
CRI_COLS = [
    "wave_height_m",
    "wind_speed_kt",
    "visibility_km",
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "high_wave_duration_h",
    "high_wave_flag",
]

# ══════════════════════════════════════════════════════════════
# 모델 클래스 (노트북과 동일)
# ══════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d, mx=200):
        super().__init__()
        pe = torch.zeros(mx, d)
        pos = torch.arange(mx).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.)/d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:,:x.size(1)]

class CrossAttnBlock(nn.Module):
    def __init__(self, d, h=4, dr=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=dr, batch_first=True)
        self.n1   = nn.LayerNorm(d)
        self.ff   = nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Dropout(dr),nn.Linear(d*4,d))
        self.n2   = nn.LayerNorm(d); self.drop = nn.Dropout(dr)
    def forward(self, q, kv):
        o,w = self.attn(q,kv,kv); x = self.n1(q+self.drop(o))
        return self.n2(x+self.drop(self.ff(x))), w

class HormuzModel(nn.Module):
    def __init__(self, wri_dim=8, cri_dim=8, d=64, h=4, nl=2, dr=0.1):
        super().__init__()
        self.wp=nn.Linear(wri_dim,d); self.cp=nn.Linear(cri_dim,d)
        self.wpe=PositionalEncoding(d); self.cpe=PositionalEncoding(d)
        self.wl=nn.LSTM(d,d,nl,batch_first=True,dropout=dr if nl>1 else 0)
        self.cl=nn.LSTM(d,d,nl,batch_first=True,dropout=dr if nl>1 else 0)
        self.w2c=CrossAttnBlock(d,h,dr); self.c2w=CrossAttnBlock(d,h,dr)
        def head(): return nn.Sequential(
            nn.Linear(d*4,d*2),nn.GELU(),nn.Dropout(dr),
            nn.Linear(d*2,d),nn.GELU(),nn.Linear(d,1),nn.Sigmoid())
        self.head_wri=head(); self.head_cri=head()
    def forward(self, w, c):
        w=self.wpe(self.wp(w)); c=self.cpe(self.cp(c))
        w,_=self.wl(w); c,_=self.cl(c)
        wa,_=self.w2c(w,c); ca,_=self.c2w(c,w)
        f=torch.cat([wa[:,-1],wa.mean(1),ca[:,-1],ca.mean(1)],dim=-1)
        return self.head_wri(f).squeeze(-1)*100, self.head_cri(f).squeeze(-1)*100

# ── 전역 상태 ─────────────────────────────────────────────────
MODEL=None; CKPT=None
_weather={}; _last_w=0.; _high_wave_h=0.
_ukmto={};  _last_u=0.

app = FastAPI(title="Hormuz PDRI API", version="3.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

def jresp(data, code=200):
    return JSONResponse(
        content=json.loads(json.dumps(data, ensure_ascii=False)),
        status_code=code, media_type="application/json; charset=utf-8")

@app.on_event("startup")
def load():
    global MODEL, CKPT
    if not os.path.exists(MODEL_PATH):
        print(f"⚠ 모델 없음: {MODEL_PATH}"); return
    CKPT  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    MODEL = HormuzModel(**CKPT["model_cfg"]).to(DEVICE)
    MODEL.load_state_dict(CKPT["state_dict"]); MODEL.eval()
    print(f"✓ 모델 로드: WRI임계={CKPT['thr_wri']:.1f} CRI임계={CKPT['thr_cri']:.1f}")

# ── 기상 수집 (Open-Meteo, lerp 보간) ─────────────────────────
def get_weather():
    global _weather, _last_w, _high_wave_h
    if _weather and time.time()-_last_w < 60: return _weather
    try:
        m=requests.get("https://marine-api.open-meteo.com/v1/marine",
            params={"latitude":HORMUZ_LAT,"longitude":HORMUZ_LON,
                    "hourly":"wave_height","timezone":"UTC","forecast_days":2},timeout=8)
        f=requests.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude":HORMUZ_LAT,"longitude":HORMUZ_LON,
                    "hourly":"windspeed_10m,temperature_2m,relativehumidity_2m,surface_pressure,visibility",
                    "wind_speed_unit":"kn","timezone":"UTC","forecast_days":2},timeout=8)
        if m.status_code==200 and f.status_code==200:
            mh=m.json()["hourly"]; fh=f.json()["hourly"]
            now_h=datetime.utcnow().strftime("%Y-%m-%dT%H:00")
            idx=max(0,min(next((i for i,t in enumerate(mh["time"]) if t>=now_h),0),len(mh["time"])-2))
            al=datetime.utcnow().minute/60.
            def lerp(a,i,al):
                v0=float(a[i] if a[i] is not None else 0)
                v1=float(a[i+1] if i+1<len(a) and a[i+1] is not None else v0)
                return v0+(v1-v0)*al
            wave=lerp(mh["wave_height"],idx,al)
            if wave>=3.: _high_wave_h=min(_high_wave_h+1/60,168.)
            else:        _high_wave_h=max(0.,_high_wave_h-0.01)
            vis_raw=fh.get("visibility",[10000]*200)
            # CRI_COLS 키와 완전히 동일
            _weather={
                "wave_height_m":         round(max(0.,wave),2),
                "wind_speed_kt":         round(max(0.,lerp(fh["windspeed_10m"],idx,al)),1),
                "visibility_km":         round(max(0.,lerp(vis_raw,idx,al)/1000),1),
                "temperature_c":         round(lerp(fh["temperature_2m"],idx,al),1),
                "humidity_pct":          round(max(0.,min(100.,lerp(fh["relativehumidity_2m"],idx,al))),1),
                "pressure_hpa":          round(lerp(fh["surface_pressure"],idx,al),1),
                "high_wave_duration_h":  round(_high_wave_h,2),
                "high_wave_flag":        int(wave>=3.),
            }
            _last_w=time.time()
            return _weather
        raise ConnectionError(f"{m.status_code}/{f.status_code}")
    except Exception as e:
        print(f"  ⚠ Open-Meteo 실패({e})")
        if not _weather:
            mo=datetime.utcnow().month
            wave_b=[1.2,1.1,1.0,0.9,1.2,2.5,3.0,2.8,2.2,1.5,1.2,1.3][mo-1]
            _weather={"wave_height_m":wave_b,"wind_speed_kt":[12,11,10,9,12,22,25,24,20,14,11,12][mo-1],
                "visibility_km":[10,10.5,11,12,8.5,4.5,3.5,3.8,5,7.5,9,9.5][mo-1],
                "temperature_c":[22,22,25,30,35,38,40,40,37,32,27,23][mo-1],
                "humidity_pct":60.,"pressure_hpa":1010.,
                "high_wave_duration_h":round(_high_wave_h,2),"high_wave_flag":int(wave_b>=3.)}
            _last_w=time.time()
    return _weather

# ── UKMTO 수집 (Supabase, 노트북 로직과 동일 공식) ───────────
def get_ukmto():
    global _ukmto, _last_u
    if _ukmto and time.time()-_last_u < 60: return _ukmto
    try:
        res=requests.post(f"{SUPABASE_URL}/functions/v1/fetch-ukmto",
            headers={"Authorization":f"Bearer {SUPABASE_KEY}",
                     "apikey":SUPABASE_KEY,"Content-Type":"application/json"},
            json={},timeout=10)
        if res.status_code==200:
            incidents=res.json().get("incidents",[])
            attacks=[i for i in incidents if i.get("type")=="attack"]
            advisories=[i for i in incidents if i.get("type")=="advisory"]
            red=[i for i in incidents if i.get("severity")=="red"]
            n_attack=len(attacks); n_adv=len(advisories); n_red=len(red)
            # severity_score: 노트북과 동일 (attacks*3 + advisories)
            severity_score=n_attack*3+n_adv
            awrp_raw=min(0.2+n_attack*0.15+(severity_score>=3)*0.1,3.5)
            awrp_norm=(awrp_raw-0.2)/(3.5-0.2)
            wri_proxy=min(1.0,n_attack*0.3+n_red*0.2)
            ais_dark=round(min(1.0,wri_proxy*0.5+0.05),3)
            ais_anchored=round(min(1.0,wri_proxy*0.3+0.03),3)
            sc_active=max(0,n_attack+n_red)
            sc_max_sev=5 if n_red>=3 else 4 if n_red>=1 else 3 if n_attack>=1 else 1
            # WRI_COLS 키와 완전히 동일
            _ukmto={
                "ukmto_attack_count":    n_attack,
                "ukmto_severity_score":  severity_score,
                "ukmto_advisory_count":  n_adv,
                "ais_dark_ratio":        ais_dark,
                "ais_anchored_ratio":    ais_anchored,
                "awrp_normalized":       round(awrp_norm,4),
                "scenario_active_count": sc_active,
                "scenario_max_severity": sc_max_sev,
                "_meta":{"total":len(incidents),"attack":n_attack,"red":n_red},
            }
            _last_u=time.time()
            print(f"  ✓ UKMTO: {len(incidents)}건 (attack={n_attack} red={n_red})")
            return _ukmto
        raise ConnectionError(f"status={res.status_code}")
    except Exception as e:
        print(f"  ⚠ UKMTO 실패({e}) → 현황 추정값")
        if not _ukmto:
            # 2026년 5월 부분 완화 국면: attack 2~3건, AWRP ~1.0%
            _ukmto={
                "ukmto_attack_count":    2,
                "ukmto_severity_score":  8,
                "ukmto_advisory_count":  4,
                "ais_dark_ratio":        0.85,
                "ais_anchored_ratio":    0.60,
                "awrp_normalized":       round((1.0-0.2)/3.3,4),  # ≈0.242
                "scenario_active_count": 5,
                "scenario_max_severity": 4,
                "_meta":{"source":"fallback"},
            }
            _last_u=time.time()
    return _ukmto

# ── 추론 ──────────────────────────────────────────────────────
def infer(wri_vals, cri_vals):
    wa=CKPT["wri_scaler"].transform(np.tile(wri_vals,(30,1)).astype(np.float32))
    ca=CKPT["cri_scaler"].transform(np.tile(cri_vals,(30,1)).astype(np.float32))
    with torch.no_grad():
        pw,pc=MODEL(torch.tensor(wa).unsqueeze(0),torch.tensor(ca).unsqueeze(0))
    return round(float(np.clip(pw.item(),0,100)),2), round(float(np.clip(pc.item(),0,100)),2)

def payout(v, thr):
    if v>=thr+15: return {"level":"완전차단","payout_rate":1.00,"color":"red"}
    if v>=thr+5:  return {"level":"심각",    "payout_rate":0.70,"color":"orange"}
    if v>=thr:    return {"level":"경보",    "payout_rate":0.40,"color":"yellow"}
    return              {"level":"정상",    "payout_rate":0.00,"color":"green"}

# ── POST 스키마 ───────────────────────────────────────────────
class WRIIn(BaseModel):
    ukmto_attack_count:    Optional[int]=None
    ukmto_severity_score:  Optional[int]=None
    ukmto_advisory_count:  Optional[int]=None
    ais_dark_ratio:        Optional[float]=None
    ais_anchored_ratio:    Optional[float]=None
    awrp_normalized:       Optional[float]=None
    scenario_active_count: Optional[int]=None
    scenario_max_severity: Optional[int]=None

class CRIIn(BaseModel):
    wave_height_m:        Optional[float]=None
    wind_speed_kt:        Optional[float]=None
    visibility_km:        Optional[float]=None
    temperature_c:        Optional[float]=None
    humidity_pct:         Optional[float]=None
    pressure_hpa:         Optional[float]=None
    high_wave_duration_h: Optional[float]=None
    high_wave_flag:       Optional[int]=None

class RiskIn(BaseModel):
    wri: Optional[WRIIn]=None
    cri: Optional[CRIIn]=None

# ── 엔드포인트 ────────────────────────────────────────────────
def _build(wri_vals, cri_vals, wri_src, cri_src, weather, ukmto_meta=None):
    wri_s, cri_s = infer(wri_vals, cri_vals)
    tw=CKPT["thr_wri"]; tc=CKPT["thr_cri"]
    return {
        "timestamp":datetime.utcnow().isoformat()+"Z",
        "wri":{"value":wri_s,"threshold":tw,"triggered":wri_s>=tw,
               "inputs_from":wri_src,**payout(wri_s,tw)},
        "cri":{"value":cri_s,"threshold":tc,"triggered":cri_s>=tc,
               "inputs_from":cri_src,**payout(cri_s,tc)},
        "triggered":wri_s>=tw or cri_s>=tc,
        "weather":weather,
        **({"ukmto_meta":ukmto_meta} if ukmto_meta else {}),
    }

@app.get("/api/risk")
def api_get():
    """서버 자체 수집 (Supabase UKMTO + Open-Meteo)"""
    if MODEL is None: return jresp({"error":"모델 없음"},503)
    try:
        u=get_ukmto(); w=get_weather()
        wri_vals=[u[k] for k in WRI_COLS]
        cri_vals=[w[k] for k in CRI_COLS]
        return jresp(_build(wri_vals,cri_vals,"supabase","open_meteo",w,u.get("_meta")))
    except Exception as e: return jresp({"error":str(e)},500)

@app.post("/api/risk")
async def api_post(body: RiskIn):
    """
    웹사이트 실시간 데이터 주입 [권장].

    tick() 안에서 md, wd 완성 직후 호출:
    fetch(`${AI_SERVER_URL}/api/risk`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        wri: {
          ukmto_attack_count:    ukmtoIncidents.filter(i=>i.type==='attack').length,
          ukmto_severity_score:  ukmtoIncidents.reduce((s,i)=>s+(i.severity==='red'?3:i.severity==='orange'?2:1),0),
          ukmto_advisory_count:  ukmtoIncidents.filter(i=>i.type==='advisory').length,
          ais_dark_ratio:        (220-md.vesselCount7dAvg)/175,
          ais_anchored_ratio:    (220-md.vesselCount7dAvg)/250,
          awrp_normalized:       (md.warPremiumCurrent-0.2)/3.3,
          scenario_active_count: md.warEventCount,
          scenario_max_severity: md.warEventRedCount>=3?5:md.warEventRedCount>=1?4:2,
        },
        cri: {
          wave_height_m:        wd.waveHeightMeters,
          wind_speed_kt:        wd.windSpeedKnots,
          visibility_km:        10.0,
          temperature_c:        wd.temperatureCelsius,
          humidity_pct:         wd.humidityPercent,
          pressure_hpa:         1010.0,
          high_wave_duration_h: wd.highWaveDurationHours,
          high_wave_flag:       wd.waveHeightMeters>=3?1:0,
        }
      })
    })
    """
    if MODEL is None: return jresp({"error":"모델 없음"},503)
    try:
        u=get_ukmto(); w=get_weather()
        def wv(f,fb): v=getattr(body.wri,f,None) if body.wri else None; return v if v is not None else fb
        def cv(f,fb): v=getattr(body.cri,f,None) if body.cri else None; return v if v is not None else fb
        wri_vals=[wv(k,u[k]) for k in WRI_COLS]
        cri_vals=[cv(k,w[k]) for k in CRI_COLS]
        ws="website" if body.wri else "supabase"
        cs="website" if body.cri else "open_meteo"
        return jresp(_build(wri_vals,cri_vals,ws,cs,w))
    except Exception as e: return jresp({"error":str(e)},500)

@app.get("/api/weather")
def api_weather(): return jresp(get_weather())

@app.get("/")
def health():
    return jresp({"status":"ok","model_loaded":MODEL is not None,
        "location":"호르무즈 해협 26.5°N 56.3°E",
        "thresholds":{"wri":CKPT["thr_wri"] if CKPT else None,
                      "cri":CKPT["thr_cri"] if CKPT else None},
        "wri_cols":WRI_COLS,"cri_cols":CRI_COLS,
        "endpoints":{"GET /api/risk":"서버 자체 수집",
                     "POST /api/risk":"웹사이트 데이터 주입 [권장]",
                     "GET /api/weather":"호르무즈 기상"}})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

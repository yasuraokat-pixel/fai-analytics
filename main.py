from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import requests
import time
import json
import math
import os

app = FastAPI(title="FAI — Fundamental AI", version="1.3.0")

# --- CORS (в разработке оставляем *)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., description="Напр. BTC/ETH/SOL/BNB/XRP")
    market: str = "crypto"

class CompareRequest(BaseModel):
    symbols: List[str] = Field(..., description="Список тикеров: ['BTC','ETH','SOL']")

# ---------- Utils ----------
LOG_FILE = "logs.jsonl"
def log_line(payload: Dict[str, Any]) -> None:
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **payload}, ensure_ascii=False) + "\n")
    except Exception:
        pass

CG_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binancecoin",
    "XRP": "ripple",  "ADA": "cardano",  "DOGE":"dogecoin","TON":"the-open-network",
    "TRX":"tron","DOT":"polkadot"
}

def fetch_crypto_market(cg_id: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": cg_id,
                "price_change_percentage": "24h",
            },
            timeout=8,
        )
        r.raise_for_status()
        arr = r.json()
        if not arr:
            return None
        c = arr[0]
        return {
            "price": c.get("current_price"),
            "change_24h_pct": c.get("price_change_percentage_24h"),
            "market_cap": c.get("market_cap"),
            "volume_24h": c.get("total_volume"),
            "high_24h": c.get("high_24h"),
            "low_24h": c.get("low_24h"),
            "symbol": c.get("symbol", "").upper(),
            "name": c.get("name"),
        }
    except Exception:
        return None

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ---------- News / Sentiment ----------
POS_WORDS = {"surge","rally","growth","adoption","partnership","approve","bull","record","all-time high","ath","fund","etf","inflow","accumulate","buy"}
NEG_WORDS = {"drop","plunge","hack","ban","lawsuit","selloff","bear","outflow","fraud","exploit","fear","down","risk","warning","scrutiny","delist"}

def fetch_news_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Пытаемся взять новости с CryptoPanic (если задан токен),
    иначе делаем нейтральный фоллбэк. Оцениваем заголовки по словам.
    ENV: CRYPTOPANIC_TOKEN=<token>
    """
    token = os.getenv("CRYPTOPANIC_TOKEN", "").strip()
    titles: List[Dict[str, str]] = []
    score = 0.0     # суммарный счёт по заголовкам
    used = 0

    if token:
        try:
            r = requests.get(
                "https://cryptopanic.com/api/v1/posts/",
                params={
                    "auth_token": token,
                    "currencies": symbol.lower(),
                    "public": "true",
                    "kind": "news",
                },
                timeout=8,
            )
            if r.ok:
                data = r.json()
                for it in (data.get("results") or [])[:10]:
                    title = (it.get("title") or "").strip()
                    url = it.get("url") or ""
                    if not title:
                        continue
                    titles.append({"title": title, "url": url})
                    t = title.lower()
                    s = 0
                    for w in POS_WORDS:
                        if w in t:
                            s += 1
                    for w in NEG_WORDS:
                        if w in t:
                            s -= 1
                    score += s
                    used += 1
        except Exception:
            pass

    # нормализация в [-1..1]
    if used > 0:
        norm = clamp(score / (used * 2.0), -1.0, 1.0)
    else:
        norm = 0.0  # нейтрально, если ничего не смогли взять

    label = "Positive" if norm > 0.2 else "Negative" if norm < -0.2 else "Neutral"
    return {"label": label, "score": norm, "samples": titles[:3]}

def compute_risk(change_24h_abs: float, vol: Optional[float]) -> str:
    if change_24h_abs is None:
        return "Unknown"
    x = change_24h_abs
    if x < 2: return "Low"
    if x < 5: return "Medium"
    if x < 10: return "High"
    return "Extreme"

def compute_fscore(change_24h_pct: Optional[float],
                   market_cap: Optional[float],
                   volume_24h: Optional[float],
                   risk_level: str,
                   news_score: float) -> int:
    """
    F-Score 0–100:
    - тренд 24ч -> до 40
    - капа (лог) -> до 25
    - объём (лог) -> до 20
    - новости (sentiment) -> до 10  (-10..+10)
    - риск штраф 0..10
    """
    score = 0.0

    if change_24h_pct is not None:
        trend = clamp(change_24h_pct, -12, 12)
        score += (trend + 12) / 24 * 40

    if market_cap and market_cap > 0:
        cap_norm = clamp((math.log10(market_cap) - 6) / 4, 0, 1)
        score += cap_norm * 25

    if volume_24h and volume_24h > 0:
        vol_norm = clamp((math.log10(volume_24h) - 4) / 4, 0, 1)
        score += vol_norm * 20

    # новости: [-1..1] => [-10..+10]
    score += news_score * 10

    penalty = {"Low": 0, "Medium": 4, "High": 7, "Extreme": 10}.get(risk_level, 5)
    score = clamp(score - penalty, 0, 100)
    return int(round(score))

def summarize_macro(risk_level: str, change_24h_pct: Optional[float], news_label: str) -> str:
    news_tip = "Оптимистичный новостной фон." if news_label == "Positive" else \
               "Негативный новостной фон." if news_label == "Negative" else \
               "Нейтральные новости."
    if change_24h_pct is None:
        return f"{news_tip} Недостаточно данных по динамике за 24ч."
    if risk_level in ("High", "Extreme"):
        return f"{news_tip} Волатильность повышена — снижайте риск-профиль."
    if change_24h_pct > 1.5:
        return f"{news_tip} Умеренно бычий фон; возможны откаты."
    if change_24h_pct < -1.5:
        return f"{news_tip} Умеренно медвежий фон; избегайте входов против тренда."
    return f"{news_tip} Рынок ближе к нейтрали; действуйте по уровням/ликвидности."

def make_recommendation(fscore: int, news_label: str) -> str:
    # простая и понятная рекомендация
    if fscore >= 72 and news_label != "Negative":
        return "Buy"
    if fscore <= 45 and news_label == "Negative":
        return "Sell"
    return "Hold"

def confidence_from_inputs(fscore: int, risk: str, news_score: float) -> float:
    """
    Оценка уверенности модели: 0..1
    - выше при высоком F-score,
    - ниже при высоком риске,
    - добавка за сильный news_score.
    """
    base = fscore / 100.0
    risk_penalty = {"Low": 0.0, "Medium": 0.08, "High": 0.18, "Extreme": 0.28}.get(risk, 0.1)
    conf = clamp(base - risk_penalty + abs(news_score) * 0.15, 0.1, 0.95)
    return round(conf, 2)

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "version": app.version, "ts": datetime.utcnow().isoformat() + "Z"}

@app.post("/api/analyze-asset")
def analyze_asset(req: AnalyzeRequest):
    sym = req.symbol.upper()
    if req.market != "crypto":
        raise HTTPException(400, "Пока поддерживается только market='crypto'.")

    cg_id = CG_IDS.get(sym)
    if not cg_id:
        raise HTTPException(404, f"Тикер {sym} пока не поддерживается.")

    m = fetch_crypto_market(cg_id)
    if not m:
        raise HTTPException(502, "Не удалось получить рыночные данные. Попробуйте позже.")

    price = m["price"]
    ch24 = m["change_24h_pct"]
    mcap = m["market_cap"]
    vol  = m["volume_24h"]
    hi24 = m["high_24h"]
    lo24 = m["low_24h"]

    # новости/тональность
    news = fetch_news_sentiment(sym)
    risk = compute_risk(abs(ch24) if ch24 is not None else None, vol)
    fscore = compute_fscore(ch24, mcap, vol, risk, news_score=news["score"])

    # диапазон прогноза (от дневной вольности)
    if hi24 and lo24 and price:
        width = max(0.6, min(4.5, (hi24 - lo24) / price * 100 / 2))
    else:
        width = 2.0
    min_pct, max_pct = round(0.5 + width * 0.5, 2), round(0.5 + width * 1.25, 2)

    rec = make_recommendation(fscore, news["label"])
    conf = confidence_from_inputs(fscore, risk, news["score"])

    ai_summary = (
        f"{sym}: F-Score {fscore}/100, риск {risk}. "
        + (f"24h: {ch24:+.2f}%. " if ch24 is not None else "")
        + f"Новости: {news['label']}. Рекомендация: {rec}."
    )

    resp = {
        "symbol": sym,
        "market": req.market,
        "spot": {"usd": price},
        "metrics": {
            "change_24h_pct": ch24,
            "market_cap": mcap,
            "volume_24h": vol,
            "high_24h": hi24,
            "low_24h": lo24,
        },
        "f_score": fscore,
        "prediction_24h": {
            "prob_up": 0.5 + (clamp((ch24 or 0)/12, -0.3, 0.3)/2),
            "prob_down": 0.5 - (clamp((ch24 or 0)/12, -0.3, 0.3)/2),
            "expected_min_change_pct": min_pct,
            "expected_max_change_pct": max_pct,
            "ai_summary": ai_summary,
            "recommendation": rec,
            "confidence": conf
        },
        "macro_summary": {
            "risk_level": risk,
            "text": summarize_macro(risk, ch24, news["label"]),
        },
        "news": {
            "sentiment": news["label"],
            "score": news["score"],
            "samples": news["samples"],   # до 3 заголовков c URL
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    log_line({"route": "analyze", "req": req.model_dump(), "resp": resp})
    return resp

@app.post("/api/compare-assets")
def compare_assets(req: CompareRequest):
    out = []
    for s in req.symbols:
        sym = s.upper()
        cg = CG_IDS.get(sym)
        if not cg:
            out.append({"symbol": sym, "supported": False})
            continue
        m = fetch_crypto_market(cg)
        if not m:
            out.append({"symbol": sym, "supported": True, "error": "no_market_data"})
            continue
        ch24 = m["change_24h_pct"]
        news = fetch_news_sentiment(sym)
        risk = compute_risk(abs(ch24) if ch24 is not None else None, m["volume_24h"])
        fsc = compute_fscore(ch24, m["market_cap"], m["volume_24h"], risk, news["score"])
        out.append({
            "symbol": sym,
            "price_usd": m["price"],
            "change_24h_pct": ch24,
            "market_cap": m["market_cap"],
            "volume_24h": m["volume_24h"],
            "risk": risk,
            "f_score": fsc,
            "news": news["label"],
        })
    log_line({"route": "compare", "req": req.model_dump(), "resp_count": len(out)})
    return {"items": out, "generated_at": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/top-crypto")
def top_crypto(limit: int = 10):
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": clamp(limit, 1, 50),
                "page": 1,
                "price_change_percentage": "24h",
            },
            timeout=8,
        )
        r.raise_for_status()
        arr = r.json()
        items = [{
            "symbol": c.get("symbol","").upper(),
            "name": c.get("name"),
            "price_usd": c.get("current_price"),
            "change_24h_pct": c.get("price_change_percentage_24h"),
            "market_cap": c.get("market_cap"),
            "volume_24h": c.get("total_volume"),
        } for c in arr]
        return {"items": items, "generated_at": datetime.utcnow().isoformat() + "Z"}
    except Exception:
        raise HTTPException(502, "Не удалось получить топ-монеты. Попробуйте позже.")

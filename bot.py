import time, json, logging, os, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════
#  APEX BOT — Liquidity Concept
#  Flow  : H1 Bias → M15 ERL/IRL Mapping → M15 Sweep
#          → M5 IFVG + MSS Konfirmasi → Entry
#  SL    : Wick sweep candle M15 + buffer
#  TP    : IRL terdekat, minimal 1:2
# ═══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN     = "8806108760:AAF7NJUz1I3unPAMg7v5hSvAlAJ34PYi5G4"
TELEGRAM_CHAT_ID   = "6273206309"
TWELVEDATA_API_KEY = "a2a93ff31acd4fd48f247d0a4e300c46"

SYMBOL         = "XAU/USD"
TF_H1          = "1h"
TF_M15         = "15min"
TF_M5          = "5min"
CHECK_INTERVAL = 120       # cek tiap 2 menit
JOURNAL_FILE   = "journal.json"
SL_BUFFER      = 0.5       # buffer di atas/bawah wick (pips)
MIN_RR         = 2.0       # minimum risk:reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. AMBIL DATA CANDLE
# ═══════════════════════════════════════════════════════════════
def get_candles(interval, size=100):
    url    = "https://api.twelvedata.com/time_series"
    params = {
        "symbol"    : SYMBOL,
        "interval"  : interval,
        "outputsize": size,
        "apikey"    : TWELVEDATA_API_KEY,
        "format"    : "JSON"
    }
    try:
        r    = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "values" not in data:
            log.error(f"API error ({interval}): {data}")
            return None
        df = pd.DataFrame(data["values"])
        df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        log.error(f"Gagal ambil {interval}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# 2. H1 BIAS
#    Baca struktur H1: HH+HL = Bullish | LH+LL = Bearish
#    Fallback EMA50 jika swing tidak cukup
# ═══════════════════════════════════════════════════════════════
def get_h1_bias(df_h1):
    if df_h1 is None or len(df_h1) < 20:
        return "NEUTRAL"

    h = df_h1["high"].values
    l = df_h1["low"].values
    c = df_h1["close"].values
    n = len(df_h1)

    swing_h, swing_l = [], []
    for i in range(2, n-2):
        if h[i]>h[i-1] and h[i]>h[i-2] and h[i]>h[i+1] and h[i]>h[i+2]:
            swing_h.append(h[i])
        if l[i]<l[i-1] and l[i]<l[i-2] and l[i]<l[i+1] and l[i]<l[i+2]:
            swing_l.append(l[i])

    if len(swing_h) >= 2 and len(swing_l) >= 2:
        hh = swing_h[-1] > swing_h[-2]
        hl = swing_l[-1] > swing_l[-2]
        lh = swing_h[-1] < swing_h[-2]
        ll = swing_l[-1] < swing_l[-2]
        if hh and hl: return "BULLISH"
        if lh and ll: return "BEARISH"

    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    if c[-1] > ema50[-1] and ema50[-1] > ema50[-5]: return "BULLISH"
    if c[-1] < ema50[-1] and ema50[-1] < ema50[-5]: return "BEARISH"
    return "NEUTRAL"


# ═══════════════════════════════════════════════════════════════
# 3. MAPPING ERL & IRL di M15
#
#  ERL (External Range Liquidity) = target sweep institusi:
#    - BSL : Swing High, Equal Highs, Prev Session High
#    - SSL : Swing Low,  Equal Lows,  Prev Session Low
#
#  IRL (Internal Range Liquidity) = target TP setelah sweep:
#    - Bullish/Bearish FVG dalam range
#    - Bullish/Bearish OB dalam range
# ═══════════════════════════════════════════════════════════════
def map_erl_irl(df_m15):
    h = df_m15["high"].values
    l = df_m15["low"].values
    c = df_m15["close"].values
    o = df_m15["open"].values
    n = len(df_m15)
    price = c[-1]
    erl, irl = [], []

    # ── ERL: Swing High/Low M15 ───────────────────────
    for i in range(3, n-3):
        if h[i]>h[i-1] and h[i]>h[i-2] and h[i]>h[i-3] and \
           h[i]>h[i+1] and h[i]>h[i+2] and h[i]>h[i+3]:
            if not any(abs(e["level"]-h[i])<1.5 for e in erl):
                erl.append({"level":round(h[i],2),"type":"BSL (Swing High M15)","side":"BSL","idx":i})
        if l[i]<l[i-1] and l[i]<l[i-2] and l[i]<l[i-3] and \
           l[i]<l[i+1] and l[i]<l[i+2] and l[i]<l[i+3]:
            if not any(abs(e["level"]-l[i])<1.5 for e in erl):
                erl.append({"level":round(l[i],2),"type":"SSL (Swing Low M15)","side":"SSL","idx":i})

    # ── ERL: Equal Highs / Equal Lows ─────────────────
    tol = 1.0
    for i in range(max(0,n-50), n-5):
        for j in range(i+4, min(i+20,n-1)):
            if abs(h[i]-h[j]) <= tol:
                eq = round((h[i]+h[j])/2,2)
                if not any(abs(e["level"]-eq)<1.5 for e in erl):
                    erl.append({"level":eq,"type":"Equal Highs (BSL)","side":"BSL","idx":j})
                break
        for j in range(i+4, min(i+20,n-1)):
            if abs(l[i]-l[j]) <= tol:
                eq = round((l[i]+l[j])/2,2)
                if not any(abs(e["level"]-eq)<1.5 for e in erl):
                    erl.append({"level":eq,"type":"Equal Lows (SSL)","side":"SSL","idx":j})
                break

    # ── ERL: Previous Session High/Low ────────────────
    if n >= 50:
        sess = df_m15.iloc[-50:-8]
        ph   = round(sess["high"].max(),2)
        pl   = round(sess["low"].min(), 2)
        if not any(abs(e["level"]-ph)<1.5 for e in erl):
            erl.append({"level":ph,"type":"Prev Session High (BSL)","side":"BSL","idx":n-50})
        if not any(abs(e["level"]-pl)<1.5 for e in erl):
            erl.append({"level":pl,"type":"Prev Session Low (SSL)","side":"SSL","idx":n-50})

    # ── IRL: FVG M15 ──────────────────────────────────
    for i in range(1, n-1):
        if l[i+1] > h[i-1]:          # Bullish FVG
            flo,fhi = h[i-1],l[i+1]
            mid = round((flo+fhi)/2,2)
            if not any(abs(ir["level"]-mid)<1.0 for ir in irl):
                irl.append({"level":mid,"zone_lo":round(flo,2),"zone_hi":round(fhi,2),
                            "type":"Bullish FVG (IRL)","dir":"BULLISH","idx":i})
        if h[i+1] < l[i-1]:          # Bearish FVG
            fhi2,flo2 = l[i-1],h[i+1]
            mid2 = round((flo2+fhi2)/2,2)
            if not any(abs(ir["level"]-mid2)<1.0 for ir in irl):
                irl.append({"level":mid2,"zone_lo":round(flo2,2),"zone_hi":round(fhi2,2),
                            "type":"Bearish FVG (IRL)","dir":"BEARISH","idx":i})

    # ── IRL: OB M15 ───────────────────────────────────
    avg_body = np.mean([abs(c[i]-o[i]) for i in range(max(0,n-20),n)]) + 1e-9
    for i in range(max(0,n-20), n-2):
        body = abs(c[i]-o[i])
        if body < avg_body*1.1: continue
        mid_ob = round((h[i]+l[i])/2,2)
        if not any(abs(ir["level"]-mid_ob)<2.0 for ir in irl):
            if o[i] > c[i]:   # bearish candle → bullish OB
                irl.append({"level":mid_ob,"zone_lo":round(l[i],2),"zone_hi":round(h[i],2),
                            "type":"Bullish OB (IRL)","dir":"BULLISH","idx":i})
            elif c[i] > o[i]: # bullish candle → bearish OB
                irl.append({"level":mid_ob,"zone_lo":round(l[i],2),"zone_hi":round(h[i],2),
                            "type":"Bearish OB (IRL)","dir":"BEARISH","idx":i})

    # Filter radius 80 pips
    erl = [e for e in erl if abs(e["level"]-price)<=80]
    irl = [ir for ir in irl if abs(ir["level"]-price)<=80]
    return erl, irl


# ═══════════════════════════════════════════════════════════════
# 4. DETEKSI SWEEP ERL di M15
# ═══════════════════════════════════════════════════════════════
def detect_m15_sweep(df_m15, erl):
    sweeps = []
    n = len(df_m15)
    for lv in erl:
        level = lv["level"]
        side  = lv["side"]
        for i in range(max(1,n-4), n):
            row  = df_m15.iloc[i]
            prev = df_m15.iloc[i-1]
            # BSL Sweep: spike atas level, close kembali di bawah
            if side=="BSL" and row["high"]>level and row["close"]<level and prev["close"]<=level:
                sweeps.append({
                    "lv_type"  : lv["type"],
                    "lv_level" : level,
                    "side"     : "BSL",
                    "direction": "BEARISH",
                    "idx"      : i,
                    "wick_hi"  : round(row["high"],2),
                    "wick_lo"  : round(row["low"], 2),
                    "detail"   : f"M15 Sweep BSL *{lv['type']}* @ {level:.2f}"
                })
            # SSL Sweep: spike bawah level, close kembali di atas
            elif side=="SSL" and row["low"]<level and row["close"]>level and prev["close"]>=level:
                sweeps.append({
                    "lv_type"  : lv["type"],
                    "lv_level" : level,
                    "side"     : "SSL",
                    "direction": "BULLISH",
                    "idx"      : i,
                    "wick_hi"  : round(row["high"],2),
                    "wick_lo"  : round(row["low"], 2),
                    "detail"   : f"M15 Sweep SSL *{lv['type']}* @ {level:.2f}"
                })
    return sweeps


# ═══════════════════════════════════════════════════════════════
# 5. KONFIRMASI ENTRY M5: IFVG + MSS
#    Keduanya harus ada agar sinyal valid
# ═══════════════════════════════════════════════════════════════
def find_m5_entry(df_m5, direction):
    n       = len(df_m5)
    h       = df_m5["high"].values
    l       = df_m5["low"].values
    c       = df_m5["close"].values
    current = c[-1]
    ifvg_found = None
    mss_found  = None

    # ── IFVG M5 ───────────────────────────────────────
    for i in range(max(1,n-40), n-1):
        if i+1 >= n: break
        if direction=="BEARISH":
            # Bullish FVG yang terinversi → sekarang jadi resistance (IFVG Bearish)
            if l[i+1] > h[i-1]:
                flo, fhi = h[i-1], l[i+1]
                # Terinversi: harga pernah turun ke bawah flo dan sekarang retracement ke zona
                if flo <= current <= fhi:
                    ifvg_found = {
                        "type"   : "IFVG",
                        "zone_lo": round(flo,2),
                        "zone_hi": round(fhi,2),
                        "detail" : f"M5 IFVG Bearish [{flo:.2f}–{fhi:.2f}]"
                    }
                    break
        elif direction=="BULLISH":
            # Bearish FVG yang terinversi → sekarang jadi support (IFVG Bullish)
            if h[i+1] < l[i-1]:
                fhi2, flo2 = l[i-1], h[i+1]
                if flo2 <= current <= fhi2:
                    ifvg_found = {
                        "type"   : "IFVG",
                        "zone_lo": round(flo2,2),
                        "zone_hi": round(fhi2,2),
                        "detail" : f"M5 IFVG Bullish [{flo2:.2f}–{fhi2:.2f}]"
                    }
                    break

    # ── MSS M5 ────────────────────────────────────────
    for i in range(3, n-3):
        is_sh = h[i]>h[i-1] and h[i]>h[i-2] and h[i]>h[i+1] and h[i]>h[i+2]
        is_sl = l[i]<l[i-1] and l[i]<l[i-2] and l[i]<l[i+1] and l[i]<l[i+2]
        if direction=="BEARISH" and is_sl:
            swing_lo = l[i]
            if c[-2] >= swing_lo and current < swing_lo:
                mss_found = {
                    "type"   : "MSS",
                    "level"  : round(swing_lo,2),
                    "detail" : f"M5 MSS Bearish — close {current:.2f} < swing low {swing_lo:.2f}"
                }
                break
        if direction=="BULLISH" and is_sh:
            swing_hi = h[i]
            if c[-2] <= swing_hi and current > swing_hi:
                mss_found = {
                    "type"   : "MSS",
                    "level"  : round(swing_hi,2),
                    "detail" : f"M5 MSS Bullish — close {current:.2f} > swing high {swing_hi:.2f}"
                }
                break

    return ifvg_found, mss_found


# ═══════════════════════════════════════════════════════════════
# 6. HITUNG SL & TP
#    SL  = wick sweep M15 + buffer
#    TP  = IRL terdekat searah, jika tidak ada pakai 1:2 dari SL
# ═══════════════════════════════════════════════════════════════
def calc_sl_tp(sweep, entry_price, direction, irl):
    if direction=="BEARISH":
        sl   = round(sweep["wick_hi"] + SL_BUFFER, 2)
        risk = abs(sl - entry_price)
        tp_default = round(entry_price - risk * MIN_RR, 2)

        # Cari IRL Bearish/Bullish di bawah harga sebagai target TP
        irl_targets = sorted(
            [ir for ir in irl if ir["level"] < entry_price - risk],
            key=lambda x: x["level"], reverse=True
        )
        if irl_targets:
            tp_irl  = irl_targets[0]["zone_lo"]
            rr_irl  = round(abs(entry_price - tp_irl) / risk, 1) if risk > 0 else 0
            if rr_irl >= MIN_RR:
                return sl, round(tp_irl,2), round(risk,2), rr_irl, irl_targets[0]["type"]

        rr = round(risk * MIN_RR / risk, 1) if risk > 0 else MIN_RR
        return sl, tp_default, round(risk,2), MIN_RR, "1:2 Default"

    else:  # BULLISH
        sl   = round(sweep["wick_lo"] - SL_BUFFER, 2)
        risk = abs(entry_price - sl)
        tp_default = round(entry_price + risk * MIN_RR, 2)

        irl_targets = sorted(
            [ir for ir in irl if ir["level"] > entry_price + risk],
            key=lambda x: x["level"]
        )
        if irl_targets:
            tp_irl  = irl_targets[0]["zone_hi"]
            rr_irl  = round(abs(tp_irl - entry_price) / risk, 1) if risk > 0 else 0
            if rr_irl >= MIN_RR:
                return sl, round(tp_irl,2), round(risk,2), rr_irl, irl_targets[0]["type"]

        return sl, tp_default, round(risk,2), MIN_RR, "1:2 Default"


# ═══════════════════════════════════════════════════════════════
# 7. SCORING
# ═══════════════════════════════════════════════════════════════
def calc_score(sweep, ifvg, mss, bias, rr):
    score = 0

    # IFVG + MSS keduanya ada = strong konfirmasi
    if ifvg and mss: score += 4
    elif ifvg:       score += 2
    elif mss:        score += 2

    # Kualitas sweep ERL
    lv = sweep["lv_type"]
    if "Equal"   in lv: score += 3
    elif "Prev"  in lv: score += 3
    elif "Swing" in lv: score += 2
    else:               score += 1

    # H1 bias alignment
    direction = sweep["direction"]
    if (direction=="BEARISH" and bias=="BEARISH") or \
       (direction=="BULLISH" and bias=="BULLISH"): score += 2
    elif bias=="NEUTRAL": score += 1

    # RR bonus
    if rr >= 3.0:   score += 2
    elif rr >= 2.0: score += 1

    return score


# ═══════════════════════════════════════════════════════════════
# 8. FORMAT PESAN SINYAL
# ═══════════════════════════════════════════════════════════════
def format_signal(sweep, ifvg, mss, sl, tp, risk, rr, tp_type,
                  bias, score, erl, irl, df_m5):
    price     = df_m5["close"].iloc[-1]
    now       = datetime.now().strftime("%Y-%m-%d %H:%M")
    direction = sweep["direction"]
    dir_lbl   = "SELL 📉" if direction=="BEARISH" else "BUY 📈"

    if score >= 9:   emoji, strength = "🚨", "🔥 SANGAT KUAT"
    elif score >= 7: emoji, strength = "📣", "💪 KUAT"
    elif score >= 5: emoji, strength = "📊", "✅ SEDANG"
    else:            emoji, strength = "💡", "⚡ LEMAH"

    # Ringkasan ERL & IRL terdekat
    erl_nearby = [e for e in erl if abs(e["level"]-price)<=30]
    irl_nearby = [ir for ir in irl if abs(ir["level"]-price)<=30]

    erl_txt = "\n".join([f"   • {e['type']}: {e['level']:.2f}" for e in erl_nearby[:3]]) or "   -"
    irl_txt = "\n".join([f"   • {ir['type']}: {ir['zone_lo']:.2f}–{ir['zone_hi']:.2f}" for ir in irl_nearby[:3]]) or "   -"

    entry_txt = []
    if ifvg: entry_txt.append(f"   ✅ {ifvg['detail']}")
    if mss:  entry_txt.append(f"   ✅ {mss['detail']}")

    lines = [
        f"{emoji} *APEX SIGNAL — {SYMBOL}*",
        f"🕐 {now}  |  H1→M15→M5",
        f"",
        f"🎯 Sinyal   : *{dir_lbl}*",
        f"💪 Kekuatan : *{strength}* (Score: {score})",
        f"📈 H1 Bias  : *{bias}*",
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"🔍 *SETUP:*",
        f"   🔸 Sweep M15 : {sweep['detail']}",
        f"      Wick : {sweep['wick_lo']:.2f} — {sweep['wick_hi']:.2f}",
        f"",
        f"   🔸 Konfirmasi M5:",
        *entry_txt,
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"🗺️ *LIQUIDITY MAP:*",
        f"   ERL (target sweep):",
        erl_txt,
        f"   IRL (target TP):",
        irl_txt,
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"📐 *MANAJEMEN RISIKO:*",
        f"   • Entry : ~*{price:.2f}*",
        f"   • SL    : *{sl:.2f}*  ← wick M15 + {SL_BUFFER} buffer",
        f"   • TP    : *{tp:.2f}*  ← {tp_type}",
        f"   • Risk  : {risk:.1f} pips",
        f"   • R:R   : *1:{rr}* ✅",
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"⚠️ _Konfirmasi di chart sebelum entry._",
        f"_SL wajib di level yang tertera!_",
        f"_📓 Sinyal otomatis dicatat di journal._",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 9. AUTO JOURNAL
# ═══════════════════════════════════════════════════════════════
def load_journal():
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE,"r") as f:
                return json.load(f)
        except: pass
    return {
        "signals": [],
        "stats"  : {"total":0,"win":0,"loss":0,"pending":0,
                    "total_pips_win":0.0,"total_pips_loss":0.0}
    }


def save_journal(journal):
    with open(JOURNAL_FILE,"w") as f:
        json.dump(journal, f, indent=2)


def log_signal_to_journal(journal, sig_id, sweep, entry, sl, tp,
                           risk, rr, ifvg, mss, bias, score):
    entry_type = []
    if ifvg: entry_type.append("IFVG")
    if mss:  entry_type.append("MSS")

    journal["signals"].append({
        "id"         : sig_id,
        "time"       : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "direction"  : sweep["direction"],
        "sweep_type" : sweep["lv_type"],
        "entry_type" : "+".join(entry_type),
        "entry"      : entry,
        "sl"         : sl,
        "tp"         : tp,
        "risk_pips"  : risk,
        "rr"         : rr,
        "bias"       : bias,
        "score"      : score,
        "result"     : "PENDING ⏳",
        "pnl_pips"   : 0.0,
        "close_time" : ""
    })
    journal["stats"]["total"]   += 1
    journal["stats"]["pending"] += 1
    save_journal(journal)


def update_trade_results(journal, df_m5):
    """
    Cek setiap sinyal PENDING apakah sudah hit TP atau SL.
    Jika ya, update journal dan kirim notif hasil.
    """
    current_hi = df_m5["high"].iloc[-1]
    current_lo = df_m5["low"].iloc[-1]
    updated    = False

    for sig in journal["signals"]:
        if "PENDING" not in sig["result"]:
            continue

        direction = sig["direction"]
        tp  = sig["tp"]
        sl  = sig["sl"]
        entry = sig["entry"]
        risk  = abs(entry - sl)

        result = None
        pnl    = 0.0

        if direction == "BEARISH":
            if current_lo <= tp:
                result = "WIN ✅"
                pnl    = round(entry - tp, 2)
                journal["stats"]["win"]            += 1
                journal["stats"]["total_pips_win"] += pnl
            elif current_hi >= sl:
                result = "LOSS ❌"
                pnl    = round(entry - sl, 2)   # negatif
                journal["stats"]["loss"]             += 1
                journal["stats"]["total_pips_loss"]  += abs(pnl)
        else:
            if current_hi >= tp:
                result = "WIN ✅"
                pnl    = round(tp - entry, 2)
                journal["stats"]["win"]            += 1
                journal["stats"]["total_pips_win"] += pnl
            elif current_lo <= sl:
                result = "LOSS ❌"
                pnl    = round(sl - entry, 2)   # negatif
                journal["stats"]["loss"]             += 1
                journal["stats"]["total_pips_loss"]  += abs(pnl)

        if result:
            sig["result"]     = result
            sig["pnl_pips"]   = pnl
            sig["close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            journal["stats"]["pending"] -= 1
            updated = True
            send_result_notif(sig)

    if updated:
        save_journal(journal)

    return journal


def send_result_notif(sig):
    won   = "WIN" in sig["result"]
    emoji = "✅" if won else "❌"
    pnl   = abs(sig["pnl_pips"])
    rr_actual = round(pnl / sig["risk_pips"], 1) if sig["risk_pips"] > 0 else 0

    msg = (
        f"{emoji} *TRADE SELESAI — {SYMBOL}*\n\n"
        f"🎯 Arah    : *{sig['direction']}*\n"
        f"📋 Setup   : {sig['entry_type']} | {sig['sweep_type']}\n"
        f"📈 Bias H1 : {sig['bias']}\n\n"
        f"💰 Entry   : {sig['entry']:.2f}\n"
        f"{'✅ TP' if won else '❌ SL'}     : {sig['tp'] if won else sig['sl']:.2f}\n\n"
        f"📊 Hasil   : *{'+' if won else '-'}{pnl:.1f} pips*\n"
        f"📐 R:R     : *1:{rr_actual}*\n\n"
        f"🕐 Buka    : {sig['time']}\n"
        f"🕐 Tutup   : {sig['close_time']}\n\n"
        f"_📓 Tercatat di journal. Ketik /journal untuk statistik._"
    )
    send_telegram(msg)


def get_stats_message(journal):
    s       = journal["stats"]
    total   = s["total"]
    win     = s["win"]
    loss    = s["loss"]
    pending = s["pending"]
    closed  = total - pending
    wr      = round(win/closed*100, 1) if closed > 0 else 0
    net     = round(s["total_pips_win"] - s["total_pips_loss"], 1)

    # Breakdown per sweep type
    breakdown = {}
    for sig in journal["signals"]:
        t = sig["sweep_type"]
        if t not in breakdown:
            breakdown[t] = {"w":0,"l":0}
        if "WIN" in sig["result"]:  breakdown[t]["w"] += 1
        if "LOSS" in sig["result"]: breakdown[t]["l"] += 1

    bdown_txt = "\n".join([
        f"   • {k}: {v['w']}W/{v['l']}L "
        f"({round(v['w']/max(v['w']+v['l'],1)*100)}%)"
        for k,v in sorted(breakdown.items(), key=lambda x: -(x[1]['w']+x[1]['l']))
        if v['w']+v['l'] > 0
    ]) or "   Belum ada data"

    # Breakdown per entry type
    entry_bdown = {}
    for sig in journal["signals"]:
        t = sig["entry_type"]
        if t not in entry_bdown:
            entry_bdown[t] = {"w":0,"l":0}
        if "WIN" in sig["result"]:  entry_bdown[t]["w"] += 1
        if "LOSS" in sig["result"]: entry_bdown[t]["l"] += 1

    entry_txt = "\n".join([
        f"   • {k}: {v['w']}W/{v['l']}L "
        f"({round(v['w']/max(v['w']+v['l'],1)*100)}%)"
        for k,v in sorted(entry_bdown.items(), key=lambda x: -(x[1]['w']+x[1]['l']))
        if v['w']+v['l'] > 0
    ]) or "   Belum ada data"

    return (
        f"📓 *APEX JOURNAL — {SYMBOL}*\n"
        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"📊 *Statistik:*\n"
        f"   Total   : {total} sinyal\n"
        f"   Closed  : {closed} trade\n"
        f"   WIN     : {win} ✅  ({s['total_pips_win']:.1f} pips)\n"
        f"   LOSS    : {loss} ❌  ({s['total_pips_loss']:.1f} pips)\n"
        f"   Pending : {pending} ⏳\n"
        f"   Winrate : *{wr}%*\n"
        f"   Net P&L : *{'+' if net>=0 else ''}{net} pips*\n\n"
        f"📈 *Per Sweep Type:*\n{bdown_txt}\n\n"
        f"🎯 *Per Entry Type:*\n{entry_txt}\n\n"
        f"_Ketik /journal kapanpun untuk update ini._"
    )


# ═══════════════════════════════════════════════════════════════
# 10. TELEGRAM
# ═══════════════════════════════════════════════════════════════
def send_telegram(msg):
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id":TELEGRAM_CHAT_ID,"text":msg,"parse_mode":"Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            log.error(f"Telegram error: {r.text}")
    except Exception as e:
        log.error(f"Send error: {e}")


def check_commands(journal):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    try:
        r    = requests.get(url, timeout=5)
        data = r.json()
        if not data.get("ok"): return
        for upd in data.get("result",[])[-5:]:
            txt = upd.get("message",{}).get("text","").strip()
            if txt == "/journal":
                send_telegram(get_stats_message(journal))
            elif txt == "/status":
                pending = [s for s in journal["signals"] if "PENDING" in s["result"]]
                if pending:
                    lines = [f"⏳ *Trade Pending ({len(pending)}):*\n"]
                    for s in pending[-5:]:
                        lines.append(
                            f"• {s['direction']} | Entry:{s['entry']:.2f} "
                            f"SL:{s['sl']:.2f} TP:{s['tp']:.2f} | {s['time']}"
                        )
                    send_telegram("\n".join(lines))
                else:
                    send_telegram("✅ Tidak ada trade pending saat ini.")
    except: pass


# ═══════════════════════════════════════════════════════════════
# 11. MAIN LOOP
# ═══════════════════════════════════════════════════════════════
sent_cache = set()
journal    = load_journal()
loop_count = 0

def run_bot():
    global loop_count, journal

    log.info("🚀 APEX Liquidity Bot mulai...")
    send_telegram(
        f"🚀 *APEX Liquidity Bot Aktif!*\n\n"
        f"*Pair    :* {SYMBOL}\n"
        f"*Flow    :* H1 Bias → M15 Sweep → M5 Entry\n\n"
        f"*Liquidity Concept:*\n"
        f"✅ ERL: BSL/SSL, Equal H/L, Prev Session H/L\n"
        f"✅ IRL: FVG + OB sebagai target TP\n"
        f"✅ Konfirmasi: IFVG + MSS di M5\n\n"
        f"*Risk Management:*\n"
        f"✅ SL = Wick sweep candle M15\n"
        f"✅ TP = IRL terdekat (min 1:{int(MIN_RR)})\n\n"
        f"*Commands:*\n"
        f"/journal → statistik lengkap\n"
        f"/status  → trade pending\n\n"
        f"_Memantau 24 jam..._ 👁️"
    )

    while True:
        try:
            loop_count += 1

            # Cek command user setiap 5 loop
            if loop_count % 5 == 0:
                check_commands(journal)

            # Kirim statistik setiap 6 jam
            if loop_count % 180 == 0 and journal["stats"]["total"] > 0:
                send_telegram(get_stats_message(journal))

            # ── Ambil semua data
            df_h1  = get_candles(TF_H1,  100)
            df_m15 = get_candles(TF_M15, 100)
            df_m5  = get_candles(TF_M5,  100)

            if any(df is None for df in [df_h1, df_m15, df_m5]):
                time.sleep(CHECK_INTERVAL); continue
            if any(len(df) < 20 for df in [df_h1, df_m15, df_m5]):
                time.sleep(CHECK_INTERVAL); continue

            # ── Update hasil trade pending
            journal = update_trade_results(journal, df_m5)

            # ── STEP 1: H1 Bias
            bias = get_h1_bias(df_h1)
            log.info(f"H1 Bias: {bias}")

            # ── STEP 2: M15 ERL + IRL Mapping
            erl, irl = map_erl_irl(df_m15)
            if not erl:
                log.info("Tidak ada ERL level"); time.sleep(CHECK_INTERVAL); continue

            # ── STEP 3: Deteksi Sweep ERL di M15
            sweeps = detect_m15_sweep(df_m15, erl)
            if not sweeps:
                log.info("Tidak ada sweep M15"); time.sleep(CHECK_INTERVAL); continue

            for sweep in sweeps:
                direction = sweep["direction"]

                # Skip jika bias berlawanan
                if bias != "NEUTRAL":
                    if direction=="BEARISH" and bias=="BULLISH":
                        log.info("Bias BULLISH vs sweep BEARISH — skip"); continue
                    if direction=="BULLISH" and bias=="BEARISH":
                        log.info("Bias BEARISH vs sweep BULLISH — skip"); continue

                # ── STEP 4: Konfirmasi M5 — IFVG + MSS
                ifvg, mss = find_m5_entry(df_m5, direction)

                # Wajib minimal salah satu ada
                if not ifvg and not mss:
                    log.info(f"Tidak ada IFVG/MSS M5 untuk {direction}"); continue

                entry_price = df_m5["close"].iloc[-1]

                # ── STEP 5: SL dari wick, TP ke IRL min 1:2
                sl, tp, risk, rr, tp_type = calc_sl_tp(sweep, entry_price, direction, irl)

                if rr < MIN_RR:
                    log.info(f"RR {rr} < {MIN_RR} — skip"); continue

                # ── STEP 6: Scoring
                score = calc_score(sweep, ifvg, mss, bias, rr)

                # ── STEP 7: Hindari duplikat
                key = f"{direction}_{sweep['lv_level']}_{entry_price:.1f}"
                if key in sent_cache: continue

                # ── STEP 8: Kirim sinyal
                msg = format_signal(sweep, ifvg, mss, sl, tp, risk, rr,
                                    tp_type, bias, score, erl, irl, df_m5)
                send_telegram(msg)

                # ── STEP 9: Catat ke journal
                sig_id = f"{datetime.now().strftime('%Y%m%d%H%M')}_{direction[:4]}"
                log_signal_to_journal(journal, sig_id, sweep, entry_price,
                                      sl, tp, risk, rr, ifvg, mss, bias, score)

                sent_cache.add(key)
                if len(sent_cache) > 100: sent_cache.clear()

                log.info(f"✅ {direction} | Score:{score} | RR:{rr} | {sweep['lv_type']}")

        except Exception as e:
            log.error(f"Error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run_bot()

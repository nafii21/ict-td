import time, json, logging, os, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════
#  APEX — ICT SILVER BULLET BOT
#  Timeframe : M5 (entry) + H1 (bias)
#  Windows   : 3 Silver Bullet NY time windows
#  Entry     : FVG setelah displacement dalam window
#  SL        : Di atas/bawah FVG
#  TP        : Minimal 1:2
# ═══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN     = "8806108760:AAF7NJUz1I3unPAMg7v5hSvAlAJ34PYi5G4"
TELEGRAM_CHAT_ID   = "6273206309"
TWELVEDATA_API_KEY = "a2a93ff31acd4fd48f247d0a4e300c46"

SYMBOL         = "XAU/USD"
TF_M5          = "5min"
TF_H1          = "1h"
CHECK_INTERVAL = 60        # cek tiap 1 menit (penting saat dalam window)
JOURNAL_FILE   = "journal.json"
SL_BUFFER      = 0.3       # buffer di atas/bawah FVG untuk SL
MIN_RR         = 2.0

# ── Silver Bullet Windows (UTC) ──────────────────────────────
# NY EDT (summer) = UTC-4 | NY EST (winter) = UTC-5
# Window 1: 03:00-04:00 NY = 07:00-09:00 UTC
# Window 2: 10:00-11:00 NY = 14:00-16:00 UTC
# Window 3: 14:00-15:00 NY = 18:00-20:00 UTC
WINDOWS = [
    {"name": "🌅 London Open SB (03:00–04:00 NY)", "utc_start":  7, "utc_end":  9},
    {"name": "🗽 NY Morning SB  (10:00–11:00 NY)", "utc_start": 14, "utc_end": 16},
    {"name": "🌆 NY Afternoon SB(14:00–15:00 NY)", "utc_start": 18, "utc_end": 20},
]

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
# 2. CEK SILVER BULLET WINDOW
#    Return (True, window_name) jika sedang dalam window
# ═══════════════════════════════════════════════════════════════
def get_active_window():
    hour_utc = datetime.now(timezone.utc).hour
    for w in WINDOWS:
        if w["utc_start"] <= hour_utc < w["utc_end"]:
            return True, w["name"]
    return False, None


# ═══════════════════════════════════════════════════════════════
# 3. H1 BIAS via EMA 50
#    Bullish → prioritaskan BUY FVG
#    Bearish → prioritaskan SELL FVG
# ═══════════════════════════════════════════════════════════════
def get_h1_bias(df_h1):
    if df_h1 is None or len(df_h1) < 55:
        return "NEUTRAL"
    c     = df_h1["close"].values
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    if c[-1] > ema50[-1] and ema50[-1] > ema50[-5]: return "BULLISH"
    if c[-1] < ema50[-1] and ema50[-1] < ema50[-5]: return "BEARISH"
    return "NEUTRAL"


# ═══════════════════════════════════════════════════════════════
# 4. DETEKSI DISPLACEMENT M5
#    Candle impulsif besar = displacement
#    Body > 1.5x rata-rata body 20 candle terakhir
# ═══════════════════════════════════════════════════════════════
def find_displacements(df_m5):
    n = len(df_m5)
    o = df_m5["open"].values
    h = df_m5["high"].values
    l = df_m5["low"].values
    c = df_m5["close"].values

    avg_body = np.mean([abs(c[i]-o[i]) for i in range(max(0,n-20), n)]) + 1e-9
    displacements = []

    for i in range(max(1, n-20), n-1):
        body = abs(c[i]-o[i])
        if body >= avg_body * 1.5:
            direction = "BULLISH" if c[i] > o[i] else "BEARISH"
            displacements.append({
                "idx"      : i,
                "direction": direction,
                "body"     : round(body, 2),
                "high"     : round(h[i], 2),
                "low"      : round(l[i], 2),
                "open"     : round(o[i], 2),
                "close"    : round(c[i], 2),
            })

    return displacements


# ═══════════════════════════════════════════════════════════════
# 5. DETEKSI FVG SETELAH DISPLACEMENT
#    Bullish FVG : low[i+1] > high[i-1]  → harga gap naik
#    Bearish FVG : high[i+1] < low[i-1]  → harga gap turun
#    Harga retrace masuk FVG = entry signal
# ═══════════════════════════════════════════════════════════════
def find_silver_bullet_fvg(df_m5, bias):
    n       = len(df_m5)
    h       = df_m5["high"].values
    l       = df_m5["low"].values
    c       = df_m5["close"].values
    o       = df_m5["open"].values
    current = c[-1]
    signals = []

    avg_body = np.mean([abs(c[i]-o[i]) for i in range(max(0,n-20), n)]) + 1e-9

    for i in range(max(2, n-25), n-1):
        # ── Cek apakah candle i-1 adalah displacement ──────────
        body_prev = abs(c[i-1]-o[i-1])
        is_displacement = body_prev >= avg_body * 1.5

        # ── BULLISH FVG ────────────────────────────────────────
        # Terbentuk saat impulse naik: low[i+1] > high[i-1]
        if i+1 < n and l[i+1] > h[i-1]:
            fvg_lo = h[i-1]
            fvg_hi = l[i+1]
            fvg_mid= round((fvg_lo+fvg_hi)/2, 2)

            # Harga retrace masuk FVG
            if fvg_lo <= current <= fvg_hi:
                # Skip jika bias berlawanan
                if bias == "BEARISH": continue

                sl  = round(fvg_lo - SL_BUFFER, 2)
                risk= round(current - sl, 2)
                tp  = round(current + risk * MIN_RR, 2) if risk > 0 else 0

                signals.append({
                    "direction"    : "BULLISH",
                    "dir_label"    : "BUY 📈",
                    "fvg_lo"       : round(fvg_lo, 2),
                    "fvg_hi"       : round(fvg_hi, 2),
                    "fvg_mid"      : fvg_mid,
                    "sl"           : sl,
                    "tp"           : tp,
                    "risk"         : risk,
                    "rr"           : MIN_RR,
                    "displacement" : is_displacement,
                    "detail"       : f"Bullish FVG [{fvg_lo:.2f} — {fvg_hi:.2f}]"
                })

        # ── BEARISH FVG ────────────────────────────────────────
        # Terbentuk saat impulse turun: high[i+1] < low[i-1]
        if i+1 < n and h[i+1] < l[i-1]:
            fvg_hi = l[i-1]
            fvg_lo = h[i+1]
            fvg_mid= round((fvg_lo+fvg_hi)/2, 2)

            if fvg_lo <= current <= fvg_hi:
                if bias == "BULLISH": continue

                sl  = round(fvg_hi + SL_BUFFER, 2)
                risk= round(sl - current, 2)
                tp  = round(current - risk * MIN_RR, 2) if risk > 0 else 0

                signals.append({
                    "direction"    : "BEARISH",
                    "dir_label"    : "SELL 📉",
                    "fvg_lo"       : round(fvg_lo, 2),
                    "fvg_hi"       : round(fvg_hi, 2),
                    "fvg_mid"      : fvg_mid,
                    "sl"           : sl,
                    "tp"           : tp,
                    "risk"         : risk,
                    "rr"           : MIN_RR,
                    "displacement" : is_displacement,
                    "detail"       : f"Bearish FVG [{fvg_lo:.2f} — {fvg_hi:.2f}]"
                })

    # Prioritaskan FVG dengan displacement
    signals.sort(key=lambda x: (x["displacement"], x["risk"]), reverse=True)
    return signals[:1] if signals else []


# ═══════════════════════════════════════════════════════════════
# 6. FORMAT PESAN SINYAL
# ═══════════════════════════════════════════════════════════════
def format_signal(sig, bias, window_name, entry_price):
    now       = datetime.now().strftime("%Y-%m-%d %H:%M")
    direction = sig["direction"]

    lines = [
        f"🎯 *ICT SILVER BULLET — {SYMBOL}*",
        f"🕐 {now}  |  M5",
        f"",
        f"⏰ Window  : *{window_name}*",
        f"📊 Sinyal  : *{sig['dir_label']}*",
        f"📈 H1 Bias : *{bias}*",
        f"💥 Displace: *{'✅ Ada' if sig['displacement'] else '⚠️ Lemah'}*",
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"📦 *FAIR VALUE GAP:*",
        f"   {sig['detail']}",
        f"   Mid FVG : {sig['fvg_mid']:.2f}",
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"📐 *MANAJEMEN RISIKO:*",
        f"   • Entry : *{entry_price:.2f}*  (di zona FVG)",
        f"   • SL    : *{sig['sl']:.2f}*",
        f"     ↳ {'Di bawah FVG' if direction=='BULLISH' else 'Di atas FVG'} + {SL_BUFFER} buffer",
        f"   • TP    : *{sig['tp']:.2f}*",
        f"     ↳ Minimal 1:{int(MIN_RR)} dari SL",
        f"   • Risk  : {sig['risk']:.1f} pips",
        f"   • R:R   : *1:{sig['rr']}* ✅",
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"⚠️ _Konfirmasi di chart M5 sebelum entry._",
        f"_Wajib pasang SL! Bukan saran finansial._",
        f"_📓 Dicatat otomatis di journal._",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 7. AUTO JOURNAL
# ═══════════════════════════════════════════════════════════════
def load_journal():
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE,"r") as f: return json.load(f)
        except: pass
    return {"signals":[],"stats":{
        "total":0,"win":0,"loss":0,"pending":0,
        "pips_win":0.0,"pips_loss":0.0,
        "by_window":{}
    }}

def save_journal(j):
    with open(JOURNAL_FILE,"w") as f: json.dump(j,f,indent=2)

def log_to_journal(journal, sig_id, sig, entry, bias, window_name):
    journal["signals"].append({
        "id"        : sig_id,
        "time"      : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "window"    : window_name,
        "direction" : sig["direction"],
        "bias"      : bias,
        "entry"     : entry,
        "sl"        : sig["sl"],
        "tp"        : sig["tp"],
        "risk_pips" : sig["risk"],
        "rr"        : sig["rr"],
        "fvg_lo"    : sig["fvg_lo"],
        "fvg_hi"    : sig["fvg_hi"],
        "result"    : "PENDING ⏳",
        "pnl_pips"  : 0.0,
        "close_time": ""
    })
    journal["stats"]["total"]   += 1
    journal["stats"]["pending"] += 1

    # Track per window
    wk = window_name.split("(")[0].strip()
    if wk not in journal["stats"]["by_window"]:
        journal["stats"]["by_window"][wk] = {"w":0,"l":0}

    save_journal(journal)

def update_results(journal, df_m5):
    hi = df_m5["high"].iloc[-1]
    lo = df_m5["low"].iloc[-1]
    updated = False
    for sig in journal["signals"]:
        if "PENDING" not in sig["result"]: continue
        d=sig["direction"]; tp=sig["tp"]; sl=sig["sl"]
        entry=sig["entry"]; risk=abs(entry-sl)
        result=None; pnl=0.0
        if d=="BEARISH":
            if lo<=tp:
                result="WIN ✅"; pnl=round(entry-tp,2)
                journal["stats"]["win"]+=1; journal["stats"]["pips_win"]+=pnl
            elif hi>=sl:
                result="LOSS ❌"; pnl=round(entry-sl,2)
                journal["stats"]["loss"]+=1; journal["stats"]["pips_loss"]+=abs(pnl)
        else:
            if hi>=tp:
                result="WIN ✅"; pnl=round(tp-entry,2)
                journal["stats"]["win"]+=1; journal["stats"]["pips_win"]+=pnl
            elif lo<=sl:
                result="LOSS ❌"; pnl=round(sl-entry,2)
                journal["stats"]["loss"]+=1; journal["stats"]["pips_loss"]+=abs(pnl)
        if result:
            sig["result"]=result; sig["pnl_pips"]=pnl
            sig["close_time"]=datetime.now().strftime("%Y-%m-%d %H:%M")
            journal["stats"]["pending"]-=1
            # Update by_window stats
            wk = sig["window"].split("(")[0].strip()
            if wk in journal["stats"]["by_window"]:
                if "WIN" in result: journal["stats"]["by_window"][wk]["w"]+=1
                else:               journal["stats"]["by_window"][wk]["l"]+=1
            updated=True; send_result_notif(sig)
    if updated: save_journal(journal)
    return journal

def send_result_notif(sig):
    won=("WIN" in sig["result"]); emoji="✅" if won else "❌"
    pnl=abs(sig["pnl_pips"])
    rr_a=round(pnl/sig["risk_pips"],1) if sig["risk_pips"]>0 else 0
    msg=(
        f"{emoji} *SILVER BULLET SELESAI — {SYMBOL}*\n\n"
        f"⏰ Window  : {sig['window']}\n"
        f"🎯 Arah    : *{sig['direction']}*\n"
        f"📈 H1 Bias : {sig['bias']}\n\n"
        f"💰 Entry   : {sig['entry']:.2f}\n"
        f"{'✅ TP' if won else '❌ SL'}    : {sig['tp'] if won else sig['sl']:.2f}\n\n"
        f"📊 Hasil   : *{'+' if won else '-'}{pnl:.1f} pips*\n"
        f"📐 R:R     : *1:{rr_a}*\n"
        f"🕐 Buka    : {sig['time']}\n"
        f"🕐 Tutup   : {sig['close_time']}\n\n"
        f"🟢 *Bot siap untuk window berikutnya!*\n"
        f"_Ketik /journal untuk statistik._"
    )
    send_telegram(msg)

def has_running_trade(journal):
    return any("PENDING" in s["result"] for s in journal["signals"])

def get_running_trade(journal):
    for s in journal["signals"]:
        if "PENDING" in s["result"]: return s
    return None

def get_stats_message(journal):
    s=journal["stats"]; total=s["total"]; win=s["win"]; loss=s["loss"]
    pending=s["pending"]; closed=total-pending
    wr=round(win/closed*100,1) if closed>0 else 0
    net=round(s["pips_win"]-s["pips_loss"],1)

    running=get_running_trade(journal)
    run_txt=""
    if running:
        run_txt=(
            f"\n⏳ *Trade Running:*\n"
            f"   {running['direction']} | Entry:{running['entry']:.2f} "
            f"SL:{running['sl']:.2f} TP:{running['tp']:.2f}\n"
        )

    # Per window breakdown
    bw = s.get("by_window",{})
    window_txt=""
    if bw:
        window_txt="\n📅 *Per Window:*\n"
        for k,v in bw.items():
            total_w=v['w']+v['l']
            wr_w=round(v['w']/total_w*100) if total_w>0 else 0
            window_txt+=f"   • {k}: {v['w']}W/{v['l']}L ({wr_w}%)\n"

    # 5 trade terakhir
    recent=[s for s in journal["signals"] if "PENDING" not in s["result"]][-5:]
    history=""
    if recent:
        history="\n📋 *5 Trade Terakhir:*\n"
        for t in reversed(recent):
            icon="✅" if "WIN" in t["result"] else "❌"
            pips=f"{'+' if 'WIN' in t['result'] else '-'}{abs(t['pnl_pips']):.1f}"
            history+=f"   {icon} {t['direction']} {pips}p | {t['close_time']}\n"

    return (
        f"📓 *SILVER BULLET JOURNAL — {SYMBOL}*\n"
        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"{run_txt}\n"
        f"📊 *Statistik:*\n"
        f"   Total   : {total} sinyal\n"
        f"   WIN     : {win} ✅  (+{s['pips_win']:.1f} pips)\n"
        f"   LOSS    : {loss} ❌  (-{s['pips_loss']:.1f} pips)\n"
        f"   Pending : {pending} ⏳\n"
        f"   Winrate : *{wr}%*\n"
        f"   Net P&L : *{'+' if net>=0 else ''}{net} pips*\n"
        f"{window_txt}"
        f"{history}\n"
        f"_/journal /status_"
    )


# ═══════════════════════════════════════════════════════════════
# 8. TELEGRAM
# ═══════════════════════════════════════════════════════════════
def send_telegram(msg):
    url=f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload={"chat_id":TELEGRAM_CHAT_ID,"text":msg,"parse_mode":"Markdown"}
    try:
        r=requests.post(url,json=payload,timeout=10)
        if r.status_code!=200: log.error(f"Telegram error: {r.text}")
    except Exception as e: log.error(f"Send error: {e}")

def check_commands(journal):
    url=f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    try:
        r=requests.get(url,timeout=5); data=r.json()
        if not data.get("ok"): return
        for upd in data.get("result",[])[-5:]:
            txt=upd.get("message",{}).get("text","").strip()
            if txt=="/journal":
                send_telegram(get_stats_message(journal))
            elif txt=="/status":
                t=get_running_trade(journal)
                if t:
                    send_telegram(
                        f"⏳ *Trade Running:*\n\n"
                        f"Window : {t['window']}\n"
                        f"Arah   : *{t['direction']}*\n"
                        f"Entry  : {t['entry']:.2f}\n"
                        f"SL     : {t['sl']:.2f}\n"
                        f"TP     : {t['tp']:.2f}\n"
                        f"Waktu  : {t['time']}\n\n"
                        f"_Bot tidak entry baru sampai selesai._"
                    )
                else:
                    # Cek window berikutnya
                    now_utc = datetime.now(timezone.utc).hour
                    next_w  = None
                    for w in WINDOWS:
                        if w["utc_start"] > now_utc:
                            next_w = w; break
                    if not next_w: next_w = WINDOWS[0]
                    send_telegram(
                        f"✅ Tidak ada trade running.\n\n"
                        f"⏰ Window berikutnya:\n"
                        f"*{next_w['name']}*\n"
                        f"(UTC {next_w['utc_start']:02d}:00)"
                    )
            elif txt=="/windows":
                now_utc = datetime.now(timezone.utc).hour
                lines=["⏰ *Silver Bullet Windows (UTC):*\n"]
                for w in WINDOWS:
                    active = "🟢 AKTIF" if w["utc_start"]<=now_utc<w["utc_end"] else "⚪ Menunggu"
                    lines.append(f"{active} {w['name']}\n   UTC {w['utc_start']:02d}:00–{w['utc_end']:02d}:00")
                send_telegram("\n".join(lines))
    except: pass


# ═══════════════════════════════════════════════════════════════
# 9. MAIN LOOP
# ═══════════════════════════════════════════════════════════════
journal    = load_journal()
loop_count = 0
sent_cache = set()

def run_bot():
    global loop_count, journal

    log.info("🎯 Silver Bullet Bot mulai...")
    send_telegram(
        f"🎯 *ICT Silver Bullet Bot Aktif!*\n\n"
        f"*Pair      :* {SYMBOL}\n"
        f"*Entry TF  :* M5  |  *Bias TF:* H1\n\n"
        f"*3 Silver Bullet Windows:*\n"
        f"🌅 London Open  : 03:00–04:00 NY\n"
        f"🗽 NY Morning   : 10:00–11:00 NY\n"
        f"🌆 NY Afternoon : 14:00–15:00 NY\n\n"
        f"*Flow:*\n"
        f"Window aktif → Displacement M5\n"
        f"→ FVG terbentuk → Harga masuk FVG\n"
        f"→ Notif entry (1 trade at a time)\n\n"
        f"*Commands:*\n"
        f"/journal → statistik\n"
        f"/status  → trade running\n"
        f"/windows → cek window aktif\n\n"
        f"_Memantau 24 jam..._ 👁️"
    )

    while True:
        try:
            loop_count += 1
            if loop_count % 5   == 0: check_commands(journal)
            if loop_count % 360 == 0 and journal["stats"]["total"] > 0:
                send_telegram(get_stats_message(journal))

            # ── Ambil data
            df_m5 = get_candles(TF_M5, 100)
            df_h1 = get_candles(TF_H1, 60)

            if df_m5 is None or df_h1 is None:
                time.sleep(CHECK_INTERVAL); continue
            if len(df_m5) < 30 or len(df_h1) < 20:
                time.sleep(CHECK_INTERVAL); continue

            # ── Update hasil trade pending
            journal = update_results(journal, df_m5)

            # ── Blok jika ada trade running
            if has_running_trade(journal):
                log.info("Trade masih running — skip")
                time.sleep(CHECK_INTERVAL); continue

            # ── STEP 1: Cek apakah dalam Silver Bullet Window
            in_window, window_name = get_active_window()
            if not in_window:
                log.info("Di luar window Silver Bullet — standby")
                time.sleep(CHECK_INTERVAL); continue

            log.info(f"✅ Dalam window: {window_name}")

            # ── STEP 2: H1 Bias
            bias = get_h1_bias(df_h1)
            log.info(f"H1 Bias: {bias}")

            # ── STEP 3: Deteksi FVG + Displacement di M5
            signals = find_silver_bullet_fvg(df_m5, bias)

            if not signals:
                log.info("Tidak ada FVG dalam window")
                time.sleep(CHECK_INTERVAL); continue

            sig = signals[0]
            entry_price = df_m5["close"].iloc[-1]

            # Validasi RR
            if sig["risk"] <= 0 or sig["rr"] < MIN_RR:
                log.info(f"RR tidak valid — skip")
                time.sleep(CHECK_INTERVAL); continue

            # ── Hindari duplikat
            key = f"{sig['direction']}_{sig['fvg_lo']}_{sig['fvg_hi']}_{window_name}"
            if key in sent_cache:
                log.info("Sinyal sudah dikirim — skip")
                time.sleep(CHECK_INTERVAL); continue

            log.info(f"🎯 Signal: {sig['direction']} | FVG {sig['fvg_lo']}-{sig['fvg_hi']} | "
                     f"Window: {window_name}")

            # ── STEP 4: Kirim notif
            msg = format_signal(sig, bias, window_name, entry_price)
            send_telegram(msg)

            # ── STEP 5: Catat journal
            sig_id = f"{datetime.now().strftime('%Y%m%d%H%M')}_{sig['direction'][:4]}"
            log_to_journal(journal, sig_id, sig, entry_price, bias, window_name)
            save_journal(journal)

            sent_cache.add(key)
            if len(sent_cache) > 100: sent_cache.clear()

        except Exception as e:
            log.error(f"Error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run_bot()

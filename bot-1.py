"""
ICT MSS/BOS Bot - XAUUSD H1
Replika logic LuxAlgo ICT Concepts (MSS + BOS)
Notifikasi otomatis via Telegram
"""

import os
import time
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# ─────────────────────────────────────────
#  KONFIGURASI — isi sesuai kebutuhan
# ─────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("8919806833:AAHJZdzA0qwsky2862y062MJskK7kLmIG24")
TELEGRAM_CHAT_ID = os.getenv("6273206309")

SYMBOL      = "GC=F"        # XAUUSD di yfinance (Gold Futures)
TIMEFRAME   = "1h"          # H1
SWING_LEN   = 5             # pivot length (sama dgn Pine Script default)
CHECK_EVERY = 300           # cek tiap 5 menit (detik)

# ─────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers= [logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────
#  STATE — simpan kondisi market terakhir
# ─────────────────────────────────────────
state = {
    "mss_dir"       : 0,      # 1=bullish, -1=bearish, 0=neutral
    "last_signal"   : None,   # hindari notif duplikat
    "last_swing_high": None,
    "last_swing_low" : None,
}

# ─────────────────────────────────────────
#  FUNGSI TELEGRAM
# ─────────────────────────────────────────
def send_telegram(message: str):
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id"   : TELEGRAM_CHAT_ID,
        "text"      : message,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            log.info("✅ Notifikasi terkirim ke Telegram")
        else:
            log.warning(f"❌ Gagal kirim Telegram: {r.text}")
    except Exception as e:
        log.error(f"❌ Error kirim Telegram: {e}")

# ─────────────────────────────────────────
#  AMBIL DATA OHLC
# ─────────────────────────────────────────
def get_ohlc(symbol: str, interval: str, bars: int = 100) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period="7d", interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            log.warning("Data kosong dari yfinance")
            return pd.DataFrame()
        df = df.tail(bars).copy()
        df.columns = [c.lower() for c in df.columns]
        df.dropna(inplace=True)
        return df
    except Exception as e:
        log.error(f"Error ambil data: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────
#  DETEKSI SWING HIGH / LOW (Pivot)
#  Sama dengan ta.pivothigh/pivotlow Pine Script
# ─────────────────────────────────────────
def get_pivots(df: pd.DataFrame, length: int):
    """
    Return list of (index, price, direction)
    direction: 1 = swing high, -1 = swing low
    """
    pivots = []
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    for i in range(length, n - length):
        # Pivot High: candle[i] adalah tertinggi dalam window
        if all(highs[i] >= highs[i - j] for j in range(1, length + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, length + 1)):
            pivots.append((i, highs[i], 1))

        # Pivot Low: candle[i] adalah terendah dalam window
        if all(lows[i] <= lows[i - j] for j in range(1, length + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, length + 1)):
            pivots.append((i, lows[i], -1))

    # Urutkan berdasarkan index
    pivots.sort(key=lambda x: x[0])
    return pivots

# ─────────────────────────────────────────
#  BUILD ZIGZAG dari pivot list
#  Replika logic aZZ di Pine Script
# ─────────────────────────────────────────
def build_zigzag(pivots: list) -> list:
    """
    Bersihkan pivot agar alternating: H-L-H-L atau L-H-L-H
    Return list of (idx, price, direction)
    """
    if not pivots:
        return []

    zz = []
    for p in pivots:
        if not zz:
            zz.append(p)
            continue
        last = zz[-1]
        if p[2] == last[2]:
            # Arah sama → ambil yang lebih ekstrem
            if p[2] == 1 and p[1] > last[1]:
                zz[-1] = p
            elif p[2] == -1 and p[1] < last[1]:
                zz[-1] = p
        else:
            zz.append(p)
    return zz

# ─────────────────────────────────────────
#  DETEKSI MSS & BOS
#  Replika logic switch MSS di Pine Script
# ─────────────────────────────────────────
def detect_mss_bos(df: pd.DataFrame, zz: list, mss_dir: int):
    """
    Cek candle terakhir apakah ada MSS atau BOS.
    Return: (signal_type, price_level) atau (None, None)

    signal_type: 'MSS_BULL', 'MSS_BEAR', 'BOS_BULL', 'BOS_BEAR'
    """
    if len(zz) < 3:
        return None, None, mss_dir

    close = df["close"].iloc[-1]

    # Ambil 3 titik zigzag terakhir
    # iH = index swing high di zigzag (Pine: aZZ.d.get(2)==1 ? 2 : 1)
    # iL = index swing low  di zigzag
    last3 = zz[-3:]

    # Cari swing high & low terbaru dari 3 titik terakhir
    swing_highs = [(i, p, d) for i, p, d in last3 if d == 1]
    swing_lows  = [(i, p, d) for i, p, d in last3 if d == -1]

    if not swing_highs or not swing_lows:
        return None, None, mss_dir

    recent_high = swing_highs[-1][1]
    recent_low  = swing_lows[-1][1]

    signal     = None
    price_level= None

    # ── MSS Bullish ──
    # close > swing high AND sebelumnya bearish/neutral
    if close > recent_high and mss_dir < 1:
        signal      = "MSS_BULL"
        price_level = recent_high
        mss_dir     = 1

    # ── MSS Bearish ──
    # close < swing low AND sebelumnya bullish/neutral
    elif close < recent_low and mss_dir > -1:
        signal      = "MSS_BEAR"
        price_level = recent_low
        mss_dir     = -1

    # ── BOS Bullish ──
    # Sudah MSS bullish, close break swing high baru lagi
    elif mss_dir == 1 and close > recent_high:
        signal      = "BOS_BULL"
        price_level = recent_high

    # ── BOS Bearish ──
    # Sudah MSS bearish, close break swing low baru lagi
    elif mss_dir == -1 and close < recent_low:
        signal      = "BOS_BEAR"
        price_level = recent_low

    return signal, price_level, mss_dir

# ─────────────────────────────────────────
#  FORMAT PESAN TELEGRAM
# ─────────────────────────────────────────
def format_message(signal: str, price: float, close: float, df: pd.DataFrame) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    emoji_map = {
        "MSS_BULL": "🟢",
        "MSS_BEAR": "🔴",
        "BOS_BULL": "🔵",
        "BOS_BEAR": "🟠",
    }
    label_map = {
        "MSS_BULL": "MSS Bullish — Market Structure Shift NAIK",
        "MSS_BEAR": "MSS Bearish — Market Structure Shift TURUN",
        "BOS_BULL": "BOS Bullish — Break of Structure NAIK",
        "BOS_BEAR": "BOS Bearish — Break of Structure TURUN",
    }
    bias_map = {
        "MSS_BULL": "📈 BULLISH",
        "MSS_BEAR": "📉 BEARISH",
        "BOS_BULL": "📈 BULLISH (lanjutan)",
        "BOS_BEAR": "📉 BEARISH (lanjutan)",
    }

    high_1h = df["high"].iloc[-1]
    low_1h  = df["low"].iloc[-1]

    msg = (
        f"{emoji_map[signal]} <b>ICT SIGNAL — XAUUSD H1</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>{label_map[signal]}</b>\n\n"
        f"💰 Close      : <b>${close:,.2f}</b>\n"
        f"🎯 Level Break: <b>${price:,.2f}</b>\n"
        f"📊 H1 High    : ${high_1h:,.2f}\n"
        f"📊 H1 Low     : ${low_1h:,.2f}\n\n"
        f"⚡ Bias       : {bias_map[signal]}\n"
        f"⏰ Waktu      : {now}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>⚠️ Bukan financial advice. DYOR.</i>"
    )
    return msg

# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────
def run():
    log.info("🚀 ICT MSS/BOS Bot dimulai — XAUUSD H1")
    send_telegram(
        "🤖 <b>ICT Bot XAUUSD H1 Aktif!</b>\n"
        "Memantau MSS & BOS secara otomatis.\n"
        "Notifikasi akan dikirim saat sinyal terdeteksi."
    )

    while True:
        try:
            log.info("🔍 Mengambil data XAUUSD H1...")
            df = get_ohlc(SYMBOL, TIMEFRAME, bars=150)

            if df.empty or len(df) < 20:
                log.warning("Data tidak cukup, skip...")
                time.sleep(CHECK_EVERY)
                continue

            close = df["close"].iloc[-1]
            log.info(f"Close saat ini: ${close:,.2f}")

            # Deteksi pivot & zigzag
            pivots = get_pivots(df, SWING_LEN)
            zz     = build_zigzag(pivots)

            log.info(f"Jumlah zigzag points: {len(zz)}")

            # Deteksi MSS / BOS
            signal, price_level, new_dir = detect_mss_bos(
                df, zz, state["mss_dir"]
            )

            # Update state
            state["mss_dir"] = new_dir

            if signal:
                # Hindari kirim sinyal yang sama berulang
                sig_key = f"{signal}_{price_level:.2f}"
                if sig_key != state["last_signal"]:
                    state["last_signal"] = sig_key
                    msg = format_message(signal, price_level, close, df)
                    log.info(f"📣 SINYAL: {signal} @ {price_level:.2f}")
                    send_telegram(msg)
                else:
                    log.info(f"Sinyal sama ({signal}), skip duplikat")
            else:
                log.info("Tidak ada sinyal baru")

        except Exception as e:
            log.error(f"Error di main loop: {e}")

        log.info(f"⏳ Tunggu {CHECK_EVERY // 60} menit...\n")
        time.sleep(CHECK_EVERY)

# ─────────────────────────────────────────
if __name__ == "__main__":
    run()

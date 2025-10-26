import os
import time
import json
import logging
import requests
import pandas as pd
import ccxt
from datetime import datetime
from dotenv import load_dotenv

# === CUSTOM MODULES ===
from ProtectedHighLowDetector import ProtectedHighLowDetector
from OrderBlockDetector import OrderBlockDetector
from FvgDetector import FVGDetector
from PoiDetector import POIDetector
from NotifierTelegram import NotifierTelegram
from AdvancedTrendDetector import AdvancedTrendDetector  # 🧩 Added Trend Detector

# ===================== LOAD ENVIRONMENT VARIABLES =====================
load_dotenv()

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_IDS = os.getenv("TELEGRAM_CHAT_IDS", "").split(",")
Telegram_proxy_address = None

# --- Exchange & Proxy ---
USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
PROXY_ADDR = os.getenv("PROXY_ADDR")

# --- Trading Config ---
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL", 60))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))
PIVOT_LEFT = int(os.getenv("PIVOT_LEFT", 3))
PIVOT_RIGHT = int(os.getenv("PIVOT_RIGHT", 3))
MIN_BREAK_MULT = float(os.getenv("MIN_BREAK_MULT", 0))
FILTER_SIGNALS = os.getenv("FILTER_SIGNALS", "true").lower() == "true"
AUTO_TRADE = os.getenv("AUTO_TRADE", "false").lower() == "true"
STRICT_LEVEL = os.getenv("STRICT_LEVEL", "medium")
RR_RATIO = float(os.getenv("RR_RATIO", 1))
ENTRY_MODE = os.getenv("ENTRY_MODE", "CURRENT_PRICE")
PLACE_TRADE_IN_THREAD = os.getenv("PLACE_TRADE_IN_THREAD", "false").lower() == "true"
ORDER_VALUE_USDT = float(os.getenv("ORDER_VALUE_USDT", 20.0))
AUTO_SET_ORDER_VALUE = os.getenv("AUTO_SET_ORDER_VALUE", "false").lower() == "true"
USE_ORDER_BLOCK = os.getenv("USE_ORDER_BLOCK", "true").lower() == "true"

# --- Trend Filter ---
USE_TREND_FILTER = os.getenv("USE_TREND_FILTER", "false").lower() == "true"
TREND_TIMEFRAME = os.getenv("TREND_TIMEFRAME", "1h")

# --- Symbol List ---
SYMBOLS = [
    s.strip().upper()
    for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
    if s.strip()  # ensures no empty symbols
]

# ===================== EXCHANGE SETUP =====================
if USE_PROXY:
    Telegram_proxy_address = PROXY_ADDR
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": PROXY_ADDR, "https": PROXY_ADDR}
    EXCHANGE = ccxt.binance({
        "session": session,
        "enableRateLimit": True,
    })
else:
    EXCHANGE = ccxt.binance({"enableRateLimit": True})

# ===================== LOGGER SETUP =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ===================== TELEGRAM NOTIFIER =====================
notifier = NotifierTelegram(
    bot_token=TELEGRAM_BOT_TOKEN,
    chat_ids=TELEGRAM_CHAT_IDS,
    atr_len=ATR_PERIOD,
    symbol_precision=2,
    request_timeout=30,
    proxy=Telegram_proxy_address,
)

# ===================== TREND DETECTOR =====================
trend_detector = AdvancedTrendDetector(ema_length=50, adx_length=14, adx_threshold=20)

# ===================== FETCH OHLCV =====================
def fetch_ohlcv(symbol, timeframe, limit=200):
    """Fetch recent OHLCV data"""
    print(f"📊 Fetching {limit} candles for {symbol} ({timeframe}) ...")
    data = EXCHANGE.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

# ===================== ANALYSIS =====================
import hashlib

def analyze_symbol(symbol, last_sent_poi):
    try:
        df = fetch_ohlcv(symbol, TIMEFRAME)

        phl = ProtectedHighLowDetector(
            left=PIVOT_LEFT,
            right=PIVOT_RIGHT,
            min_break_atr_mult=MIN_BREAK_MULT,
            atr_len=3
        )
        ob = OrderBlockDetector(
            swing_length=10,
            max_atr_mult=3.5,
            atr_len=3,
            ob_end_method="Wick",
            combine_obs=True,
            max_orderblocks=30,
            max_boxes_count=500,
            overlap_threshold_percentage=0.0,
        )
        fvg = FVGDetector(fvg_history_limit=10, reduce_mitigated=True)

        poi_detector = POIDetector(
            config={"swing_proximity": 3},
            protected_detector=phl,
            ob_detector=ob,
            fvg_detector=fvg,
            symbol=symbol,
            timeframe=TIMEFRAME,
        )

        result = poi_detector.find_pois(df)
        pois = result.get("pois", [])

        if not pois:
            logging.info(f"{symbol},NO_POI,No POIs detected")
            return

        latest_poi = pois[-1]
        ob_top, ob_bottom = latest_poi["ob_top"], latest_poi["ob_bottom"]
        fvg_top, fvg_bottom = latest_poi["fvg_top"], latest_poi["fvg_bottom"]
        protected_price = latest_poi["protected_price"]
        current_price = df["close"].iloc[-1]

        # === Degenerate POI Filter ===
        if (
            ob_top is None or ob_bottom is None or
            abs(ob_top - ob_bottom) < 1e-6 or
            abs(fvg_top - fvg_bottom) < 1e-6 or
            abs(protected_price - current_price) < 1e-8
        ):
            logging.info(f"{symbol},POI_SKIPPED,Degenerate POI values detected (flat structure)")
            return

        # === Debug: Identical Prices Warning ===
        if len({protected_price, ob_top, ob_bottom, fvg_top, fvg_bottom, current_price}) == 1:
            logging.warning(f"{symbol},WARNING,All prices identical -> possible detection bug")

        # === ATR Validation ===
        atr_val = notifier._calculate_atr(df)
        if (
            atr_val is None or pd.isna(atr_val) or
            atr_val == 0 or atr_val < 1e-6 or
            atr_val / current_price < 0.00005
        ):
            logging.info(f"{symbol},POI_SKIPPED,ATR too small or invalid (ATR={atr_val})")
            return

        # === Mislabel Correction ===
        if "Low" in latest_poi["protected_type"] and protected_price > current_price:
            logging.warning(f"{symbol},CORRECTION,Protected Low above price -> relabeled as High")
            latest_poi["protected_type"] = "Protected High"
        elif "High" in latest_poi["protected_type"] and protected_price < current_price:
            logging.warning(f"{symbol},CORRECTION,Protected High below price -> relabeled as Low")
            latest_poi["protected_type"] = "Protected Low"

        # === Generate Strong Unique POI Signature ===
        poi_signature_str = (
            f"{symbol}_{TIMEFRAME}_"
            f"{latest_poi['protected_type']}_"
            f"{round(protected_price, 4)}_"
            f"{round(ob_top or 0, 4)}_{round(ob_bottom or 0, 4)}_"
            f"{round(fvg_top or 0, 4)}_{round(fvg_bottom or 0, 4)}"
        )
        poi_hash = hashlib.sha256(poi_signature_str.encode()).hexdigest()

        # === Check for Duplicate Signal ===
        if last_sent_poi.get(symbol) == poi_hash:
            logging.info(f"{symbol},POI_SKIPPED,Duplicate POI signature — not sent")
            return

        # === Build Telegram Message (only if valid ATR) ===
        msg = notifier._build_message(latest_poi, df, timeframe=TIMEFRAME)
        if not msg or "ATR(14):</b> <i>0.00" in msg:
            logging.info(f"{symbol},POI_SKIPPED,Invalid or zero ATR in message — not sent")
            return

        # === Send ===
        notifier.send_poi(latest_poi, df)
        logging.info(f"{symbol},POI,{json.dumps(latest_poi)}")
        last_sent_poi[symbol] = poi_hash

    except Exception as e:
        logging.error(f"{symbol},ERROR,{e}")

# ===================== MAIN LOOP =====================
def main_loop():
    print(f"🔍 Monitoring {len(SYMBOLS)} pairs on {TIMEFRAME} timeframe\n")
    if USE_TREND_FILTER:
        print(f"📈 Trend Filter Enabled ({TREND_TIMEFRAME})\n")

    last_sent_poi = {symbol: None for symbol in SYMBOLS}

    while True:
        for symbol in SYMBOLS:
            analyze_symbol(symbol, last_sent_poi)

        print(f"\n🕒 Cycle complete. Waiting {LOOP_INTERVAL}s before next scan...", end="", flush=True)
        for sec in range(LOOP_INTERVAL, 0, -1):
            print(f"\r🕒 Next cycle in {sec:02d}s...", end="", flush=True)
            time.sleep(1)
        print("\r🕒 Starting next cycle...                           ", flush=True)

# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n\n🛑 Script stopped by user.")

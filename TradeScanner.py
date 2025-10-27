#!/usr/bin/env python3
import os
import time
import json
import logging
import requests
import pandas as pd
import ccxt
import hashlib
from datetime import datetime, timezone
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# === CUSTOM MODULES (same as before) ===
from ProtectedHighLowDetector import ProtectedHighLowDetector
from OrderBlockDetector import OrderBlockDetector
from FvgDetector import FVGDetector
from PoiDetector import POIDetector
from NotifierTelegram import NotifierTelegram
from AdvancedTrendDetector import AdvancedTrendDetector

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

HEARTBEAT_INTERVAL = 60  # seconds

PERSIST_FILE = "last_sent_poi.json"
POI_RETENTION_DAYS = 3  # auto-clean entries older than 3 days



def load_last_sent_poi():
    """Load persistent duplicate POI hashes from disk."""
    try:
        if os.path.exists(PERSIST_FILE):
            with open(PERSIST_FILE, "r") as f:
                data = json.load(f)

            # If old format (string hash only), upgrade to dict
            upgraded = {}
            for sym, val in data.items():
                if isinstance(val, str):
                    upgraded[sym] = {"hash": val, "timestamp": time.time()}
                else:
                    upgraded[sym] = val
            data = upgraded

            logging.info(f"Loaded persistent POI hashes for {len(data)} symbols.")
            return data
    except Exception as e:
        logging.warning(f"Failed to load {PERSIST_FILE}: {e}")
    return {}

def save_last_sent_poi():
    """Write last_sent_poi to disk safely with auto-cleaning."""
    try:
        cutoff = time.time() - (POI_RETENTION_DAYS * 86400)
        with state_lock:
            # remove entries older than retention
            cleaned = {
                s: d for s, d in last_sent_poi.items()
                if d and d.get("timestamp", 0) >= cutoff
            }
            if len(cleaned) < len(last_sent_poi):
                logging.info(f"Cleaned {len(last_sent_poi) - len(cleaned)} old POI entries.")
            last_sent_poi.clear()
            last_sent_poi.update(cleaned)

            with open(PERSIST_FILE, "w") as f:
                json.dump(last_sent_poi, f)
    except Exception as e:
        logging.error(f"Failed to save {PERSIST_FILE}: {e}")


# --- Symbol List ---
SYMBOLS = [
    s.strip().upper()
    for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
    if s.strip()
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
    format="%(asctime)s,%(levelname)s,%(message)s",
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

# ===================== TREND DETECTOR (single instance) =====================
trend_detector = AdvancedTrendDetector(ema_length=50, adx_length=14, adx_threshold=20)

# ================ UTIL: timeframe -> milliseconds =================
def timeframe_to_ms(tf: str) -> int:
    """
    Convert timeframe strings like '1m', '15m', '1h', '4h', '1d' to milliseconds.
    If unknown format, default to 1 minute.
    """
    tf = tf.strip().lower()
    try:
        if tf.endswith("m"):
            return int(tf[:-1]) * 60_000
        if tf.endswith("h"):
            return int(tf[:-1]) * 60 * 60_000
        if tf.endswith("d"):
            return int(tf[:-1]) * 24 * 60 * 60_000
    except Exception:
        pass
    # fallback: 1 minute
    return 60_000

TIMEFRAME_MS = timeframe_to_ms(TIMEFRAME)

# ===================== DETECTORS: instantiate once per symbol =====================
def create_detectors_for_symbol(symbol: str):
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

    poi = POIDetector(
        config={"swing_proximity": 3},
        protected_detector=phl,
        ob_detector=ob,
        fvg_detector=fvg,
        symbol=symbol,
        timeframe=TIMEFRAME,
    )

    return {"phl": phl, "ob": ob, "fvg": fvg, "poi": poi}

detectors = {s: create_detectors_for_symbol(s) for s in SYMBOLS}

# ===================== CACHES & STATE =====================
ohlcv_cache = {s: pd.DataFrame() for s in SYMBOLS}  # cached DataFrame per symbol
last_candle_ts = {s: None for s in SYMBOLS}        # last candle timestamp (pd.Timestamp)

# last_sent_poi = {s: None for s in SYMBOLS}         # last sent poi hash per symbol
last_sent_poi = load_last_sent_poi()
for s in SYMBOLS:
    last_sent_poi.setdefault(s, None)

# concurrency lock for shared state updates
state_lock = Lock()

# ===================== FETCH OHLCV (incremental) =====================
def fetch_ohlcv_incremental(symbol: str, timeframe: str, since: int = None, limit: int = 200):
    """
    Fetch only new candles since `since` (ms). If since is None, fetch `limit` candles.
    Returns list of ohlcv rows as returned by ccxt.
    """
    # ccxt expects since in ms or None
    params = {}
    try:
        if since:
            data = EXCHANGE.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        else:
            data = EXCHANGE.fetch_ohlcv(symbol, timeframe, limit=limit)
        return data
    except Exception as e:
        logging.error(f"{symbol},FETCH_ERROR,{e}")
        return []

def update_cache_for_symbol(symbol: str):
    """
    Fetch recent candles and update the cached DataFrame.
    Returns tuple (df, new_candle_added: bool)
    """
    global ohlcv_cache, last_candle_ts

    prev_last_ts = last_candle_ts.get(symbol)
    since_ms = None
    if prev_last_ts is not None:
        # fetch after last stored timestamp (convert to ms + 1 to avoid duplication)
        since_ms = int(prev_last_ts.timestamp() * 1000) + 1

    raw = fetch_ohlcv_incremental(symbol, TIMEFRAME, since=since_ms, limit=200)

    if not raw:
        # no new data
        return ohlcv_cache[symbol], False

    df_new = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df_new["ts"] = pd.to_datetime(df_new["ts"], unit="ms", utc=True)

    # combine with existing cache
    if ohlcv_cache[symbol].empty:
        df_combined = df_new
    else:
        df_combined = pd.concat([ohlcv_cache[symbol], df_new], ignore_index=True)
        # drop duplicates by timestamp and keep last
        df_combined = df_combined.drop_duplicates(subset=["ts"], keep="last")

    # keep a reasonable window — detectors likely don't need >500 candles
    df_combined = df_combined.sort_values("ts").tail(500).reset_index(drop=True)
    ohlcv_cache[symbol] = df_combined

    new_last_ts = df_combined["ts"].iloc[-1]
    added = (prev_last_ts is None) or (new_last_ts > prev_last_ts)

    with state_lock:
        last_candle_ts[symbol] = new_last_ts

    return df_combined, added

# ===================== ANALYSIS per SYMBOL =====================
def analyze_symbol(symbol: str):
    try:
        df, added = update_cache_for_symbol(symbol)
        if df.empty:
            logging.info(f"{symbol},NO_DATA,No OHLCV cached")
            return

        # Skip if no new candle since last analysis
        if not added:
            logging.debug(f"{symbol},SKIP,No new candle")
            return

        # --- Run POI detection ---
        poi_detector = detectors[symbol]["poi"]
        df_tail = df.tail(300).copy()
        result = poi_detector.find_pois(df_tail)
        pois = result.get("pois", [])
        if not pois:
            logging.info(f"{symbol},NO_POI,No POIs detected")
            return

        latest_poi = pois[-1]
        ob_top = latest_poi.get("ob_top")
        ob_bottom = latest_poi.get("ob_bottom")
        protected_price = latest_poi.get("protected_price")
        protected_type = latest_poi.get("protected_type", "")
        order_block_type = latest_poi.get("order_block_type", "")  # Bullish / Bearish expected
        current_price = float(df_tail["close"].iloc[-1])

        # --- Quick degeneracy filters ---
        if (
            ob_top is None or ob_bottom is None or
            abs((ob_top or 0) - (ob_bottom or 0)) < 1e-8 or
            protected_price is None or abs(protected_price - current_price) < 1e-8
        ):
            logging.info(f"{symbol},POI_SKIPPED,Degenerate POI values")
            return

        # --- ATR sanity check ---
        atr_val = notifier._calculate_atr(df_tail)
        if (
            atr_val is None or pd.isna(atr_val) or
            atr_val == 0 or atr_val < 1e-8 or
            atr_val / current_price < 0.00005
        ):
            logging.info(f"{symbol},POI_SKIPPED,ATR invalid or too small ({atr_val})")
            return

        # --- Direction and label correction ---
        if "Low" in protected_type and protected_price > current_price:
            latest_poi["protected_type"] = "Protected High"
        elif "High" in protected_type and protected_price < current_price:
            latest_poi["protected_type"] = "Protected Low"

        # Derive signal side (BUY for Protected Low, SELL for Protected High)
        side = "BUY" if "Low" in latest_poi["protected_type"] else "SELL"

        # --- Build clean signature (only the consistent fields you requested) ---
        ob_top_r = round(ob_top or 0, 3)
        ob_bottom_r = round(ob_bottom or 0, 3)
        key_protected_type = latest_poi["protected_type"]
        key_ob_type = order_block_type or ("Bullish" if side == "BUY" else "Bearish")

        poi_signature_str = (
            f"{symbol}_{side}_{key_protected_type}_"
            f"{key_ob_type}_"
            f"OB{ob_top_r}-{ob_bottom_r}"
        )
        poi_hash = hashlib.sha256(poi_signature_str.encode()).hexdigest()

        # --- Check for duplicate ---
        with state_lock:
            prev_entry = last_sent_poi.get(symbol)
            prev_hash = prev_entry.get("hash") if isinstance(prev_entry, dict) else prev_entry
            if prev_hash == poi_hash:
                logging.info(f"{symbol},POI_SKIPPED,Duplicate POI (same structure)")
                return

        # --- Build and validate message ---
        msg = notifier._build_message(latest_poi, df_tail, timeframe=TIMEFRAME)
        if not msg or f"ATR({ATR_PERIOD}):</b> <i>0.00" in msg:
            logging.info(f"{symbol},POI_SKIPPED,Invalid or zero ATR in message")
            return

        # --- Send notification ---
        notifier.send_poi(latest_poi, df_tail)
        logging.info(f"{symbol},POI_SENT,{json.dumps(latest_poi)}")

        # --- Save persistent hash ---
        with state_lock:
            last_sent_poi[symbol] = {"hash": poi_hash, "timestamp": time.time()}

    except Exception as e:
        logging.error(f"{symbol},ERROR,{e}")




# ===================== MAIN LOOP =====================
def main_loop():
    logging.info(f"Starting monitor for {len(SYMBOLS)} symbols on {TIMEFRAME}")
    if USE_TREND_FILTER:
        logging.info(f"Trend filter enabled ({TREND_TIMEFRAME})")

    # warm caches once at start but keep lightweight (fetch last small window)
    logging.info("Priming OHLCV cache...")
    counter = 0
    for s in SYMBOLS:
        counter += 1
        try:
            print(f"\r[{counter}/{len(SYMBOLS)}]Priming {s}...", end="")
            df_init = EXCHANGE.fetch_ohlcv(s, TIMEFRAME, limit=200)
            if df_init:
                df_init = pd.DataFrame(df_init, columns=["ts", "open", "high", "low", "close", "volume"])
                df_init["ts"] = pd.to_datetime(df_init["ts"], unit="ms", utc=True)
                ohlcv_cache[s] = df_init.sort_values("ts").tail(500).reset_index(drop=True)
                last_candle_ts[s] = ohlcv_cache[s]["ts"].iloc[-1]
        except Exception as e:
            logging.warning(f"{s},INIT_FETCH_FAILED,{e}")

    max_workers = min(len(SYMBOLS), 6) or 1  # limit concurrency to a small number to reduce rate pressure

    last_heartbeat = time.time()
    while True:
        start_time = time.time()

        # Use ThreadPoolExecutor to parallelize I/O-bound symbol updates/analysis
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(analyze_symbol, SYMBOLS)

        # Save POI state once per cycle (persistent duplicate protection)
        save_last_sent_poi()

        # --- Compute smart sleep: wake near next candle close but at most LOOP_INTERVAL ---
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        time_to_next_candle_ms = TIMEFRAME_MS - (now_ms % TIMEFRAME_MS)
        sleep_seconds = min(LOOP_INTERVAL, max(1, time_to_next_candle_ms / 1000.0))

        elapsed = time.time() - start_time
        # ensure we don't oversleep if processing already took longer
        sleep_seconds = max(0, sleep_seconds - elapsed)

        logging.debug(f"Cycle done. Sleeping {sleep_seconds:.1f}s")

        # --- Heartbeat while sleeping ---
        sleep_start = time.time()
        while time.time() - sleep_start < sleep_seconds:
            # check if it's time to send heartbeat
            if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL:
                logging.info("💓 Heartbeat: bot alive and waiting for next candle...")
                last_heartbeat = time.time()
            time.sleep(1)  # check heartbeat every 1 second

# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("Script stopped by user.")

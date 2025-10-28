import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
from zoneinfo import ZoneInfo  # ✅ add this import at the top of the file

import requests
import pandas as pd
import numpy as np
import time


class NotifierTelegram:
    """
    Telegram notifier for POI alerts.
    Sends formatted messages to multiple chat IDs with ATR-based SL/TP logic.
    """

    def __init__(
        self,
        bot_token: str,
        chat_ids: list,
        atr_len: int = 14,
        symbol_precision: int = 2,
        proxy: str = None,
        request_timeout: int = 30,
        max_retries: int = 3,
        project_tag: str = "@jbadonaiventures V4"
    ):
        self.bot_token = bot_token
        self.chat_ids = chat_ids if isinstance(chat_ids, list) else [chat_ids]
        self.atr_len = atr_len
        self.symbol_precision = symbol_precision
        self.proxy = proxy
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.project_tag = project_tag

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate ATR using standard formula."""
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = np.maximum.reduce([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ])
        atr_series = pd.Series(tr).rolling(self.atr_len).mean()
        return float(atr_series.iloc[-1])

    def _fmt(self, value: float) -> str:
        if pd.isna(value):
            return "-"
        return f"{value:.{self.symbol_precision}f}"

    # -------------------------------------------------------------------------
    # Message building
    # -------------------------------------------------------------------------



    def _build_message(self, poi: dict, df: pd.DataFrame, timeframe: str = "15m") -> str:
        """Build formatted Telegram message similar to the user’s preferred format."""
        atr_val = self._calculate_atr(df)

        protected_type = poi["protected_type"]
        is_bullish = "Low" in protected_type
        signal_type = "🟢 LONG" if is_bullish else "🔴 SHORT"
        trade_type = "BUY" if is_bullish else "SELL"

        protected_price = poi["protected_price"]
        ob_top, ob_bottom = poi["ob_top"], poi["ob_bottom"]
        fvg_top, fvg_bottom = poi["fvg_top"], poi["fvg_bottom"]

        # Trade setup
        if is_bullish:
            limit_entry = fvg_top if fvg_top is not None else ""
            stop_loss = protected_price - atr_val
        else:
            limit_entry = fvg_bottom if fvg_bottom is not None else ""
            stop_loss = protected_price + atr_val

        rr_distance = abs(limit_entry - stop_loss)
        take_profit = limit_entry + rr_distance if is_bullish else limit_entry - rr_distance

        current_price = df["close"].iloc[-1]
        detected_price = protected_price

        # ✅ Convert UTC timestamp to Nigeria Time (WAT)
        detected_time = poi.get("timestamp", "N/A")
        try:
            if detected_time != "N/A":
                utc_dt = pd.to_datetime(detected_time, utc=True)
                wat_dt = utc_dt.astimezone(ZoneInfo("Africa/Lagos"))
                detected_time = wat_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception as tz_err:
            detected_time = f"{detected_time} (tz_conv_fail: {tz_err})"

        msg = (
            f"<pre><b>📢 NEW SIGNAL DETECTED - {timeframe.upper()}</b></pre>\n"
            f"<b>PAIR:</b> <i>🪙 {poi['symbol']} 🪙 {poi['symbol']}.P </i>\n"
            f"<b>{'Bullish' if is_bullish else 'Bearish'} OB + {protected_type}:</b> <i>@{self._fmt(detected_price)}</i>\n"
            f"ORDER BLOCK: <b>{'Bullish' if is_bullish else 'Bearish'}</b> @ {self._fmt(ob_top)} / {self._fmt(ob_bottom)}\n\n"
            f"<pre><b>Trade Setup</b></pre>\n"
            f"<b>SIGNAL:</b> <i>{trade_type}</i>\n"
            f"<b>MARKET ENTRY:</b> <i>{self._fmt(current_price)}</i>\n"
            f"<b>LIMIT ENTRY:</b> <i>{self._fmt(limit_entry)}</i>\n"
            f"<b>STOP LOSS:</b> <i>{self._fmt(stop_loss)}</i>\n"
            f"<b>TAKE PROFIT (1:1):</b> <i>{self._fmt(take_profit)}</i>\n"
            f"<b>ATR({self.atr_len}):</b> <i>{self._fmt(atr_val)}</i>\n\n"
            f"lines: {self._fmt(detected_price)}, {self._fmt(current_price)}, {self._fmt(stop_loss)}, {self._fmt(take_profit)}\n"
            f"ob box: {self._fmt(ob_top)}, {self._fmt(ob_bottom)}\n\n"
            f"<b>Timestamp:</b> <i>{detected_time}</i>\n\n"
            f"<i>{self.project_tag}</i>"
        )

        return msg

    # -------------------------------------------------------------------------
    # Telegram send logic
    # -------------------------------------------------------------------------

    def _send_to_telegram(self, text: str):
        """Sends a formatted message to all chat IDs."""
        url = f"{self.base_url}/sendMessage"
        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None

        for chat_id in self.chat_ids:
            payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = requests.post(
                        url,
                        data=payload,  # ✅ form-encoded (more compatible)
                        proxies=proxies,
                        timeout=self.request_timeout
                    )
                    if resp.status_code == 200:
                        print(f"✅ Telegram message sent successfully to chat {chat_id}")
                        break
                    else:
                        print(f"⚠️ Telegram error for chat {chat_id}: {resp.status_code} {resp.text}")
                except Exception as e:
                    print(f"❌ Telegram exception for chat {chat_id}: {e}")

                if attempt < self.max_retries:
                    sleep_time = 2 ** attempt
                    print(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"❌ Telegram failed after {self.max_retries} attempts for chat {chat_id}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def send_poi(self, poi: dict, df: pd.DataFrame, timeframe: str = "15m"):
        """Formats and sends the POI signal message."""
        if not poi:
            print("⚠️ No POI data to send.")
            return

        msg = self._build_message(poi, df, timeframe)
        # print(" MESSAGE !!!!!!!!!!!!!!!")
        # print(msg)
        self._send_to_telegram(msg)
        print("📨 Telegram notification sent successfully.")

class NotifierTelegramOld:
    def __init__(
            self,
            bot_token: str,
            chat_ids,
            atr_len: int = 14,
            symbol_precision: int = 2,
            request_timeout: int = 10
    ):
        """
        Telegram notifier for POI events.

        :param bot_token: Telegram bot token.
        :param chat_ids: Single chat_id (str/int) or list of chat_ids.
        :param atr_len: ATR length for stop-loss calculation.
        :param symbol_precision: Decimal precision for symbol prices.
        :param request_timeout: Timeout for Telegram API requests.
        """
        self.bot_token = bot_token
        # Accept both single and multiple IDs
        if isinstance(chat_ids, (str, int)):
            self.chat_ids = [chat_ids]
        elif isinstance(chat_ids, list):
            self.chat_ids = chat_ids
        else:
            raise TypeError("chat_ids must be str, int, or list of str/int")

        self.atr_len = atr_len
        self.symbol_precision = symbol_precision
        self.request_timeout = request_timeout
        self.telegram_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    # ----------------------------------------------------------------------
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate the latest ATR value using pandas operations."""
        if len(df) < self.atr_len + 1:
            return 0.0

        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = pd.concat([
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=self.atr_len).mean().iloc[-1]
        return float(atr)

    # ----------------------------------------------------------------------
    def _format_price(self, value: float) -> str:
        """Format price according to symbol tick precision."""
        return f"{value:.{self.symbol_precision}f}"

    # ----------------------------------------------------------------------
    def _build_message_old(self, poi: dict, df: pd.DataFrame) -> str:
        """Build the Telegram message body for a detected POI."""
        atr_val = self._calculate_atr(df)
        direction = poi["protected_type"].lower()

        # --- directional SL/TP logic ---
        if direction == "high":
            sl = poi["protected_price"] + atr_val
            limit_entry = poi["fvg_bottom"]
            rr_dist = abs(sl - limit_entry)
            tp = limit_entry - rr_dist
            type_label = "Protected HIGH + Bearish OB"
        else:
            sl = poi["protected_price"] - atr_val
            limit_entry = poi["fvg_top"]
            rr_dist = abs(sl - limit_entry)
            tp = limit_entry + rr_dist
            type_label = "Protected LOW + Bullish OB"

        # Format all numbers
        fmt = self._format_price
        symbol = poi.get("symbol", "UNKNOWN")
        timestamp = poi.get("timestamp", datetime.now(timezone.utc).isoformat())
        note = poi.get("note", "POI automatically detected.")

        # --- Construct message ---
        msg = (
            f"POI FOUND — [SYMBOL: {symbol}]\n"
            f"Type: {type_label}\n"
            f"Protected Price: {fmt(poi['protected_price'])}\n"
            f"Order Block: Top = {fmt(poi['ob_top'])} | Bottom = {fmt(poi['ob_bottom'])}\n"
            f"FVG: Top = {fmt(poi['fvg_top'])} | Bottom = {fmt(poi['fvg_bottom'])}\n"
            f"Limit Entry: {fmt(limit_entry)}\n"
            f"Stop Loss: {fmt(sl)} (ATR({self.atr_len}) = {fmt(atr_val)})\n"
            f"Take Profit (1:1): {fmt(tp)}\n"
            f"Timestamp: {timestamp}\n"
            f"Notes: {note}"
        )
        return msg

    # ----------------------------------------------------------------------
    def _send_to_single_chat(self, chat_id, message: str) -> bool:
        """Send message to a single Telegram chat ID."""
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(
                self.telegram_url,
                data=payload,
                timeout=self.request_timeout
            )
            if response.status_code == 200:
                print(f"✅ Sent POI to chat {chat_id}")
                return True
            else:
                print(f"⚠️ Failed to send POI to chat {chat_id} — [{response.status_code}]: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Telegram error for chat {chat_id}: {e}")
            return False

    # ----------------------------------------------------------------------
    def send_poi(self, poi: dict, df: pd.DataFrame) -> bool:
        """
        Send formatted POI message to one or multiple Telegram recipients.

        :param poi: POI dictionary from POIDetector
        :param df: full OHLCV dataframe (used for ATR)
        :return: True if all sends succeeded, False otherwise
        """
        message = self._build_message(poi, df)
        print(message)

        success = True
        for chat_id in self.chat_ids:
            ok = self._send_to_single_chat(chat_id, message)
            success = success and ok

        return success

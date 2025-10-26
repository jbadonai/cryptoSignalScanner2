import pandas as pd
import numpy as np

class AdvancedTrendDetector:
    """
    A 1:1 Python implementation of the Pine Script Advanced Trend Detector.
    Replicates EMA, smoothed slope, ADX, +DI/-DI, and trend logic (Uptrend / Downtrend / Ranging).
    """

    def __init__(self, ema_length=50, adx_length=14, adx_threshold=20):
        self.ema_length = ema_length
        self.adx_length = adx_length
        self.adx_threshold = adx_threshold

    def _ema(self, series, length):
        """Exponential Moving Average"""
        return series.ewm(span=length, adjust=False).mean()

    def _sma(self, series, length):
        """Simple Moving Average"""
        return series.rolling(window=length, min_periods=1).mean()

    def _true_range(self, df):
        """True Range (same as ta.tr)"""
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr

    def _rma(self, series, length):
        """RMA (Wilder’s smoothing used in ADX calculations)"""
        alpha = 1 / length
        return series.ewm(alpha=alpha, adjust=False).mean()

    def detect_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects trend type for each bar: Uptrend / Downtrend / Ranging
        Returns DataFrame with added columns:
          ['ema', 'ema_slope', 'adx', 'plus_di', 'minus_di', 'trend']
        """
        df = df.copy()

        # === EMA Calculation ===
        df["ema"] = self._ema(df["close"], self.ema_length)

        # === Smoothed EMA slope ===
        df["ema_diff"] = df["ema"] - df["ema"].shift(1)
        df["ema_slope"] = self._sma(df["ema_diff"], 3)

        # === ADX Calculation ===
        up_move = df["high"] - df["high"].shift(1)
        down_move = df["low"].shift(1) - df["low"]

        df["plus_dm"] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        df["minus_dm"] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        tr = self._true_range(df)
        tr_rma = self._rma(tr, self.adx_length)

        df["plus_di"] = 100 * self._rma(df["plus_dm"], self.adx_length) / tr_rma
        df["minus_di"] = 100 * self._rma(df["minus_dm"], self.adx_length) / tr_rma

        dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"])
        df["adx"] = self._rma(dx, self.adx_length)

        # === Trend Logic ===
        df["is_uptrend"] = (df["close"] > df["ema"]) & (df["ema_slope"] > 0) & (df["adx"] > self.adx_threshold)
        df["is_downtrend"] = (df["close"] < df["ema"]) & (df["ema_slope"] < 0) & (df["adx"] > self.adx_threshold)
        df["is_ranging"] = ~(df["is_uptrend"] | df["is_downtrend"])

        df["trend"] = np.select(
            [df["is_uptrend"], df["is_downtrend"], df["is_ranging"]],
            ["Uptrend", "Downtrend", "Ranging"],
            default="Ranging"
        )

        return df

    def get_current_trend(self, df: pd.DataFrame) -> str:
        """Return the most recent trend value"""
        trend_df = self.detect_trend(df)
        return trend_df["trend"].iloc[-1]

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional


class ProtectedHighLowDetector:
    """
    Replicates the Pine script Protected High/Low logic exactly (pivots, ATR thresholds,
    BOS detection, array push/remove semantics, plotted outputs captured).
    """

    def __init__(
        self,
        left: int = 3,
        right: int = 3,
        min_break_atr_mult: float = 0.0,
        atr_len: int = 14,
        show_labels: bool = True,
        show_lines: bool = True,
        line_width: int = 2,
        bull_color: str = "green",
        bear_color: str = "red",
        label_text_low: str = "Protected Low",
        label_text_high: str = "Protected High",
        show_bias_bg: bool = True,
        show_color_sample: bool = True,
        show_protected: bool = True,   # assumption: ShowProtected = True
    ):
        # Pine params
        self.left = int(left)
        self.right = int(right)
        self.min_break_atr_mult = float(min_break_atr_mult)
        self.atr_len = int(atr_len)

        # visual params (we capture them but don't draw)
        self.show_labels = bool(show_labels)
        self.show_lines = bool(show_lines)
        self.line_width = int(line_width)
        self.bull_color = bull_color
        self.bear_color = bear_color
        self.label_text_low = label_text_low
        self.label_text_high = label_text_high
        self.show_bias_bg = bool(show_bias_bg)
        self.show_color_sample = bool(show_color_sample)
        self.show_protected = bool(show_protected)  # corresponds to ShowProtected in Pine

        # internal arrays to mimic Pine arrays
        self.prot_low_prices: List[float] = []
        self.prot_low_bars: List[int] = []
        self.prot_high_prices: List[float] = []
        self.prot_high_bars: List[int] = []

    @staticmethod
    def _compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
        # True Range (TR)
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's RMA via ewm with alpha=1/length (adjust=False) approximates Pine's ta.atr
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr

    def _detect_pivots(self, series: pd.Series, left: int, right: int, is_high: bool) -> pd.Series:
        """
        Returns a series of the same length with pivot value at its pivot-confirmation bar index.
        Implementation uses strict comparisons (>) for highs and (<) for lows, matching Pine.
        The pivot value will be placed at index pivot_index + right (i.e., the bar where Pine's
        ta.pivothigh/low becomes non-na).
        """
        s = series.values
        n = len(s)
        out = [np.nan] * n

        for j in range(left, n - right):
            center = s[j]
            left_slice = s[j - left: j]
            right_slice = s[j + 1: j + 1 + right]

            if is_high:
                if (center > left_slice.max()) and (center > right_slice.max()):
                    # Place pivot value at confirmation bar (j + right)
                    out[j + right] = center
            else:
                # pivot low (strict)
                if (center < left_slice.min()) and (center < right_slice.min()):
                    out[j + right] = center

        return pd.Series(out, index=series.index)

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process OHLC dataframe and return:
        {
          "levels": [ {metadata for each detected protected insertion and removal} ... ],
          "plotted_values": { ... per-bar latest info (list aligned with df) ... }
        }
        """
        df = df.copy().reset_index(drop=True)  # use integer bar indices 0..N-1 to match Pine bar_index semantics
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Input df must contain columns: {required_cols}")

        n = len(df)
        atr = self._compute_atr(df, self.atr_len)

        # compute pivot series using Pine-like strict logic
        ph_confirmed = self._detect_pivots(df['high'], self.left, self.right, is_high=True)
        pl_confirmed = self._detect_pivots(df['low'], self.left, self.right, is_high=False)

        # last confirmed pivot price/bar (behaviour in Pine)
        last_ph_price = np.nan
        last_ph_bar = None
        last_pl_price = np.nan
        last_pl_bar = None

        # output capture structures
        levels_events = []  # list of events: insertions and removals (detailed)
        plotted_per_bar = []  # capture per-bar plotting info like markers and bias

        # Working arrays to mimic Pine arrays (we reinitialize to ensure deterministic)
        prot_low_prices = []
        prot_low_bars = []
        prot_high_prices = []
        prot_high_bars = []

        for idx in range(n):
            high = df.at[idx, 'high']
            low = df.at[idx, 'low']
            close = df.at[idx, 'close']
            cur_atr = atr.iat[idx] if idx < len(atr) else np.nan

            # If a pivot confirmed here (ph_confirmed/pl_confirmed), update last_* variables exactly as Pine does:
            if not np.isnan(pl_confirmed.iat[idx]):
                # When pivot low confirmed at idx, pivot bar = idx - right (per Pine)
                pivot_bar = idx - self.right
                pivot_price = pl_confirmed.iat[idx]
                last_pl_price = pivot_price
                last_pl_bar = pivot_bar

            if not np.isnan(ph_confirmed.iat[idx]):
                pivot_bar = idx - self.right
                pivot_price = ph_confirmed.iat[idx]
                last_ph_price = pivot_price
                last_ph_bar = pivot_bar

            # BULLISH BOS → Protected Low (replicate Pine exact boolean logic)
            bull_bos = (
                (not np.isnan(last_ph_price)) and
                (not np.isnan(last_pl_price)) and
                (last_pl_bar is not None) and (last_ph_bar is not None) and
                (last_pl_bar < last_ph_bar) and
                (close > last_ph_price) and
                ( (self.min_break_atr_mult <= 0) or ((close - last_ph_price) >= (self.min_break_atr_mult * cur_atr)) )
            )

            if bull_bos:
                pl_price = float(last_pl_price)
                pl_bar = int(last_pl_bar)
                is_new_protected = (len(prot_low_bars) == 0) or (prot_low_bars[-1] != pl_bar)
                if is_new_protected:
                    # add to arrays
                    prot_low_prices.append(pl_price)
                    prot_low_bars.append(pl_bar)
                    levels_events.append({
                        "event": "create_protected_low",
                        "bar_index": idx,
                        "pivot_bar": pl_bar,
                        "price": pl_price,
                        "pine_params": {
                            "left": self.left, "right": self.right, "atr_len": self.atr_len,
                            "min_break_atr_mult": self.min_break_atr_mult
                        }
                    })

            # BEARISH BOS → Protected High
            bear_bos = (
                (not np.isnan(last_ph_price)) and
                (not np.isnan(last_pl_price)) and
                (last_ph_bar is not None) and (last_pl_bar is not None) and
                (last_ph_bar < last_pl_bar) and
                (close < last_pl_price) and
                ( (self.min_break_atr_mult <= 0) or ((last_pl_price - close) >= (self.min_break_atr_mult * cur_atr)) )
            )

            if bear_bos:
                ph_price = float(last_ph_price)
                ph_bar = int(last_ph_bar)
                is_new_h_protected = (len(prot_high_bars) == 0) or (prot_high_bars[-1] != ph_bar)
                if is_new_h_protected:
                    prot_high_prices.append(ph_price)
                    prot_high_bars.append(ph_bar)
                    levels_events.append({
                        "event": "create_protected_high",
                        "bar_index": idx,
                        "pivot_bar": ph_bar,
                        "price": ph_price,
                        "pine_params": {
                            "left": self.left, "right": self.right, "atr_len": self.atr_len,
                            "min_break_atr_mult": self.min_break_atr_mult
                        }
                    })

            # REMOVE MITIGATED LEVELS (iterate backwards as Pine does)
            # Protected lows removed when low < prot_low_price (strict)
            for i_rev in range(len(prot_low_prices) - 1, -1, -1):
                plv = prot_low_prices[i_rev]
                if low < plv:
                    # remove
                    removed_price = prot_low_prices.pop(i_rev)
                    removed_bar = prot_low_bars.pop(i_rev)
                    levels_events.append({
                        "event": "remove_protected_low",
                        "bar_index": idx,
                        "pivot_bar": removed_bar,
                        "price": removed_price
                    })

            # Protected highs removed when high > prot_high_price (strict)
            for j_rev in range(len(prot_high_prices) - 1, -1, -1):
                phv = prot_high_prices[j_rev]
                if high > phv:
                    removed_price = prot_high_prices.pop(j_rev)
                    removed_bar = prot_high_bars.pop(j_rev)
                    levels_events.append({
                        "event": "remove_protected_high",
                        "bar_index": idx,
                        "pivot_bar": removed_bar,
                        "price": removed_price
                    })

            # Determine last plotted/protected bar (as Pine does with array.size - 1)
            last_prot_low_bar = prot_low_bars[-1] if len(prot_low_bars) > 0 else None
            last_prot_high_bar = prot_high_bars[-1] if len(prot_high_bars) > 0 else None

            # Plot markers (Pine: if last_prot_low_bar == bar_index ? low : na)
            marker_low = low if (self.show_protected and (last_prot_low_bar == idx)) else np.nan
            marker_high = high if (self.show_protected and (last_prot_high_bar == idx)) else np.nan

            # Bias background: current_bias_protected = "bullish" if bull_bos else "bearish" if bear_bos else previous state
            # We'll keep a small state variable to persist bias between bars
            if idx == 0:
                current_bias = "neutral"
            # The Pine code changes bias immediately on detection, so mimic:
            if bull_bos:
                current_bias = "bullish"
            elif bear_bos:
                current_bias = "bearish"
            # else keep previous (we set neutral initially or previous value)

            # Build the per-bar plotted capture
            plotted_per_bar.append({
                "bar_index": idx,
                "marker_low": None if np.isnan(marker_low) else float(marker_low),
                "marker_high": None if np.isnan(marker_high) else float(marker_high),
                "bias": current_bias if self.show_bias_bg else None,
                "last_protected_low_list": prot_low_prices.copy(),
                "last_protected_low_bars": prot_low_bars.copy(),
                "last_protected_high_list": prot_high_prices.copy(),
                "last_protected_high_bars": prot_high_bars.copy(),
                # Table latest_type/latest_price logic:
                "latest_type": (None if (len(prot_low_prices) == 0 and len(prot_high_prices) == 0)
                                else ("Protected Low" if (len(prot_low_prices) > 0 and (len(prot_high_prices) == 0 or prot_low_bars[-1] > prot_high_bars[-1])) else "Protected High")),
                "latest_price": (None if (len(prot_low_prices) == 0 and len(prot_high_prices) == 0)
                                else (prot_low_prices[-1] if (len(prot_low_prices) > 0 and (len(prot_high_prices) == 0 or prot_low_bars[-1] > prot_high_bars[-1])) else prot_high_prices[-1]))
            })

        # Final packaged return:
        return {
            "levels_events": levels_events,
            "plotted_per_bar": plotted_per_bar,
            "final_state": {
                "prot_low_prices": prot_low_prices,
                "prot_low_bars": prot_low_bars,
                "prot_high_prices": prot_high_prices,
                "prot_high_bars": prot_high_bars
            },
            "pine_params_used": {
                "left": self.left, "right": self.right, "atr_len": self.atr_len,
                "min_break_atr_mult": self.min_break_atr_mult
            }
        }


# -------------------------
# # Example: small synthetic test
# # -------------------------
# if __name__ == "__main__":
#     # Build a tiny DataFrame with made-up bars that will create a pivot low then a bullish BOS
#     data = [
#         # o, h, l, c
#         (10, 10.5, 9.8, 10.2),
#         (10.2, 10.6, 10.0, 10.3),
#         (10.3, 10.4, 9.9, 10.1),
#         (10.1, 10.2, 9.7, 9.9),
#         (9.9, 10.3, 9.6, 10.25),
#         (10.25, 10.7, 10.2, 10.8),  # bullish break above previous ph (synthetic)
#         (10.8, 11.0, 10.7, 10.9),
#     ]
#     df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
#
#     detector = ProtectedHighLowDetector(left=1, right=1, min_break_atr_mult=0.0, atr_len=3)
#     result = detector.detect(df)
#     import json
#     print(json.dumps(result["levels_events"], indent=2))
#     print("Per-bar plotted summary (last 3 bars):")
#     for p in result["plotted_per_bar"][-3:]:
#         print(p)

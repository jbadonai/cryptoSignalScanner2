# OrderBlockDetector.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple


class OrderBlockDetector:
    """
    Python translation of Pine Order Block Detector with full parity adjustments.
    - Detects bullish and bearish order blocks using swing detection.
    - Implements breaker/invalidation logic and combination/merging of OBs.
    """

    def __init__(
        self,
        swing_length: int = 10,
        max_atr_mult: float = 3.5,
        atr_len: int = 10,
        ob_end_method: str = "Wick",  # "Wick" or "Close"
        combine_obs: bool = True,
        max_orderblocks: int = 30,
        max_boxes_count: int = 500,
        overlap_threshold_percentage: float = 0.0,
    ):
        self.swing_length = int(swing_length)
        self.max_atr_mult = float(max_atr_mult)
        self.atr_len = int(atr_len)
        self.ob_end_method = ob_end_method
        self.combine_obs = bool(combine_obs)
        self.max_orderblocks = int(max_orderblocks)
        self.max_boxes_count = int(max_boxes_count)
        self.overlap_threshold_percentage = float(overlap_threshold_percentage)

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr

    def _detect_swings(self, df: pd.DataFrame) -> Tuple[List[Optional[dict]], List[Optional[dict]]]:
        """
        Detect swing tops and bottoms using Pine-style lookback only.
        """
        n = len(df)
        s_high = df['high'].values
        s_low = df['low'].values
        s_vol = df['volume'].values if 'volume' in df.columns else np.zeros(n, dtype=float)

        top_swings = [None] * n
        bottom_swings = [None] * n

        for j in range(self.swing_length, n):
            lookback_high = s_high[j - self.swing_length:j + 1]
            lookback_low = s_low[j - self.swing_length:j + 1]

            center_high = s_high[j]
            center_low = s_low[j]

            # Pine-style: include equality
            if center_high >= lookback_high.max():
                top_swings[j] = {"bar": j, "price": float(center_high), "volume": float(s_vol[j])}
            if center_low <= lookback_low.min():
                bottom_swings[j] = {"bar": j, "price": float(center_low), "volume": float(s_vol[j])}

        return top_swings, bottom_swings

    def _compute_ob_box(self, df: pd.DataFrame, start_bar: int, end_bar: int) -> Tuple[float, float]:
        """
        Mimic Pine iterative boxTop/boxBtm calculation
        """
        window = df.iloc[start_bar:end_bar + 1]
        min_low = float('inf')
        max_high = -float('inf')
        box_top = None
        box_bottom = None

        for idx, row in window.iterrows():
            low = row['low']
            high = row['high']
            if low < min_low:
                min_low = low
                box_bottom = min_low
                box_top = high
            else:
                box_top = max(box_top, high) if box_top is not None else high

        return float(box_top), float(box_bottom)

    def _compute_ob_volume(self, df: pd.DataFrame, cross_idx: int, bullish: bool = True) -> Tuple[float, float, float]:
        """
        Assign volumes exactly like Pine
        """
        v0 = df.at[cross_idx, 'volume'] if cross_idx >= 0 else 0
        v1 = df.at[cross_idx - 1, 'volume'] if cross_idx - 1 >= 0 else 0
        v2 = df.at[cross_idx - 2, 'volume'] if cross_idx - 2 >= 0 else 0

        if bullish:
            ob_vol_total = v0 + v1 + v2
            ob_low_vol = v2
            ob_high_vol = v0 + v1
        else:  # Bearish
            ob_vol_total = v0 + v1 + v2
            ob_low_vol = v0 + v1
            ob_high_vol = v2

        return ob_vol_total, ob_low_vol, ob_high_vol


    @staticmethod
    def _safe_get_window(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        if end_idx < start_idx:
            return df.iloc[0:0]
        return df.iloc[start_idx:end_idx + 1]

    @staticmethod
    def _area_of_ob(start_idx: int, end_idx: int, top: float, bottom: float) -> float:
        width = float(max(1, end_idx - start_idx))
        height = float(max(0.0, top - bottom))
        return width * height

    def _do_obs_touch(self, ob1: dict, ob2: dict) -> bool:
        XA1, XA2, YA1, YA2 = ob1['start_idx'], ob1['end_idx'], ob1['top'], ob1['bottom']
        XB1, XB2, YB1, YB2 = ob2['start_idx'], ob2['end_idx'], ob2['top'], ob2['bottom']

        inter_w = max(0.0, min(XA2, XB2) - max(XA1, XB1))
        inter_h = max(0.0, min(YA1, YB1) - max(YA2, YB2))
        intersection_area = inter_w * inter_h
        area1 = self._area_of_ob(XA1, XA2, YA1, YA2)
        area2 = self._area_of_ob(XB1, XB2, YB1, YB2)
        union = max(1.0, area1 + area2 - intersection_area)
        overlap_percentage = (intersection_area / union) * 100.0
        return overlap_percentage > self.overlap_threshold_percentage

    # -------------------------
    # Main detect()
    # -------------------------
    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = df.reset_index(drop=True).copy()
        n = len(df)
        if n == 0:
            return {"orderblocks": [], "events": [], "per_bar": [], "final_state": {}}

        # Validate required columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f"Input df must contain column '{col}'")

        atr = self._compute_atr(df, self.atr_len)
        top_swings, bottom_swings = self._detect_swings(df)

        bullish_list: List[dict] = []
        bearish_list: List[dict] = []
        events: List[dict] = []
        per_bar: List[dict] = []

        self._top_crossed_flags = {}
        self._btm_crossed_flags = {}

        def make_ob_info(ob_type: str, top: float, bottom: float, start_idx: int, end_idx: Optional[int],
                         vol_total: float, low_vol: float, high_vol: float, bb_vol: Optional[float],
                         breaker: bool, combined: bool) -> dict:
            return {
                "type": ob_type,
                "top": float(top),
                "bottom": float(bottom),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx) if end_idx is not None else int(start_idx),
                "breaker": bool(breaker),
                "break_idx": None if end_idx is None else int(end_idx),
                "vol_total": float(vol_total),
                "obLowVolume": float(low_vol),
                "obHighVolume": float(high_vol),
                "bbVolume": float(bb_vol) if bb_vol is not None else None,
                "combined": bool(combined),
                "timeframe": None
            }

        # Iterate over all bars
        for idx in range(n):
            high = float(df.at[idx, 'high'])
            low = float(df.at[idx, 'low'])
            open_ = float(df.at[idx, 'open'])
            close = float(df.at[idx, 'close'])
            vol = float(df.at[idx, 'volume'])
            cur_atr = float(atr.iat[idx]) if idx < len(atr) else float('nan')

            # --- Step 1: Update existing OBs ---
            for ob_list, check_val, remove_cond, event_prefix in [
                (bullish_list, low if self.ob_end_method == "Wick" else min(open_, close), lambda o, v: v > o['top'],
                 "bull"),
                (bearish_list, high if self.ob_end_method == "Wick" else max(open_, close),
                 lambda o, v: v < o['bottom'], "bear")
            ]:
                i = len(ob_list) - 1
                while i >= 0:
                    ob = ob_list[i]
                    if not ob['breaker']:
                        if (event_prefix == "bull" and check_val < ob['bottom']) or (
                                event_prefix == "bear" and check_val > ob['top']):
                            ob['breaker'] = True
                            ob['break_idx'] = idx
                            ob['bbVolume'] = vol
                            events.append({"event": f"mark_breaker_{event_prefix}", "bar_index": idx, "ob": ob.copy()})
                    else:
                        if remove_cond(ob, check_val):
                            removed = ob_list.pop(i)
                            events.append({"event": f"remove_{event_prefix}_ob", "bar_index": idx, "ob": removed})
                    i -= 1

            # --- Step 2: Bullish OB creation ---
            top_candidate = next((top_swings[j] for j in range(idx - 1, -1, -1) if top_swings[j]), None)
            if top_candidate:
                top_bar, top_price = top_candidate['bar'], top_candidate['price']
                if not self._top_crossed_flags.get(top_bar, False) and close > top_price:
                    self._top_crossed_flags[top_bar] = True
                    box_top, box_bottom = self._compute_ob_box(df, top_bar, idx - 1)
                    ob_vol_total, ob_low_vol, ob_high_vol = self._compute_ob_volume(df, top_bar, bullish=True)
                    ob_size = abs(box_top - box_bottom)
                    if ob_size <= cur_atr * self.max_atr_mult:
                        new_ob = make_ob_info("Bull", box_top, box_bottom, top_bar, None,
                                              ob_vol_total, ob_low_vol, ob_high_vol, None, False, False)
                        bullish_list.insert(0, new_ob)
                        events.append({"event": "create_bullish_ob", "bar_index": idx, "ob": new_ob.copy()})
                        if len(bullish_list) > self.max_orderblocks:
                            popped = bullish_list.pop()
                            events.append({"event": "pop_old_bullish_ob", "bar_index": idx, "ob": popped})

            # --- Step 3: Bearish OB creation ---
            btm_candidate = next((bottom_swings[j] for j in range(idx - 1, -1, -1) if bottom_swings[j]), None)
            if btm_candidate:
                btm_bar, btm_price = btm_candidate['bar'], btm_candidate['price']
                if not self._btm_crossed_flags.get(btm_bar, False) and close < btm_price:
                    self._btm_crossed_flags[btm_bar] = True
                    box_top, box_bottom = self._compute_ob_box(df, btm_bar, idx - 1)
                    ob_vol_total, ob_low_vol, ob_high_vol = self._compute_ob_volume(df, btm_bar, bullish=False)
                    ob_size = abs(box_top - box_bottom)
                    if ob_size <= cur_atr * self.max_atr_mult:
                        new_ob = make_ob_info("Bear", box_top, box_bottom, btm_bar, None,
                                              ob_vol_total, ob_low_vol, ob_high_vol, None, False, False)
                        bearish_list.insert(0, new_ob)
                        events.append({"event": "create_bearish_ob", "bar_index": idx, "ob": new_ob.copy()})
                        if len(bearish_list) > self.max_orderblocks:
                            popped = bearish_list.pop()
                            events.append({"event": "pop_old_bearish_ob", "bar_index": idx, "ob": popped})

            # --- Step 4: Combine OBs (Optional) ---
            if self.combine_obs:
                for ob_list, ob_type in [(bullish_list, "Bull"), (bearish_list, "Bear")]:
                    combined_any = True
                    iterations = 0
                    while combined_any and iterations < 10:
                        combined_any = False
                        iterations += 1
                        outer = 0
                        while outer < len(ob_list):
                            inner = outer + 1
                            while inner < len(ob_list):
                                ob1, ob2 = ob_list[outer], ob_list[inner]
                                ob1_end = ob1.get('break_idx') or idx
                                ob2_end = ob2.get('break_idx') or idx
                                if self._do_obs_touch(
                                        {"start_idx": ob1['start_idx'], "end_idx": ob1_end, "top": ob1['top'],
                                         "bottom": ob1['bottom']},
                                        {"start_idx": ob2['start_idx'], "end_idx": ob2_end, "top": ob2['top'],
                                         "bottom": ob2['bottom']}):
                                    new_top = max(ob1['top'], ob2['top'])
                                    new_bottom = min(ob1['bottom'], ob2['bottom'])
                                    new_start = min(ob1['start_idx'], ob2['start_idx'])
                                    new_break = max(ob1.get('break_idx') or 0, ob2.get('break_idx') or 0) or None
                                    new_vol = ob1['vol_total'] + ob2['vol_total']
                                    new_lowvol = ob1['obLowVolume'] + ob2['obLowVolume']
                                    new_highvol = ob1['obHighVolume'] + ob2['obHighVolume']
                                    new_bbvol = (ob1.get('bbVolume') or 0) + (ob2.get('bbVolume') or 0)
                                    combined_ob = make_ob_info(ob_type, new_top, new_bottom, new_start, new_break,
                                                               new_vol, new_lowvol, new_highvol, new_bbvol,
                                                               ob1['breaker'] or ob2['breaker'], True)
                                    ob_list.pop(inner)
                                    ob_list.pop(outer)
                                    ob_list.insert(0, combined_ob)
                                    events.append({"event": f"combine_{ob_type.lower()}_ob", "bar_index": idx,
                                                   "ob": combined_ob.copy(), "components": [ob1.copy(), ob2.copy()]})
                                    combined_any = True
                                    outer = 0
                                    inner = outer + 1
                                    continue
                                inner += 1
                            outer += 1

            # --- Step 5: Snapshot per bar ---
            latest_ob = bullish_list[0] if bullish_list else (bearish_list[0] if bearish_list else None)
            latest_type = (
                "Bullish OB" if latest_ob and latest_ob["type"] == "Bull" else ("Bearish OB" if latest_ob else None))
            per_bar.append({
                "bar_index": idx,
                "active_bullish_count": len(bullish_list),
                "active_bearish_count": len(bearish_list),
                "active_bullish": [
                    {"top": ob['top'], "bottom": ob['bottom'], "start_idx": ob['start_idx'], "breaker": ob['breaker'],
                     "vol_total": ob['vol_total']} for ob in bullish_list],
                "active_bearish": [
                    {"top": ob['top'], "bottom": ob['bottom'], "start_idx": ob['start_idx'], "breaker": ob['breaker'],
                     "vol_total": ob['vol_total']} for ob in bearish_list],
                "latest_type": latest_type,
                "latest_ob": latest_ob
            })

        final_obs = []
        for ob in bullish_list: ob.update({"direction": "Bull"}); final_obs.append(ob.copy())
        for ob in bearish_list: ob.update({"direction": "Bear"}); final_obs.append(ob.copy())

        return {
            "orderblocks": final_obs,
            "events": events,
            "per_bar": per_bar,
            "final_state": {"bullish_list": bullish_list, "bearish_list": bearish_list},
            "params": {"swing_length": self.swing_length, "max_atr_mult": self.max_atr_mult, "atr_len": self.atr_len,
                       "ob_end_method": self.ob_end_method, "combine_obs": self.combine_obs}
        }

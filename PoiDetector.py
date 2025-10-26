import pandas as pd
from typing import Any, Dict, List, Optional


class POIDetector:
    """
    Combines outputs from:
      - ProtectedHighLowDetector
      - OrderBlockDetector
      - FVGDetector
    into Points of Interest (POI).

    Returns both POI objects and per-bar events for later plotting/logging.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]],
        protected_detector,
        ob_detector,
        fvg_detector,
        symbol: str = "LTCUSDT",
        timeframe: str = "15m",
    ):
        self.config = config or {}
        self.protected = protected_detector
        self.ob = ob_detector
        self.fvg = fvg_detector
        self.symbol = symbol
        self.timeframe = timeframe

        # Pull default parameters from config or detectors
        self.swing_proximity: int = int(
            self.config.get(
                "swing_proximity",
                getattr(self.protected, "left", 3) + getattr(self.protected, "right", 3)
            )
        )
        self.fvg_search_limit: Optional[int] = self.config.get("fvg_search_limit", None)
        self.breakout_window: int = int(self.config.get("breakout_window", 10))

    # -------------------------
    # Normalizers
    # -------------------------
    def _normalize_protected(self, protected_out: Any) -> List[Dict]:
        """
        Converts ProtectedHighLowDetector output into a uniform list
        of {'event', 'price', 'pivot_bar'} dicts.
        """
        if protected_out is None:
            return []

        # Handle new structure with 'final_state'
        if isinstance(protected_out, dict):
            fs = protected_out.get("final_state", {})
            highs, lows = [], []

            # Extract protected highs
            for price, bar in zip(fs.get("prot_high_prices", []), fs.get("prot_high_bars", [])):
                highs.append({
                    "event": "protected_high",
                    "price": float(price),
                    "pivot_bar": int(bar)
                })

            # Extract protected lows
            for price, bar in zip(fs.get("prot_low_prices", []), fs.get("prot_low_bars", [])):
                lows.append({
                    "event": "protected_low",
                    "price": float(price),
                    "pivot_bar": int(bar)
                })

            # Fallback if legacy field 'levels_events' exists
            if not highs and not lows and "levels_events" in protected_out:
                return list(protected_out.get("levels_events", []))

            return highs + lows

        # If already a list-based structure (legacy)
        if isinstance(protected_out, list):
            return protected_out

        return []

    def _normalize_obs(self, ob_out: Any) -> List[Dict]:
        if ob_out is None:
            return []
        if isinstance(ob_out, dict):
            return list(ob_out.get("orderblocks", []))
        if isinstance(ob_out, list):
            return ob_out
        return []

    def _normalize_fvgs(self, fvg_out: Any) -> List[Dict]:
        if fvg_out is None:
            return []

        if isinstance(fvg_out, dict):
            fvgs = fvg_out.get("fvgs") or fvg_out.get("fvg_list") or []
        else:
            fvgs = fvg_out

        normalized = []
        for f in fvgs:
            if hasattr(f, "__dict__"):
                d = vars(f)
            else:
                d = dict(f)

            normalized.append({
                "start_idx": int(d.get("start_index") or d.get("start_idx") or 0),
                "end_idx": int(d.get("end_index") or d.get("end_idx") or 0),
                "top": float(d.get("top", 0)),
                "bottom": float(d.get("bottom", 0)),
                "type": str(d.get("type", "")).lower(),
                "mitigated": bool(d.get("mitigated", False))
            })

        if self.fvg_search_limit:
            normalized = normalized[-self.fvg_search_limit:]
        return normalized

    # -------------------------
    # Helpers
    # -------------------------
    def _split_protected(self, protected_events: List[Dict]):
        highs, lows = [], []
        for e in protected_events:
            ev = e.get("event", "").lower()
            if "high" in ev:
                highs.append(e)
            elif "low" in ev:
                lows.append(e)
        return highs, lows

    def _find_closest_unmitigated_fvg(self, fvgs, ref_price, direction):
        candidates = []
        for f in fvgs:
            if f["mitigated"]:
                continue
            if direction == "below" and f["top"] < ref_price:
                candidates.append((abs(ref_price - f["top"]), f))
            elif direction == "above" and f["bottom"] > ref_price:
                candidates.append((abs(f["bottom"] - ref_price), f))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _timestamp(self, df: pd.DataFrame, idx: int) -> Optional[str]:
        if "ts" in df.columns and 0 <= idx < len(df):
            return pd.to_datetime(df.iloc[idx]["ts"]).isoformat()
        return None

    # -------------------------
    # Core Detection
    # -------------------------
    def find_pois(self, df: pd.DataFrame) -> Dict[str, Any]:
        protected_raw = self.protected.detect(df)
        ob_raw = self.ob.detect(df)
        fvg_raw = self.fvg.detect(df)

        protected_events = self._normalize_protected(protected_raw)
        obs = self._normalize_obs(ob_raw)
        fvgs = self._normalize_fvgs(fvg_raw)

        highs, lows = self._split_protected(protected_events)

        pois, events = [], []

        # --- BEARISH (Protected High + Bearish OB + FVG below)
        for ph in highs:
            pivot_bar = int(ph.get("pivot_bar", ph.get("bar_index", 0)))
            protected_price = float(ph.get("price", 0))

            for ob in obs:
                if "bear" not in ob.get("type", "").lower():
                    continue
                start_idx = int(ob.get("start_idx", 0))
                if abs(start_idx - pivot_bar) > self.swing_proximity:
                    continue

                ob_top = float(ob.get("top", 0))
                ob_bottom = float(ob.get("bottom", 0))
                fvg = self._find_closest_unmitigated_fvg(fvgs, ob_top, "below")

                if fvg:
                    fvg_idx = int(fvg["start_idx"])
                    poi_idx = max(pivot_bar, start_idx, fvg_idx)
                    fvg_top = fvg["top"]
                    fvg_bottom = fvg["bottom"]
                else:
                    # fallback: find breakout OR last close after OB
                    breakout_idx = None
                    for i in range(start_idx, min(len(df), start_idx + self.breakout_window)):
                        if df["close"].iloc[i] < ob_bottom:
                            breakout_idx = i
                            break

                    if breakout_idx is not None:
                        poi_idx = breakout_idx
                        fvg_top = df["close"].iloc[breakout_idx]
                        fvg_bottom = df["close"].iloc[breakout_idx]
                    else:
                        poi_idx = min(len(df) - 1, start_idx + 1)
                        fvg_top = df["close"].iloc[poi_idx]
                        fvg_bottom = df["close"].iloc[poi_idx]

                poi = {
                    "symbol": self.symbol,
                    "protected_type": "Protected High",
                    "protected_price": protected_price,
                    "ob_top": ob_top,
                    "ob_bottom": ob_bottom,
                    "fvg_top": fvg_top,
                    "fvg_bottom": fvg_bottom,
                    "fvg_index": fvg["start_idx"] if fvg else None,
                    "poi_index": poi_idx,
                    "timestamp": self._timestamp(df, poi_idx),
                }

                pois.append(poi)
                events.append({
                    "event": "create_poi",
                    "timestamp": poi["timestamp"],
                    "poi_index": poi["poi_index"],
                    "protected_type": poi["protected_type"],
                    "ob_range": [ob_bottom, ob_top],
                    "fvg": [poi["fvg_bottom"], poi["fvg_top"]],
                })

        # --- BULLISH (Protected Low + Bullish OB + FVG above)
        for pl in lows:
            pivot_bar = int(pl.get("pivot_bar", pl.get("bar_index", 0)))
            protected_price = float(pl.get("price", 0))

            for ob in obs:
                if "bull" not in ob.get("type", "").lower():
                    continue
                start_idx = int(ob.get("start_idx", 0))
                if abs(start_idx - pivot_bar) > self.swing_proximity:
                    continue

                ob_top = float(ob.get("top", 0))
                ob_bottom = float(ob.get("bottom", 0))
                fvg = self._find_closest_unmitigated_fvg(fvgs, ob_bottom, "above")

                if fvg:
                    fvg_idx = int(fvg["start_idx"])
                    poi_idx = max(pivot_bar, start_idx, fvg_idx)
                    fvg_top = fvg["top"]
                    fvg_bottom = fvg["bottom"]
                else:
                    # fallback: find breakout OR last close after OB
                    breakout_idx = None
                    for i in range(start_idx, min(len(df), start_idx + self.breakout_window)):
                        if df["close"].iloc[i] > ob_top:
                            breakout_idx = i
                            break

                    if breakout_idx is not None:
                        poi_idx = breakout_idx
                        fvg_top = df["close"].iloc[breakout_idx]
                        fvg_bottom = df["close"].iloc[breakout_idx]
                    else:
                        poi_idx = min(len(df) - 1, start_idx + 1)
                        fvg_top = df["close"].iloc[poi_idx]
                        fvg_bottom = df["close"].iloc[poi_idx]

                poi = {
                    "symbol": self.symbol,
                    "protected_type": "Protected Low",
                    "protected_price": protected_price,
                    "ob_top": ob_top,
                    "ob_bottom": ob_bottom,
                    "fvg_top": fvg_top,
                    "fvg_bottom": fvg_bottom,
                    "fvg_index": fvg["start_idx"] if fvg else None,
                    "poi_index": poi_idx,
                    "timestamp": self._timestamp(df, poi_idx),
                }

                pois.append(poi)
                events.append({
                    "event": "create_poi",
                    "timestamp": poi["timestamp"],
                    "poi_index": poi["poi_index"],
                    "protected_type": poi["protected_type"],
                    "ob_range": [ob_bottom, ob_top],
                    "fvg": [poi["fvg_bottom"], poi["fvg_top"]],
                })

        pois = sorted(pois, key=lambda x: x["poi_index"])
        return {"pois": pois, "events": events}

    def find_latest_poi(self, df: pd.DataFrame) -> Optional[Dict]:
        result = self.find_pois(df)
        pois = result["pois"]
        return pois[-1] if pois else None

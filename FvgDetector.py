import pandas as pd
from dataclasses import dataclass

@dataclass
class FVG:
    start_index: int
    end_index: int
    top: float
    bottom: float
    type: str  # 'bullish' or 'bearish'
    mitigated: bool = False

class FVGDetector:
    """
    Detects and tracks Fair Value Gaps (FVG) in OHLC data.

    Logic adapted from Pine Script:
        - Bullish FVG: high[3] < low[1]
        - Bearish FVG: low[3] > high[1]
    The detector maintains a rolling history of active FVGs and checks if they
    become mitigated when price fills the gap.
    """

    def __init__(self, fvg_history_limit: int = 5, reduce_mitigated: bool = False):
        """
        :param fvg_history_limit: Number of FVGs to keep active in memory.
        :param reduce_mitigated: If True, reduces the gap visually when mitigated.
        """
        self.fvg_history_limit = fvg_history_limit
        self.reduce_mitigated = reduce_mitigated
        self.fvgs: list[FVG] = []

    def detect(self, data: pd.DataFrame) -> list[FVG]:
        """
        Detects and tracks FVGs over the OHLC DataFrame.

        DataFrame must contain columns: ['open', 'high', 'low', 'close'].
        Returns a list of FVG objects, including mitigation updates.
        """
        highs, lows = data['high'].values, data['low'].values
        n = len(data)

        for i in range(3, n):
            # --- Detect Bullish FVG (high[3] < low[1]) ---
            if highs[i - 3] < lows[i - 1]:
                top = lows[i - 1]
                bottom = highs[i - 3]
                fvg = FVG(start_index=i - 2, end_index=i - 1, top=top, bottom=bottom, type='bullish')
                self._add_fvg(fvg)

            # --- Detect Bearish FVG (low[3] > high[1]) ---
            elif lows[i - 3] > highs[i - 1]:
                top = lows[i - 3]
                bottom = highs[i - 1]
                fvg = FVG(start_index=i - 2, end_index=i - 1, top=top, bottom=bottom, type='bearish')
                self._add_fvg(fvg)

            # --- Update existing FVGs for mitigation ---
            self._update_mitigation(i, highs[i], lows[i])

        return self.fvgs

    def _add_fvg(self, fvg: FVG):
        """Add a new FVG, ensuring we don’t exceed the rolling history."""
        self.fvgs.append(fvg)
        if len(self.fvgs) > self.fvg_history_limit:
            self.fvgs.pop(0)

    def _update_mitigation(self, index: int, high: float, low: float):
        """
        Update mitigation status of existing FVGs based on new price data.
        Mitigation means the price has traded back into the gap range.
        """
        for fvg in self.fvgs:
            if fvg.mitigated:
                continue  # Skip already mitigated ones

            if fvg.type == 'bullish':
                # Bullish FVG mitigated if price low touches gap top
                if low <= fvg.top:
                    fvg.mitigated = True
                    if self.reduce_mitigated:
                        fvg.top = low  # visually reduce

            elif fvg.type == 'bearish':
                # Bearish FVG mitigated if price high touches gap bottom
                if high >= fvg.bottom:
                    fvg.mitigated = True
                    if self.reduce_mitigated:
                        fvg.bottom = high  # visually reduce

    def get_active_fvgs(self) -> list[FVG]:
        """Returns a list of unmitigated FVGs."""
        return [fvg for fvg in self.fvgs if not fvg.mitigated]

    def get_all_fvgs(self) -> list[FVG]:
        """Returns all detected FVGs (both mitigated and active)."""
        return self.fvgs

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all detected FVGs into a Pandas DataFrame."""
        return pd.DataFrame([vars(fvg) for fvg in self.fvgs])

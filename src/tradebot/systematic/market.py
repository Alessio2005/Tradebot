"""De markt zoals het boek hem ziet: prijzen, rendementen, funding, ADV en risico.

Alles op één gatenloos dagraster (`data/weekly_market.py`). Vóór de notering van een
munt staat er NaN; dat is geen gat maar het feit dat het instrument nog niet bestond.
Elke afgeleide grootheid hier is causaal: de waarde op bar *t* gebruikt alleen data
die op de close van *t* bekend is.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..data.weekly_market import WeeklyMarket
from ..utils.failfast import DataContractError, require

__all__ = ["BookMarket", "ewma_covariance", "ewma_vol", "from_weekly_market"]

BARS_PER_YEAR = 365.0


@dataclass(frozen=True)
class BookMarket:
    close: pd.DataFrame
    ret: pd.DataFrame
    funding: pd.DataFrame
    adv_usd: pd.DataFrame
    sigma_daily: pd.DataFrame
    source_hashes: dict[str, str]

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.close.columns)

    def live(self, min_history_bars: int) -> pd.DataFrame:
        """Handelbaar: `min_history_bars` geldige closes ÉN een gedefinieerd kostenmodel.

        Het kostenmodel vraagt een ADV en een dagvolatiliteit (impact); een munt waarvoor
        die nog niet bestaan, is op die bar niet te verhandelen tegen bekende kosten.
        """
        history = self.close.notna().cumsum()
        return ((history >= int(min_history_bars)) & self.close.notna()
                & self.sigma_daily.notna() & self.adv_usd.gt(0.0))

    def subset(self, symbols: tuple[str, ...]) -> BookMarket:
        cols = list(symbols)
        return replace(
            self, close=self.close[cols], ret=self.ret[cols], funding=self.funding[cols],
            adv_usd=self.adv_usd[cols], sigma_daily=self.sigma_daily[cols])

    def truncate(self, end: pd.Timestamp) -> BookMarket:
        """Alles strikt vóór `end`; voor de causaliteitstoetsen."""
        keep = self.index < end
        return replace(
            self, close=self.close.loc[keep], ret=self.ret.loc[keep],
            funding=self.funding.loc[keep], adv_usd=self.adv_usd.loc[keep],
            sigma_daily=self.sigma_daily.loc[keep])


def ewma_vol(ret: pd.DataFrame, *, span: int) -> pd.DataFrame:
    """Dagelijkse EWMA-volatiliteit, causaal (inclusief het rendement van *t*), nul-gemiddeld."""
    var = (ret ** 2).ewm(span=span, min_periods=max(span // 3, 10)).mean()
    return np.sqrt(var)


def ewma_covariance(ret: pd.DataFrame, *, span: int) -> np.ndarray:
    """EWMA-covariantie per bar, nul-gemiddeld, shape (T, N, N); NaN waar niet geschat.

    Ontbrekende rendementen (vóór de notering) tellen als nul in het product maar de
    rij en kolom van zo'n munt worden NaN tot hij genoteerd is, zodat een nog niet
    bestaand instrument nooit een covariantie suggereert.
    """
    x = ret.to_numpy(dtype=np.float64)
    t_len, n = x.shape
    alpha = 2.0 / (span + 1.0)
    min_obs = max(span // 3, 10)
    out = np.full((t_len, n, n), np.nan)
    s = np.zeros((n, n))
    w = 0.0
    seen = np.zeros(n)
    for t in range(t_len):
        row = x[t]
        ok = np.isfinite(row)
        seen += ok
        r = np.where(ok, row, 0.0)
        s = (1.0 - alpha) * s + alpha * np.outer(r, r)
        w = (1.0 - alpha) * w + alpha
        cov = s / w
        valid = seen >= min_obs
        cov = np.where(np.outer(valid, valid), cov, np.nan)
        out[t] = cov
    return out


def from_weekly_market(wm: WeeklyMarket) -> BookMarket:
    """Bouw de boekmarkt uit de gecertificeerde dagmarkt."""
    close = pd.DataFrame({s: f["close"] for s, f in wm.ohlcv.items()})
    ret = close.pct_change(fill_method=None)
    # Een munt die ná zijn notering een ontbrekende close heeft, is een gat dat het
    # boek niet stil mag overbruggen.
    listed = close.notna().cummax()
    require(not bool((listed & close.isna()).any().any()),
            "Ontbrekende close na de notering.", DataContractError)
    funding = wm.funding.reindex(columns=close.columns).where(listed, 0.0).fillna(0.0)
    return BookMarket(
        close=close, ret=ret, funding=funding,
        adv_usd=wm.adv_usd.reindex(columns=close.columns),
        sigma_daily=wm.sigma_daily.reindex(columns=close.columns),
        source_hashes=dict(wm.source_hashes),
    )

"""
q23.strategy.factors

V4-Enhanced factor library (24 factors) implemented with pure xarray/numpy/pandas,
so the modular engine can reproduce the original "robust hybrid alpha" behavior.

Design:
- Input: MarketDataBundle-like or xr.Dataset with variables: close, high, low, vol, is_liquid(optional)
- Output: F (factor, time, asset) cross-sectionally z-scored (robust or standard)

Key safety:
- All returns clipped to > -1 for log1p
- All outputs finite-filled
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import pandas as pd

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from q23.shared.math_utils import safe_zscore


def _require_xr() -> None:
    if xr is None:
        raise ImportError("xarray is required for q23.strategy.factors")


# --------------------------
# Helpers (pure xarray)
# --------------------------

def _clip_ret(r: "xr.DataArray") -> "xr.DataArray":
    return r.fillna(0.0).clip(min=-0.999999)


def _log1p_safe(x: "xr.DataArray") -> "xr.DataArray":
    x = x.fillna(0.0).clip(min=-0.999999)
    return xr.apply_ufunc(np.log1p, x)


def _sma(x: "xr.DataArray", win: int) -> "xr.DataArray":
    return x.rolling(time=win, min_periods=win).mean()


def _std(x: "xr.DataArray", win: int) -> "xr.DataArray":
    return x.rolling(time=win, min_periods=win).std()


def _ema(x: "xr.DataArray", span: int) -> "xr.DataArray":
    # xarray has ewm in newer versions but not always; do pandas along time.
    # Efficient enough for SPX universe.
    df = x.transpose("time", "asset").to_pandas()
    out = df.ewm(span=span, adjust=False, min_periods=span).mean()
    return xr.DataArray(out.values, coords={"time": x.time, "asset": x.asset}, dims=["time", "asset"]).fillna(0.0)


def _rolling_max(x: "xr.DataArray", win: int) -> "xr.DataArray":
    return x.rolling(time=win, min_periods=win).max()


def _rolling_min(x: "xr.DataArray", win: int) -> "xr.DataArray":
    return x.rolling(time=win, min_periods=win).min()


def _atr(high: "xr.DataArray", low: "xr.DataArray", close: "xr.DataArray", win: int) -> "xr.DataArray":
    prev = close.shift(time=1)
    hl = high - low
    hc = np.abs(high - prev)
    lc = np.abs(low - prev)
    tr = xr.where(hl > hc, hl, hc)
    tr = xr.where(tr > lc, tr, lc)
    return _sma(tr.fillna(0.0), win).fillna(0.0)


def _market_return(r: "xr.DataArray") -> "xr.DataArray":
    return r.fillna(0.0).mean("asset")


def _rolling_beta(returns: "xr.DataArray", mkt: "xr.DataArray", win: int, eps: float = 1e-12) -> "xr.DataArray":
    r, m = xr.align(returns, mkt, join="inner")
    cov = _sma(r * m, win) - _sma(r, win) * _sma(m, win)
    var = _sma(m * m, win) - _sma(m, win) ** 2
    return (cov / (var + eps)).fillna(0.0)


def _residual_returns(returns: "xr.DataArray", beta: "xr.DataArray", mkt: "xr.DataArray") -> "xr.DataArray":
    r, b, m = xr.align(returns, beta, mkt, join="inner")
    return (r - b * m).fillna(0.0)


def _robust_zscore(x: "xr.DataArray", dim: str = "asset", eps: float = 1e-12, clip: float = 5.0) -> "xr.DataArray":
    med = x.median(dim=dim, skipna=True)
    mad = np.abs(x - med).median(dim=dim, skipna=True)
    z = (x - med) / (mad * 1.4826 + eps)
    return z.fillna(0.0).clip(-clip, clip)


# --------------------------
# Params + factor set
# --------------------------

@dataclass(frozen=True)
class FactorParams:
    eps: float = 1e-12
    zclip: float = 5.0
    robust_cs_z: bool = True  # match your original robust_zscore feel

    # windows (ported from your v4 config defaults)
    BETA_WIN: int = 162
    IDIO_WIN: int = 63
    DOWN_WIN: int = 63
    CORR_WIN: int = 63
    ADV_WIN_LONG: int = 63
    ADV_WIN_SHORT: int = 5

    MOM_WIN: int = 84
    REV_WIN: int = 5
    DON_WIN: int = 63
    EMA_FAST: int = 20
    EMA_SLOW: int = 42
    ATR_WIN: int = 42

    SECTOR_MOM_WIN: int = 63


V4_24_FACTORS = (
    # Defensive / low risk
    "inv_vol",
    "inv_idio",
    "inv_down",
    "low_corr",
    "low_beta",
    "liquidity",
    "amihud_inv",

    # Alpha
    "resid_mom",
    "srev",
    "breakout",
    "slope",
    "calm_flow",
    "prox_52w_high",
    "resid_mom_mix",
    "vol_surprise",
    "vol_breakout",
    "ma_cloud",

    # Idio-quality / microstructure
    "idio_tail_risk",
    "idio_jump_freq",
    "beta_stability",
    "micro_noise",

    # Interactions
    "value_mom",
    "quality_defensive",

    # Sector-relative proxy
    "rel_sector_mom",
)


class FactorLibrary:
    def __init__(
        self,
        market_data: Union["xr.Dataset", Any],
        *,
        params: Optional[FactorParams] = None,
        factors: Sequence[str] = V4_24_FACTORS,
        asset_dim: str = "asset",
    ):
        _require_xr()
        self.params = params or FactorParams()
        self.factors = list(factors)
        self.asset_dim = asset_dim

        if isinstance(market_data, xr.Dataset):
            self.data = market_data
        else:
            if hasattr(market_data, "data") and isinstance(getattr(market_data, "data"), xr.Dataset):
                self.data = getattr(market_data, "data")
            else:
                raise TypeError("market_data must be xr.Dataset or bundle-like with .data")

        # expected variables
        for v in ("close", "high", "low", "vol"):
            if v not in self.data:
                raise ValueError(f"Dataset missing required variable '{v}'")

        self.data = self.data.transpose("time", asset_dim, missing_dims="ignore")
        self._artifacts: Dict[str, "xr.DataArray"] = {}

        # precompute common series
        self.close = self.data["close"].fillna(0.0)
        self.high = self.data["high"].fillna(0.0)
        self.low = self.data["low"].fillna(0.0)
        self.vol = self.data["vol"].fillna(0.0)

        self.ret = _clip_ret(self.close / self.close.shift(time=1) - 1.0)
        self.mkt = _market_return(self.ret)

        self.beta = _rolling_beta(self.ret, self.mkt, self.params.BETA_WIN, eps=self.params.eps)
        self.resid = _residual_returns(self.ret, self.beta, self.mkt)

        self._artifacts["beta"] = self.beta
        self._artifacts["resid"] = self.resid

    def _cs_z(self, da: "xr.DataArray") -> "xr.DataArray":
        if self.params.robust_cs_z:
            return _robust_zscore(da, dim=self.asset_dim, eps=self.params.eps, clip=self.params.zclip)
        return safe_zscore(da, dim=self.asset_dim, eps=self.params.eps, clip=self.params.zclip)

    def compute(self) -> Tuple["xr.DataArray", Dict[str, "xr.DataArray"]]:
        Fs: List["xr.DataArray"] = []
        names: List[str] = []

        for name in self.factors:
            fn = getattr(self, f"_factor_{name}", None)
            if fn is None:
                raise ValueError(f"Unknown factor '{name}'. Implement _factor_{name}.")
            raw = fn().transpose("time", self.asset_dim).fillna(0.0)
            z = self._cs_z(raw).fillna(0.0)
            z.name = name
            Fs.append(z)
            names.append(name)

        F = xr.concat(Fs, dim="factor").assign_coords(factor=names).transpose("factor", "time", self.asset_dim)
        F.name = "F"
        return F, dict(self._artifacts)

    # --------------------------
    # Defensive / low risk
    # --------------------------

    def _factor_inv_vol(self) -> "xr.DataArray":
        atr = _atr(self.high, self.low, self.close, self.params.ATR_WIN)
        return (1.0 / (atr + self.params.eps)).fillna(0.0)

    def _factor_inv_idio(self) -> "xr.DataArray":
        idio_vol = _std(self.resid, self.params.IDIO_WIN).fillna(0.0)
        return (1.0 / (idio_vol + self.params.eps)).fillna(0.0)

    def _factor_inv_down(self) -> "xr.DataArray":
        down = self.ret.where(self.ret < 0.0, 0.0)
        down_vol = _std(down, self.params.DOWN_WIN).fillna(0.0)
        return (1.0 / (down_vol + self.params.eps)).fillna(0.0)

    def _factor_low_corr(self) -> "xr.DataArray":
        # corr(asset, mkt) over CORR_WIN using pandas for stability
        r_df = self.ret.transpose("time", "asset").to_pandas()
        m = self.mkt.to_pandas()
        corr = pd.DataFrame(index=r_df.index, columns=r_df.columns, dtype=float)
        for c in r_df.columns:
            corr[c] = r_df[c].rolling(self.params.CORR_WIN).corr(m)
        da = xr.DataArray(corr.fillna(0.0).values, coords=self.ret.coords, dims=self.ret.dims)
        return (-da).fillna(0.0)

    def _factor_low_beta(self) -> "xr.DataArray":
        return (-self.beta).fillna(0.0)

    def _factor_liquidity(self) -> "xr.DataArray":
        adv = _sma(self.vol * self.close, self.params.ADV_WIN_LONG).fillna(0.0)
        # normalize vs cross-sectional mean
        mu = adv.mean("asset")
        return (adv / (mu + self.params.eps)).fillna(0.0)

    def _factor_amihud_inv(self) -> "xr.DataArray":
        dv = (self.vol * self.close).fillna(0.0)
        amihud = (np.abs(self.ret) / (dv + self.params.eps)).fillna(0.0)
        return (1.0 / (amihud + self.params.eps)).fillna(0.0)

    # --------------------------
    # Alpha
    # --------------------------

    def _factor_resid_mom(self) -> "xr.DataArray":
        return _sma(self.resid, self.params.MOM_WIN).fillna(0.0)

    def _factor_srev(self) -> "xr.DataArray":
        return (-(self.close / self.close.shift(time=self.params.REV_WIN) - 1.0)).fillna(0.0)

    def _factor_breakout(self) -> "xr.DataArray":
        mx = _rolling_max(self.close, self.params.DON_WIN)
        mn = _rolling_min(self.close, self.params.DON_WIN)
        return ((self.close - mn) / (mx - mn + self.params.eps)).fillna(0.0)

    def _factor_slope(self) -> "xr.DataArray":
        atr = _atr(self.high, self.low, self.close, self.params.ATR_WIN)
        ema_f = _ema(self.close, self.params.EMA_FAST)
        ema_s = _ema(self.close, self.params.EMA_SLOW)
        return ((ema_f - ema_s) / (atr + self.params.eps)).fillna(0.0)

    def _factor_calm_flow(self) -> "xr.DataArray":
        vs = _sma(self.vol, self.params.ADV_WIN_SHORT)
        vl = _sma(self.vol, self.params.ADV_WIN_LONG)
        return (-(vs / (vl + self.params.eps) - 1.0)).fillna(0.0)

    def _factor_prox_52w_high(self) -> "xr.DataArray":
        roll_max = _rolling_max(self.close, 252)
        prox = (self.close / (roll_max + self.params.eps) - 1.0)
        return (-prox).fillna(0.0)

    def _factor_resid_mom_mix(self) -> "xr.DataArray":
        mom21 = _sma(self.resid, 21)
        mom63 = _sma(self.resid, 63)
        mom126 = _sma(self.resid, 126)
        return (0.50 * mom63 + 0.30 * mom126 + 0.20 * mom21).fillna(0.0)

    def _factor_vol_surprise(self) -> "xr.DataArray":
        dv = (self.vol * self.close).fillna(0.0)
        base = _sma(dv, 20)
        return (dv / (base + self.params.eps) - 1.0).fillna(0.0)

    def _factor_vol_breakout(self) -> "xr.DataArray":
        atr = _atr(self.high, self.low, self.close, self.params.ATR_WIN)
        atr_exp = (atr / (_sma(atr, 20) + self.params.eps) - 1.0).fillna(0.0)
        mom63 = _sma(self.resid, 63)
        return (atr_exp * (mom63 > 0).astype(float)).fillna(0.0)

    def _factor_ma_cloud(self) -> "xr.DataArray":
        ma50 = _sma(self.close, 50)
        ma200 = _sma(self.close, 200)
        atr20 = _atr(self.high, self.low, self.close, 20)
        z50 = (self.close - ma50) / (atr20 + self.params.eps)
        z200 = (self.close - ma200) / (atr20 + self.params.eps)
        return (0.6 * z50 + 0.4 * z200).fillna(0.0)

    # --------------------------
    # Idio-quality / microstructure
    # --------------------------

    def _factor_idio_tail_risk(self) -> "xr.DataArray":
        resid = self.resid.fillna(0.0)
        sigma20 = _std(resid, 20).fillna(0.0)
        # rolling 5% quantile via rolling.reduce
        q05 = resid.rolling(time=20, min_periods=10).reduce(np.nanquantile, q=0.05)
        return (-(q05 / (sigma20 + self.params.eps))).fillna(0.0)

    def _factor_idio_jump_freq(self) -> "xr.DataArray":
        resid = self.resid.fillna(0.0)
        sigma63 = _std(resid, 63).fillna(0.0)
        thr = 2.5 * (sigma63 + self.params.eps)
        jumps = (np.abs(resid) > thr).astype(float)
        return (-(jumps.rolling(time=20, min_periods=20).mean())).fillna(0.0)

    def _factor_beta_stability(self) -> "xr.DataArray":
        beta21 = _sma(self.beta.fillna(0.0), 21).fillna(0.0)
        return (-np.abs(beta21 - beta21.shift(time=21))).fillna(0.0)

    def _factor_micro_noise(self) -> "xr.DataArray":
        # close-to-close var vs Parkinson var proxy
        cc = _clip_ret(self.close / self.close.shift(time=1) - 1.0)
        var_cc = _sma(cc * cc, 20)

        hl = (self.high / (self.low + self.params.eps)).clip(min=1.0)
        ln_hl2 = _sma((xr.apply_ufunc(np.log, hl) ** 2), 20)
        var_pk = (1.0 / (4.0 * np.log(2.0))) * ln_hl2
        return (-(var_cc / (var_pk + self.params.eps))).fillna(0.0)

    # --------------------------
    # Interactions
    # --------------------------

    def _factor_value_mom(self) -> "xr.DataArray":
        # original: resid_mom * srev
        return (self._factor_resid_mom() * self._factor_srev()).fillna(0.0)

    def _factor_quality_defensive(self) -> "xr.DataArray":
        return (self._factor_liquidity() * self._factor_inv_vol()).fillna(0.0)

    # --------------------------
    # Sector-relative proxy
    # --------------------------

    def _factor_rel_sector_mom(self) -> "xr.DataArray":
        # original approximation: (asset mom - "sector" mom proxy),
        # here proxy = cross-sectional mean mom
        if self.ret.sizes.get("time", 0) < self.params.SECTOR_MOM_WIN + 2:
            return xr.zeros_like(self.ret)

        asset_mom = _sma(self.ret, self.params.SECTOR_MOM_WIN)
        proxy = asset_mom.mean("asset")
        return (asset_mom - proxy.broadcast_like(asset_mom)).fillna(0.0)

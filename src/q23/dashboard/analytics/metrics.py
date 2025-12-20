from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def summary_exposure(w_last: pd.Series) -> Dict[str, float]:
    gross = float(np.abs(w_last).sum())
    net = float(w_last.sum())
    long = float(w_last[w_last > 0].sum())
    short = float(w_last[w_last < 0].sum())
    return {
        "gross": gross,
        "net": net,
        "long": long,
        "short": short,
        "n_pos": int((w_last.abs() > 1e-12).sum()),
        "herfindahl": float((w_last ** 2).sum()),
    }


def top_positions(w_last: pd.Series, n: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    longs = w_last[w_last > 0].sort_values(ascending=False).head(n)
    shorts = w_last[w_last < 0].sort_values(ascending=True).head(n)
    return longs.rename("weight").to_frame(), shorts.rename("weight").to_frame()


def turnover_series(weights: pd.DataFrame) -> pd.Series:
    d = weights.fillna(0.0).diff().abs().sum(axis=1)
    if len(d) > 0:
        d.iloc[0] = 0.0
    d.name = "turnover"
    return d


def select_return_series(diag: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if diag is None or diag.empty:
        return None

    candidates = ["port_ret_net_tc", "port_ret", "active_ret_net_tc", "active_ret"]
    for c in candidates:
        if c in diag.columns:
            s = diag[c].astype(float).copy()
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s[~s.index.isna()].sort_index()
            return s
    return None


def period_return(s: pd.Series, start: pd.Timestamp) -> float:
    s2 = s[s.index >= start]
    if s2.empty:
        return float("nan")
    cum = (1.0 + s2).cumprod() - 1.0
    return float(cum.iloc[-1])


def period_windows(diag: Optional[pd.DataFrame], last_dt: pd.Timestamp) -> Dict[str, float]:
    out: Dict[str, float] = {}
    s = select_return_series(diag)
    if s is None or s.empty:
        return out

    last_dt = pd.Timestamp(last_dt)
    w_start = last_dt - pd.Timedelta(days=7)
    m_start = last_dt - pd.DateOffset(months=1)
    y_start = last_dt - pd.DateOffset(years=1)

    out["wtd"] = period_return(s, w_start)
    out["mtd"] = period_return(s, m_start)
    out["ytd"] = period_return(s, y_start)
    return out


def drawdown(ret_series: pd.Series) -> pd.Series:
    if ret_series is None or ret_series.empty:
        return pd.Series(dtype=float)
    perf = (1.0 + ret_series).cumprod()
    peak = perf.cummax()
    dd = perf / peak - 1.0
    dd.name = "drawdown"
    return dd


def compute_trade_list(
    w: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, min_abs_delta: float = 0.002
) -> pd.DataFrame:
    w0 = w.loc[t0].fillna(0.0)
    w1 = w.loc[t1].fillna(0.0)
    delta = (w0 - w1).astype(float)
    trades = pd.DataFrame({"prev_w": w1, "target_w": w0, "delta_w": delta})
    trades = trades[np.abs(trades["delta_w"]) >= float(min_abs_delta)].copy()
    trades["action"] = np.where(trades["delta_w"] > 0, "BUY", "SELL")
    trades["abs_delta"] = trades["delta_w"].abs()
    trades = trades.sort_values("abs_delta", ascending=False).drop(columns=["abs_delta"])
    trades.index.name = "symbol"
    return trades


def rolling_sharpe(ret_series: pd.Series, window: int = 252) -> pd.Series:
    if ret_series is None or ret_series.empty:
        return pd.Series(dtype=float)
    mean_ret = ret_series.rolling(window).mean()
    std_ret = ret_series.rolling(window).std()
    sharpe = (mean_ret / (std_ret + 1e-12)) * np.sqrt(252)
    sharpe.name = "rolling_sharpe"
    return sharpe


def hit_rate(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    w_lag = weights.shift(1).fillna(0.0)
    contrib = (w_lag * returns.fillna(0.0)).fillna(0.0)

    hits = (contrib > 0).sum(axis=1)
    total = (w_lag.abs() > 1e-12).sum(axis=1)
    hit_rate_series = (hits / total.replace(0, np.nan)).fillna(0.0)
    hit_rate_series.name = "hit_rate"
    return hit_rate_series


def winner_loser_sizing(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    w_lag = weights.shift(1).fillna(0.0)
    contrib = (w_lag * returns.fillna(0.0)).fillna(0.0)

    winners = contrib > 0
    losers = contrib < 0

    avg_winner_weight = w_lag.where(winners).abs().mean(axis=1)
    avg_loser_weight = w_lag.where(losers).abs().mean(axis=1)

    df = pd.DataFrame({
        "avg_winner_weight": avg_winner_weight,
        "avg_loser_weight": avg_loser_weight,
        "winner_loser_ratio": avg_winner_weight / (avg_loser_weight + 1e-12),
    })
    return df

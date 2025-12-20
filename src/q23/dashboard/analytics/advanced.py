from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from q23.shared.config import TransactionCostConfig, TransactionCostScheme
from q23.strategy.transaction_costs import (
    TransactionCostModel,
    compute_atr_pandas,
    compute_quantiacs_tc,
    compute_flat_tc,
)


class RegimeDetector:
    def __init__(self, returns: pd.Series, lookback: int = 252):
        self.returns = returns
        self.lookback = lookback

    def detect_regimes(self) -> pd.DataFrame:
        vol = self.returns.rolling(self.lookback).std() * np.sqrt(252)
        vol_ma = vol.rolling(63).mean()
        vol_z = (vol - vol_ma) / (vol.rolling(63).std() + 1e-12)

        corr = self.returns.rolling(63).corr(
            self.returns.rolling(252).mean().shift(1).fillna(0)
        )

        skew = self.returns.rolling(63).apply(lambda x: stats.skew(x), raw=True)

        regimes = pd.DataFrame(
            {
                "vol": vol,
                "vol_z": vol_z,
                "trend_corr": corr,
                "skew": skew,
                "high_vol": vol_z > 1.0,
                "trending": corr.abs() > 0.3,
                "negative_skew": skew < -0.5,
            },
            index=self.returns.index,
        )

        regimes["regime"] = "normal"
        regimes.loc[regimes["high_vol"] & regimes["negative_skew"], "regime"] = "crisis"
        regimes.loc[regimes["trending"] & ~regimes["high_vol"], "regime"] = "trend"
        regimes.loc[regimes["vol_z"] < -0.5, "regime"] = "calm"

        return regimes


class TailRiskAnalyzer:
    def __init__(self, returns: pd.Series):
        self.returns = returns

    def compute_tail_metrics(self) -> Dict[str, float]:
        rets = self.returns.dropna()
        if len(rets) < 50:
            return {}

        sorted_rets = np.sort(rets.values)
        n = len(sorted_rets)

        var_95 = np.percentile(sorted_rets, 5)
        var_99 = np.percentile(sorted_rets, 1)

        tail_95 = sorted_rets[sorted_rets <= var_95]
        tail_99 = sorted_rets[sorted_rets <= var_99]

        cvar_95 = tail_95.mean() if len(tail_95) > 0 else var_95
        cvar_99 = tail_99.mean() if len(tail_99) > 0 else var_99

        worst_day = sorted_rets[0]
        worst_week = rets.rolling(5).sum().min()
        worst_month = rets.rolling(21).sum().min()

        skewness = stats.skew(rets)
        kurtosis = stats.kurtosis(rets)

        beyond_3sigma = np.sum(np.abs(rets - rets.mean()) > 3 * rets.std())
        tail_ratio = beyond_3sigma / len(rets)

        return {
            "var_95": float(var_95),
            "var_99": float(var_99),
            "cvar_95": float(cvar_95),
            "cvar_99": float(cvar_99),
            "worst_day": float(worst_day),
            "worst_week": float(worst_week),
            "worst_month": float(worst_month),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "tail_ratio": float(tail_ratio),
        }

    def rolling_tail_risk(self, window: int = 252) -> pd.DataFrame:
        results = []
        for i in range(window, len(self.returns) + 1):
            subset = self.returns.iloc[i - window : i]
            metrics = TailRiskAnalyzer(subset).compute_tail_metrics()
            metrics["date"] = self.returns.index[i - 1]
            results.append(metrics)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).set_index("date")
        return df


class ConvexityAnalyzer:
    def __init__(self, weights: pd.DataFrame, returns: pd.DataFrame):
        self.weights = weights
        self.returns = returns

    def compute_gamma_exposure(self) -> pd.Series:
        w_lag = self.weights.shift(1).fillna(0.0)
        r = self.returns.fillna(0.0)

        w_lag_aligned, r_aligned = w_lag.align(r, join="inner", axis=0)
        w_lag_aligned = w_lag_aligned.fillna(0.0)
        r_aligned = r_aligned.fillna(0.0)

        gamma_proxy = []
        for idx in w_lag_aligned.index:
            if idx not in r_aligned.index:
                gamma_proxy.append(0.0)
                continue

            w_vec = w_lag_aligned.loc[idx].values
            r_vec = r_aligned.loc[idx].values

            valid_mask = ~(np.isnan(w_vec) | np.isnan(r_vec))
            w_vec = w_vec[valid_mask]
            r_vec = r_vec[valid_mask]

            if len(w_vec) == 0:
                gamma_proxy.append(0.0)
                continue

            pnl_linear = np.sum(w_vec * r_vec)

            r_squared = r_vec**2
            pnl_quadratic = np.sum(w_vec * r_squared)

            gamma = pnl_quadratic - pnl_linear
            gamma_proxy.append(gamma)

        return pd.Series(gamma_proxy, index=w_lag_aligned.index, name="gamma_exposure")

    def option_like_payoff_analysis(self) -> pd.DataFrame:
        w_lag = self.weights.shift(1).fillna(0.0)
        r = self.returns.fillna(0.0)

        contrib = (w_lag * r).fillna(0.0)
        port_ret = contrib.sum(axis=1)

        mkt_ret = r.mean(axis=1)

        up_days = mkt_ret > 0
        down_days = mkt_ret < 0

        upside_beta = (
            port_ret[up_days].cov(mkt_ret[up_days]) / (mkt_ret[up_days].var() + 1e-12)
            if up_days.sum() > 10
            else 1.0
        )

        downside_beta = (
            port_ret[down_days].cov(mkt_ret[down_days]) / (mkt_ret[down_days].var() + 1e-12)
            if down_days.sum() > 10
            else 1.0
        )

        asymmetry = upside_beta - downside_beta

        results = pd.DataFrame(
            {
                "port_ret": port_ret,
                "mkt_ret": mkt_ret,
                "is_up_day": up_days,
                "upside_beta": upside_beta,
                "downside_beta": downside_beta,
                "asymmetry": asymmetry,
            }
        )

        return results


class AlphaDecayAnalyzer:
    def __init__(self, scores: pd.DataFrame, returns: pd.DataFrame):
        self.scores = scores
        self.returns = returns

    def compute_ic_decay(self, max_horizon: int = 21) -> pd.DataFrame:
        results = []

        for h in range(1, max_horizon + 1):
            fwd_ret = self.returns.shift(-h)

            ic_series = []
            for date in self.scores.index:
                if date not in fwd_ret.index:
                    continue

                s = self.scores.loc[date].dropna()
                r = fwd_ret.loc[date].reindex(s.index).dropna()

                common = s.index.intersection(r.index)
                if len(common) < 10:
                    continue

                ic = s.loc[common].corr(r.loc[common])
                ic_series.append({"date": date, "horizon": h, "ic": ic})

            if ic_series:
                results.extend(ic_series)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        pivot = df.pivot_table(index="date", columns="horizon", values="ic")
        return pivot

    def optimal_holding_period(self, ic_decay: pd.DataFrame) -> int:
        if ic_decay.empty:
            return 5

        avg_ic = ic_decay.mean(axis=0)

        cumulative_ic = avg_ic.cumsum()

        marginal_ic = avg_ic.diff().fillna(avg_ic.iloc[0])

        optimal = marginal_ic[marginal_ic > 0].index[-1] if (marginal_ic > 0).any() else 5

        return int(optimal)


class CapacityEstimator:
    def __init__(self, weights: pd.DataFrame, volumes: Optional[pd.DataFrame] = None):
        self.weights = weights
        self.volumes = volumes

    def estimate_capacity(
        self, max_adv_pct: float = 0.05, days_to_build: int = 5
    ) -> Dict[str, float]:
        if self.volumes is None:
            return {"estimated_capacity_usd": float("inf"), "note": "No volume data"}

        w = self.weights.iloc[-1].abs()
        positions = w[w > 1e-12]

        if positions.empty:
            return {"estimated_capacity_usd": 0.0}

        common_symbols = positions.index.intersection(self.volumes.columns)
        if len(common_symbols) == 0:
            return {"estimated_capacity_usd": float("inf"), "note": "No volume overlap"}

        recent_vol = self.volumes[common_symbols].tail(63).mean()

        capacity_per_stock = recent_vol * max_adv_pct * days_to_build

        position_usd_normalized = positions.loc[common_symbols]

        capacity_ratios = capacity_per_stock / (position_usd_normalized + 1e-12)

        min_ratio = capacity_ratios.min()

        current_gross = w.sum()
        estimated_capacity = current_gross * min_ratio

        return {
            "estimated_capacity_usd": float(estimated_capacity),
            "current_gross": float(current_gross),
            "capacity_multiple": float(min_ratio),
            "most_constrained_stock": capacity_ratios.idxmin(),
        }


class TransactionCostAnalyzer:
    """Enhanced transaction cost analyzer supporting multiple TC schemes.
    
    Default is Quantiacs ATR model: TC = 5% × ATR(14) × |ΔPosition|
    
    Also supports:
    - Flat basis points
    - Per-asset ATR-based costs
    - TC breakdown by asset, time, position size
    - Sharpe sensitivity analysis
    """
    
    def __init__(
        self,
        weights: pd.DataFrame,
        tc_config: Optional[TransactionCostConfig] = None,
        tc_bps: float = 10.0,  # Legacy support
    ):
        """Initialize TC analyzer.
        
        Args:
            weights: Portfolio weights (time x asset)
            tc_config: Transaction cost configuration (overrides tc_bps)
            tc_bps: Legacy flat basis points (used if tc_config is None)
        """
        self.weights = weights.fillna(0.0)
        self.weights_delta = self.weights.diff().fillna(0.0)
        
        if tc_config is not None:
            self.tc_config = tc_config
        else:
            # Legacy mode: flat BPS
            self.tc_config = TransactionCostConfig(
                scheme=TransactionCostScheme.FLAT_BPS,
                flat_bps=tc_bps,
            )
        
        self.tc_model = TransactionCostModel(self.tc_config)
        
        # Computed on demand
        self._tc_costs: Optional[pd.DataFrame] = None
        self._atr: Optional[pd.DataFrame] = None

    def set_ohlc_data(
        self,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
    ) -> None:
        """Set OHLC data for ATR-based TC computation.
        
        Args:
            high: High prices (time x asset)
            low: Low prices (time x asset)
            close: Close prices (time x asset)
        """
        self._high = high
        self._low = low
        self._close = close
        self._atr = compute_atr_pandas(high, low, close, self.tc_config.atr_window)
        self._tc_costs = None  # Reset computed costs

    def compute_costs(self) -> pd.DataFrame:
        """Compute per-asset transaction costs.
        
        Returns:
            DataFrame of TC costs (time x asset)
        """
        if self._tc_costs is not None:
            return self._tc_costs
        
        if self.tc_config.scheme == TransactionCostScheme.QUANTIACS_ATR:
            if not hasattr(self, '_close') or self._close is None:
                # Fall back to flat BPS if no OHLC data
                self._tc_costs = compute_flat_tc(
                    self.weights_delta,
                    bps=self.tc_config.flat_bps,
                )
            else:
                self._tc_costs = self.tc_model.compute_costs(
                    self.weights_delta,
                    close=self._close,
                    atr=self._atr,
                    high=self._high,
                    low=self._low,
                )
        else:
            self._tc_costs = self.tc_model.compute_costs(self.weights_delta)
        
        return self._tc_costs

    def compute_implementation_shortfall(
        self,
        prices: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute total portfolio TC per day.
        
        Args:
            prices: Optional close prices (for ATR scheme)
            
        Returns:
            Series of daily TC costs
        """
        if prices is not None and not hasattr(self, '_close'):
            # Use prices as close proxy, assume H=L=C for simple ATR
            self.set_ohlc_data(prices, prices, prices)
        
        costs = self.compute_costs()
        total_costs = costs.sum(axis=1)
        total_costs.name = "tc_cost"
        
        return total_costs

    def compute_tc_breakdown_by_asset(self) -> pd.DataFrame:
        """Compute TC breakdown by asset.
        
        Returns:
            DataFrame with per-asset TC statistics
        """
        costs = self.compute_costs()
        weights_delta = self.weights_delta
        
        results = []
        for asset in costs.columns:
            if asset not in weights_delta.columns:
                continue
            
            asset_costs = costs[asset]
            asset_delta = weights_delta[asset].abs()
            
            # Only consider trading days
            trading_days = asset_delta > 1e-12
            
            results.append({
                "asset": asset,
                "total_tc": float(asset_costs.sum()),
                "avg_tc_per_trade": float(asset_costs[trading_days].mean()) if trading_days.any() else 0.0,
                "n_trades": int(trading_days.sum()),
                "total_turnover": float(asset_delta.sum()),
                "tc_per_turnover_bps": float(asset_costs.sum() / (asset_delta.sum() + 1e-12) * 10000),
            })
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("total_tc", ascending=False)
        
        return df

    def compute_tc_breakdown_by_time(self, resample: str = "ME") -> pd.DataFrame:
        """Compute TC breakdown by time period.
        
        Args:
            resample: Resampling frequency ('ME' for month-end, 'W' for weekly)
            
        Returns:
            DataFrame with TC by period
        """
        costs = self.compute_costs()
        total_costs = costs.sum(axis=1)
        turnover = self.weights_delta.abs().sum(axis=1)
        
        tc_resampled = total_costs.resample(resample).sum()
        turnover_resampled = turnover.resample(resample).sum()
        
        df = pd.DataFrame({
            "total_tc": tc_resampled,
            "total_turnover": turnover_resampled,
            "tc_per_turnover_bps": tc_resampled / (turnover_resampled + 1e-12) * 10000,
            "n_trading_days": total_costs.resample(resample).count(),
        })
        
        return df

    def compute_tc_by_position_size(self) -> pd.DataFrame:
        """Analyze TC by position size buckets.
        
        Returns:
            DataFrame with TC by position size bucket
        """
        costs = self.compute_costs()
        weights_delta = self.weights_delta
        
        # Flatten to long format
        tc_flat = costs.stack()
        delta_flat = weights_delta.stack().abs()
        
        # Filter to trading events
        mask = delta_flat > 1e-12
        tc_flat = tc_flat[mask]
        delta_flat = delta_flat[mask]
        
        if len(tc_flat) == 0:
            return pd.DataFrame()
        
        # Create size buckets
        buckets = pd.qcut(delta_flat, q=5, labels=["XS", "S", "M", "L", "XL"], duplicates="drop")
        
        results = []
        for bucket in buckets.unique():
            bucket_mask = buckets == bucket
            results.append({
                "size_bucket": str(bucket),
                "avg_delta_pct": float(delta_flat[bucket_mask].mean() * 100),
                "total_tc": float(tc_flat[bucket_mask].sum()),
                "avg_tc_bps": float(tc_flat[bucket_mask].mean() / (delta_flat[bucket_mask].mean() + 1e-12) * 10000),
                "n_trades": int(bucket_mask.sum()),
            })
        
        return pd.DataFrame(results)

    def turnover_efficiency_ratio(self, returns: pd.Series) -> float:
        """Compute turnover efficiency ratio.
        
        TER = Cumulative Return / Average Daily Turnover
        Higher is better (more return per unit of trading).
        
        Args:
            returns: Portfolio returns series
            
        Returns:
            Turnover efficiency ratio
        """
        turnover = self.weights_delta.abs().sum(axis=1)
        avg_turnover = turnover.mean()

        cum_ret = (1 + returns).prod() - 1

        if avg_turnover < 1e-12:
            return float("inf")

        ter = cum_ret / avg_turnover
        return float(ter)

    def compute_sharpe_sensitivity(
        self,
        returns: pd.Series,
        tc_bps_range: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Compute Sharpe ratio sensitivity to TC.
        
        Args:
            returns: Portfolio gross returns
            tc_bps_range: List of TC bps values to test
            
        Returns:
            DataFrame with Sharpe at each TC level
        """
        if tc_bps_range is None:
            tc_bps_range = [0, 5, 10, 15, 20, 30, 50]
        
        turnover = self.weights_delta.abs().sum(axis=1)
        
        results = []
        for bps in tc_bps_range:
            tc_cost = turnover * (bps / 10000.0)
            net_returns = returns - tc_cost.reindex(returns.index).fillna(0.0)
            
            ann_ret = float(net_returns.mean() * 252)
            ann_vol = float(net_returns.std() * np.sqrt(252))
            sharpe = ann_ret / (ann_vol + 1e-12)
            
            results.append({
                "tc_bps": bps,
                "annual_return": ann_ret,
                "annual_vol": ann_vol,
                "sharpe": sharpe,
                "total_tc_drag": float(tc_cost.sum()),
            })
        
        return pd.DataFrame(results)

    def compute_atr_tc_breakdown(self) -> Dict[str, Any]:
        """Compute ATR-specific TC breakdown.
        
        Returns:
            Dictionary with ATR TC analysis
        """
        if not hasattr(self, '_atr') or self._atr is None:
            return {"error": "No ATR data available. Call set_ohlc_data() first."}
        
        atr = self._atr
        costs = self.compute_costs()
        
        # Average ATR by asset
        avg_atr = atr.mean()
        
        # ATR normalized by price (volatility %)
        if hasattr(self, '_close') and self._close is not None:
            atr_pct = (atr / (self._close + 1e-12)).mean() * 100
        else:
            atr_pct = pd.Series(0.0, index=avg_atr.index)
        
        # High volatility assets (top quartile ATR%)
        high_vol_threshold = atr_pct.quantile(0.75)
        high_vol_assets = atr_pct[atr_pct >= high_vol_threshold].index.tolist()
        
        # TC from high-vol assets
        high_vol_tc = costs[high_vol_assets].sum().sum() if high_vol_assets else 0.0
        total_tc = costs.sum().sum()
        
        return {
            "avg_atr_pct": float(atr_pct.mean()),
            "max_atr_pct": float(atr_pct.max()),
            "min_atr_pct": float(atr_pct.min()),
            "high_vol_asset_count": len(high_vol_assets),
            "high_vol_tc_contribution": float(high_vol_tc / (total_tc + 1e-12) * 100),
            "high_vol_assets": high_vol_assets[:10],  # Top 10
            "atr_window": self.tc_config.atr_window,
            "atr_multiplier": self.tc_config.atr_multiplier,
        }

    def compare_tc_schemes(
        self,
        returns: pd.Series,
    ) -> pd.DataFrame:
        """Compare TC and Sharpe across different schemes.
        
        Args:
            returns: Portfolio gross returns
            
        Returns:
            DataFrame comparing schemes
        """
        results = []
        turnover = self.weights_delta.abs().sum(axis=1)
        
        # Current scheme
        current_tc = self.compute_costs().sum(axis=1)
        net_ret_current = returns - current_tc.reindex(returns.index).fillna(0.0)
        
        results.append({
            "scheme": f"Current ({self.tc_config.scheme.value})",
            "total_tc": float(current_tc.sum()),
            "annualized_tc_drag": float(current_tc.mean() * 252),
            "sharpe_net": float(net_ret_current.mean() * 252 / (net_ret_current.std() * np.sqrt(252) + 1e-12)),
        })
        
        # Flat BPS comparison
        for bps in [5, 10, 20]:
            tc_flat = turnover * (bps / 10000.0)
            net_ret = returns - tc_flat.reindex(returns.index).fillna(0.0)
            
            results.append({
                "scheme": f"Flat {bps} bps",
                "total_tc": float(tc_flat.sum()),
                "annualized_tc_drag": float(tc_flat.mean() * 252),
                "sharpe_net": float(net_ret.mean() * 252 / (net_ret.std() * np.sqrt(252) + 1e-12)),
            })
        
        return pd.DataFrame(results)

    def summary(self) -> Dict[str, float]:
        """Compute summary TC statistics.
        
        Returns:
            Dictionary of summary statistics
        """
        costs = self.compute_costs()
        total_costs = costs.sum(axis=1)
        turnover = self.weights_delta.abs().sum(axis=1)
        
        return {
            "scheme": self.tc_config.scheme.value,
            "total_tc": float(total_costs.sum()),
            "avg_daily_tc": float(total_costs.mean()),
            "annualized_tc_drag": float(total_costs.mean() * 252),
            "avg_daily_turnover": float(turnover.mean()),
            "tc_per_turnover_bps": float(total_costs.sum() / (turnover.sum() + 1e-12) * 10000),
            "max_daily_tc": float(total_costs.max()),
            "n_trading_days": int((turnover > 1e-12).sum()),
        }


class MicrostructureAnalyzer:
    @staticmethod
    def effective_spread_proxy(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        log_prices = np.log(prices + 1e-12)
        realized_vol = log_prices.diff().rolling(window).std()

        spread_proxy = realized_vol * 2.0

        return spread_proxy

    @staticmethod
    def price_impact_estimate(
        weights: pd.DataFrame, volumes: pd.DataFrame, constant: float = 0.1
    ) -> pd.DataFrame:
        w = weights.abs()

        vol_median = volumes.rolling(63).median().replace(0, np.nan)

        impact = constant * (w / (vol_median + 1e-12)) ** 0.5

        impact = impact.fillna(0.0)

        return impact

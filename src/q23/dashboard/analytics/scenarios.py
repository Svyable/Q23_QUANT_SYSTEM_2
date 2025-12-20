from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


class ScenarioAnalyzer:
    def __init__(self, weights: pd.DataFrame, returns: pd.DataFrame):
        self.weights = weights
        self.returns = returns

    def stress_test_scenarios(self) -> pd.DataFrame:
        scenarios = {
            "COVID_crash_2020": ("2020-02-19", "2020-03-23"),
            "taper_tantrum_2022": ("2022-01-01", "2022-06-30"),
            "volmageddon_2018": ("2018-01-26", "2018-02-09"),
            "flash_crash_2010": ("2010-05-06", "2010-05-07"),
        }

        results = []

        for scenario_name, (start, end) in scenarios.items():
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)

            w_scenario = self.weights.loc[
                (self.weights.index >= start_dt) & (self.weights.index <= end_dt)
            ]
            r_scenario = self.returns.loc[
                (self.returns.index >= start_dt) & (self.returns.index <= end_dt)
            ]

            if w_scenario.empty or r_scenario.empty:
                continue

            w_lag = w_scenario.shift(1).fillna(0.0)
            contrib = (w_lag * r_scenario.reindex_like(w_lag).fillna(0.0)).fillna(0.0)
            port_ret = contrib.sum(axis=1)

            cum_ret = (1 + port_ret).prod() - 1
            max_dd = (port_ret.cumsum() - port_ret.cumsum().cummax()).min()
            volatility = port_ret.std() * np.sqrt(252)
            worst_day = port_ret.min()

            results.append(
                {
                    "scenario": scenario_name,
                    "start": start,
                    "end": end,
                    "cum_return": cum_ret,
                    "max_dd": max_dd,
                    "volatility": volatility,
                    "worst_day": worst_day,
                }
            )

        return pd.DataFrame(results)

    def regime_conditional_performance(
        self, regime_series: pd.Series
    ) -> pd.DataFrame:
        w_lag = self.weights.shift(1).fillna(0.0)
        contrib = (w_lag * self.returns.reindex_like(w_lag).fillna(0.0)).fillna(0.0)
        port_ret = contrib.sum(axis=1)

        aligned_ret, aligned_regime = port_ret.align(regime_series, join="inner")

        regimes = aligned_regime.unique()
        results = []

        for regime in regimes:
            mask = aligned_regime == regime
            ret_regime = aligned_ret[mask]

            if len(ret_regime) < 5:
                continue

            cum_ret = (1 + ret_regime).prod() - 1
            sharpe = (
                ret_regime.mean() / (ret_regime.std() + 1e-12) * np.sqrt(252)
                if ret_regime.std() > 0
                else 0.0
            )
            max_dd = (ret_regime.cumsum() - ret_regime.cumsum().cummax()).min()

            results.append(
                {
                    "regime": regime,
                    "n_days": len(ret_regime),
                    "cum_return": cum_ret,
                    "sharpe": sharpe,
                    "max_dd": max_dd,
                    "avg_daily_ret": ret_regime.mean(),
                    "volatility": ret_regime.std() * np.sqrt(252),
                }
            )

        return pd.DataFrame(results)


class ABTestingEngine:
    @staticmethod
    def compare_strategies(
        weights_a: pd.DataFrame,
        weights_b: pd.DataFrame,
        returns: pd.DataFrame,
        labels: Tuple[str, str] = ("Strategy A", "Strategy B"),
    ) -> pd.DataFrame:
        w_a_lag = weights_a.shift(1).fillna(0.0)
        w_b_lag = weights_b.shift(1).fillna(0.0)

        r_aligned_a = returns.reindex_like(w_a_lag).fillna(0.0)
        r_aligned_b = returns.reindex_like(w_b_lag).fillna(0.0)

        contrib_a = (w_a_lag * r_aligned_a).sum(axis=1)
        contrib_b = (w_b_lag * r_aligned_b).sum(axis=1)

        cum_a = (1 + contrib_a).cumprod() - 1
        cum_b = (1 + contrib_b).cumprod() - 1

        sharpe_a = (
            contrib_a.mean() / (contrib_a.std() + 1e-12) * np.sqrt(252)
            if contrib_a.std() > 0
            else 0.0
        )
        sharpe_b = (
            contrib_b.mean() / (contrib_b.std() + 1e-12) * np.sqrt(252)
            if contrib_b.std() > 0
            else 0.0
        )

        dd_a = (contrib_a.cumsum() - contrib_a.cumsum().cummax()).min()
        dd_b = (contrib_b.cumsum() - contrib_b.cumsum().cummax()).min()

        turnover_a = weights_a.fillna(0.0).diff().abs().sum(axis=1).mean()
        turnover_b = weights_b.fillna(0.0).diff().abs().sum(axis=1).mean()

        corr = contrib_a.corr(contrib_b)

        comparison = pd.DataFrame(
            {
                labels[0]: [
                    cum_a.iloc[-1] if not cum_a.empty else 0.0,
                    sharpe_a,
                    dd_a,
                    contrib_a.std() * np.sqrt(252),
                    turnover_a,
                ],
                labels[1]: [
                    cum_b.iloc[-1] if not cum_b.empty else 0.0,
                    sharpe_b,
                    dd_b,
                    contrib_b.std() * np.sqrt(252),
                    turnover_b,
                ],
            },
            index=["Cum Return", "Sharpe", "Max DD", "Volatility", "Avg Turnover"],
        )

        comparison["Delta"] = comparison[labels[1]] - comparison[labels[0]]
        comparison["Correlation"] = [corr] + [np.nan] * 4

        return comparison


class ParameterSurfaceExplorer:
    def __init__(self, base_params: Dict, param_ranges: Dict[str, List[float]]):
        self.base_params = base_params
        self.param_ranges = param_ranges

    def grid_search_surface(
        self, eval_func, param1: str, param2: str
    ) -> pd.DataFrame:
        if param1 not in self.param_ranges or param2 not in self.param_ranges:
            raise ValueError("Parameters must be in param_ranges")

        range1 = self.param_ranges[param1]
        range2 = self.param_ranges[param2]

        results = []

        for val1 in range1:
            for val2 in range2:
                test_params = self.base_params.copy()
                test_params[param1] = val1
                test_params[param2] = val2

                try:
                    score = eval_func(test_params)
                    results.append({param1: val1, param2: val2, "score": score})
                except Exception:
                    results.append({param1: val1, param2: val2, "score": np.nan})

        df = pd.DataFrame(results)
        pivot = df.pivot(index=param1, columns=param2, values="score")

        return pivot


class RankICAnalyzer:
    @staticmethod
    def compute_rank_ic(
        scores: pd.DataFrame, forward_returns: pd.DataFrame
    ) -> pd.Series:
        ic_series = []

        for date in scores.index:
            if date not in forward_returns.index:
                continue

            s = scores.loc[date].dropna()
            r = forward_returns.loc[date].reindex(s.index).dropna()

            common = s.index.intersection(r.index)
            if len(common) < 10:
                continue

            rank_ic, _ = kendalltau(s.loc[common], r.loc[common])
            ic_series.append(rank_ic)

        return pd.Series(ic_series, index=scores.index[: len(ic_series)], name="rank_ic")

    @staticmethod
    def quintile_analysis(
        scores: pd.DataFrame, forward_returns: pd.DataFrame, n_quantiles: int = 5
    ) -> pd.DataFrame:
        results = []

        for date in scores.index:
            if date not in forward_returns.index:
                continue

            s = scores.loc[date].dropna()
            r = forward_returns.loc[date].reindex(s.index).dropna()

            common = s.index.intersection(r.index)
            if len(common) < n_quantiles * 2:
                continue

            s_common = s.loc[common]
            r_common = r.loc[common]

            quantiles = pd.qcut(s_common, q=n_quantiles, labels=False, duplicates="drop")

            for q in range(n_quantiles):
                mask = quantiles == q
                if mask.sum() == 0:
                    continue

                avg_ret = r_common[mask].mean()
                results.append({"date": date, "quintile": q + 1, "avg_return": avg_ret})

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        pivot = df.pivot_table(index="date", columns="quintile", values="avg_return")

        return pivot


class CrossSectionalMomentumTracker:
    def __init__(self, weights: pd.DataFrame):
        self.weights = weights

    def rotation_velocity(self, window: int = 21) -> pd.Series:
        w = self.weights.fillna(0.0)

        rank = w.rank(axis=1, ascending=False, method="dense")

        rank_change = rank.diff(window).abs().sum(axis=1)

        velocity = rank_change / w.shape[1]

        return velocity

    def sector_rotation_heatmap(
        self, sector_map: Dict[str, str], window: int = 63
    ) -> pd.DataFrame:
        w = self.weights.fillna(0.0)

        sector_weights = {}
        for stock, sector in sector_map.items():
            if stock in w.columns:
                if sector not in sector_weights:
                    sector_weights[sector] = w[stock]
                else:
                    sector_weights[sector] += w[stock]

        sector_df = pd.DataFrame(sector_weights)

        rolling_change = sector_df.diff(window)

        return rolling_change


class FactorTimingAnalyzer:
    def __init__(
        self, factor_weights: pd.DataFrame, factor_returns: pd.DataFrame
    ):
        self.factor_weights = factor_weights
        self.factor_returns = factor_returns

    def timing_skill_test(self) -> pd.DataFrame:
        aligned_w, aligned_r = self.factor_weights.align(
            self.factor_returns, join="inner", axis=0
        )

        if aligned_w.empty or aligned_r.empty:
            return pd.DataFrame()

        results = []

        for factor in aligned_w.columns:
            if factor not in aligned_r.columns:
                continue

            w = aligned_w[factor].shift(1).dropna()
            r = aligned_r[factor].reindex(w.index).dropna()

            common = w.index.intersection(r.index)
            if len(common) < 20:
                continue

            w_common = w.loc[common]
            r_common = r.loc[common]

            corr = w_common.corr(r_common)

            correct_sign = ((w_common > 0) & (r_common > 0)) | (
                (w_common < 0) & (r_common < 0)
            )
            hit_rate = correct_sign.mean()

            weighted_ret = (w_common * r_common).sum()

            results.append(
                {
                    "factor": factor,
                    "timing_corr": corr,
                    "hit_rate": hit_rate,
                    "weighted_return": weighted_ret,
                }
            )

        return pd.DataFrame(results)

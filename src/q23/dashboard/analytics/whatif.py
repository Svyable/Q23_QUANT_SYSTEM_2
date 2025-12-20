from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from q23.shared.math_utils import project_to_capped_simplex


@dataclass
class WhatIfParams:
    gross_target: float = 1.0
    net_target: Optional[float] = None
    max_pos: float = 0.10
    min_pos: float = 0.002
    topn_long: Optional[int] = None
    topn_short: Optional[int] = None
    tilt_alpha: float = 0.0
    weight_overlay: Dict[str, float] = None
    blacklist: List[str] = None
    pin_list: List[str] = None
    round_to: Optional[float] = None

    def __post_init__(self):
        if self.weight_overlay is None:
            self.weight_overlay = {}
        if self.blacklist is None:
            self.blacklist = []
        if self.pin_list is None:
            self.pin_list = []


@dataclass
class WhatIfResult:
    transformed_weights: pd.Series
    baseline_weights: pd.Series
    delta: pd.Series
    turnover: float
    metrics: Dict[str, float]


class WhatIfEngine:
    def __init__(self, baseline_weights: pd.Series):
        self.baseline = baseline_weights.copy()
        self.eps = 1e-12

    def apply_transform(self, params: WhatIfParams) -> WhatIfResult:
        w = self.baseline.copy()

        w = self._apply_blacklist(w, params.blacklist)
        w = self._apply_seat_selection(w, params)
        w = self._apply_overlay(w, params.weight_overlay)
        w = self._apply_constraints(w, params)
        w = self._apply_tilt(w, params.tilt_alpha)
        if params.round_to:
            w = self._apply_rounding(w, params.round_to)

        delta = w - self.baseline
        turnover = delta.abs().sum()

        metrics = self._compute_metrics(w, params)

        return WhatIfResult(
            transformed_weights=w,
            baseline_weights=self.baseline,
            delta=delta,
            turnover=turnover,
            metrics=metrics,
        )

    def _apply_blacklist(self, w: pd.Series, blacklist: List[str]) -> pd.Series:
        w = w.copy()
        for sym in blacklist:
            if sym in w.index:
                w.loc[sym] = 0.0
        return w

    def _apply_seat_selection(self, w: pd.Series, params: WhatIfParams) -> pd.Series:
        if params.topn_long is None and params.topn_short is None:
            return w

        w = w.copy()
        long_positions = w[w > 0].sort_values(ascending=False)
        short_positions = w[w < 0].sort_values(ascending=True)

        keep = set(params.pin_list)

        if params.topn_long is not None:
            top_long = long_positions.head(params.topn_long).index.tolist()
            keep.update(top_long)

        if params.topn_short is not None:
            top_short = short_positions.head(params.topn_short).index.tolist()
            keep.update(top_short)

        mask = w.index.isin(keep)
        w_filtered = w.copy()
        w_filtered[~mask] = 0.0

        return w_filtered

    def _apply_overlay(self, w: pd.Series, overlay: Dict[str, float]) -> pd.Series:
        w = w.copy()
        for sym, target_w in overlay.items():
            if sym in w.index:
                w.loc[sym] = target_w
            else:
                w = pd.concat([w, pd.Series({sym: target_w})])
        return w

    def _apply_constraints(self, w: pd.Series, params: WhatIfParams) -> pd.Series:
        long_w = w[w > 0].copy()
        short_w = w[w < 0].copy()

        long_gross = long_w.sum()
        short_gross = short_w.abs().sum()
        total_gross = long_gross + short_gross

        if params.net_target is not None:
            target_net = params.net_target
            target_long = (params.gross_target + target_net) / 2.0
            target_short = (params.gross_target - target_net) / 2.0
        else:
            target_long = long_gross / (total_gross + self.eps) * params.gross_target if total_gross > 0 else params.gross_target / 2.0
            target_short = short_gross / (total_gross + self.eps) * params.gross_target if total_gross > 0 else params.gross_target / 2.0

        if len(long_w) > 0:
            lo_long = np.where(long_w.index.isin(params.pin_list), params.min_pos, 0.0)
            hi_long = np.full(len(long_w), params.max_pos)
            long_proj = project_to_capped_simplex(
                long_w.values, T=target_long, lo=lo_long, hi=hi_long, eps=self.eps
            )
            long_w[:] = long_proj

        if len(short_w) > 0:
            short_abs = short_w.abs()
            lo_short = np.where(short_abs.index.isin(params.pin_list), params.min_pos, 0.0)
            hi_short = np.full(len(short_abs), params.max_pos)
            short_proj = project_to_capped_simplex(
                short_abs.values, T=target_short, lo=lo_short, hi=hi_short, eps=self.eps
            )
            short_w[:] = -short_proj

        w_out = pd.Series(0.0, index=w.index)
        w_out.update(long_w)
        w_out.update(short_w)

        return w_out

    def _apply_tilt(self, w: pd.Series, alpha: float) -> pd.Series:
        if alpha <= 0 or alpha >= 1:
            return w

        w_abs = w.abs()
        w_sum = w_abs.sum()
        if w_sum < self.eps:
            return w

        scores = w_abs / w_sum
        scores = scores - scores.min() + self.eps
        exp_scores = np.exp(alpha * np.log(scores + self.eps))
        exp_scores = exp_scores / (exp_scores.sum() + self.eps)

        signs = np.sign(w)
        w_tilted = signs * exp_scores * w_sum

        return w_tilted

    def _apply_rounding(self, w: pd.Series, round_to: float) -> pd.Series:
        return (w / round_to).round() * round_to

    def _compute_metrics(self, w: pd.Series, params: WhatIfParams) -> Dict[str, float]:
        gross = w.abs().sum()
        net = w.sum()
        long = w[w > 0].sum()
        short = w[w < 0].sum()
        n_pos = (w.abs() > self.eps).sum()
        herf = (w ** 2).sum()

        return {
            "gross": float(gross),
            "net": float(net),
            "long": float(long),
            "short": float(short),
            "n_positions": int(n_pos),
            "herfindahl": float(herf),
            "concentration": float(herf / (gross + self.eps)) if gross > 0 else 0.0,
            "gross_target_delta": float(gross - params.gross_target),
            "net_target_delta": float(net - params.net_target) if params.net_target is not None else 0.0,
        }


def compute_metric_deltas(whatif: WhatIfResult, baseline_metrics: Dict[str, float]) -> pd.DataFrame:
    deltas = []
    for key in whatif.metrics.keys():
        base_val = baseline_metrics.get(key, 0.0)
        new_val = whatif.metrics[key]
        delta = new_val - base_val
        deltas.append({
            "metric": key,
            "baseline": base_val,
            "whatif": new_val,
            "delta": delta,
            "delta_pct": (delta / (abs(base_val) + 1e-12)) * 100.0 if base_val != 0 else 0.0,
        })

    return pd.DataFrame(deltas)

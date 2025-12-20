from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def _align_multiple_objects(
    *objects: Union[pd.Series, pd.DataFrame],
    join: str = 'inner',
    axis: int = 0,
) -> Tuple[Union[pd.Series, pd.DataFrame], ...]:
    """
    Align multiple pandas objects to a common index.
    
    This is a utility function to work around pandas' limitation that align()
    only accepts 2 objects at a time. It chains alignments to find the common
    index across all objects.
    
    Args:
        *objects: Variable number of Series or DataFrame objects to align
        join: Type of join ('inner', 'outer', 'left', 'right')
        axis: Axis to align on (0 for index, 1 for columns)
        
    Returns:
        Tuple of aligned objects with the same index
        
    Raises:
        ValueError: If fewer than 2 objects are provided
    """
    if len(objects) < 2:
        raise ValueError("At least 2 objects required for alignment")
    
    if len(objects) == 2:
        return objects[0].align(objects[1], join=join, axis=axis)
    
    # For 3+ objects, find common index by aligning each with the first object
    # Step 1: Align first object with second to get initial common index
    aligned_first, aligned_second = objects[0].align(objects[1], join=join, axis=axis)
    common_index = aligned_first.index
    
    # Step 2: Align first object with each remaining object and find intersection
    aligned_list = [aligned_first, aligned_second]
    for obj in objects[2:]:
        _, aligned_obj = objects[0].align(obj, join=join, axis=axis)
        common_index = common_index.intersection(aligned_obj.index)
        aligned_list.append(aligned_obj)
    
    if len(common_index) == 0:
        # Return empty objects with same structure
        return tuple(obj.reindex(common_index) for obj in aligned_list)
    
    # Step 3: Reindex all objects to the common index
    return tuple(obj.reindex(common_index) for obj in aligned_list)


def compute_factor_attribution(
    factor_exposure: pd.DataFrame,
    factor_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    """
    Compute factor attribution by multiplying factor exposures with factor returns.
    
    Args:
        factor_exposure: DataFrame with factor exposures (time x factors)
        factor_returns: DataFrame with factor returns (time x factors)
        portfolio_returns: Series with portfolio returns (time) - currently unused but kept for API consistency
        
    Returns:
        DataFrame with factor contributions (time x factors)
    """
    if factor_exposure.empty or factor_returns.empty:
        return pd.DataFrame()
    
    aligned_exp, aligned_ret = factor_exposure.align(factor_returns, join='inner', axis=0)
    
    if aligned_exp.empty or aligned_ret.empty:
        return pd.DataFrame()

    factor_contrib = aligned_exp.multiply(aligned_ret, axis=0)

    return factor_contrib


def compute_rolling_factor_regression(
    active_returns: pd.Series,
    factor_exposures: pd.DataFrame,
    window: int = 252,
    min_periods: int = 126,
) -> pd.DataFrame:
    """
    Compute rolling factor regression to decompose returns into factor and idiosyncratic components.
    
    For each rolling window, performs OLS regression:
        active_returns = alpha + beta_1 * factor_1 + ... + beta_n * factor_n + epsilon
    
    Args:
        active_returns: Series of active portfolio returns (time)
        factor_exposures: DataFrame of factor exposures (time x factors)
        window: Rolling window size in periods
        min_periods: Minimum number of valid periods required for regression
        
    Returns:
        DataFrame with columns: date, alpha, r_squared, factor_explained_ret, idio_ret, beta_*
    """
    if active_returns.empty or factor_exposures.empty:
        return pd.DataFrame()
    
    aligned_ret, aligned_exp = active_returns.align(factor_exposures, join='inner', axis=0)

    if aligned_ret.empty or aligned_exp.empty:
        return pd.DataFrame()

    n_periods = len(aligned_ret)
    n_factors = aligned_exp.shape[1]
    factor_names = aligned_exp.columns.tolist()

    results = []

    for i in range(window - 1, n_periods):
        if i < min_periods - 1:
            continue

        start = max(0, i - window + 1)
        end = i + 1

        y = aligned_ret.iloc[start:end].values
        X = aligned_exp.iloc[start:end].values

        if len(y) < min_periods or np.isnan(y).all() or np.isnan(X).all():
            continue

        valid_mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        if valid_mask.sum() < min_periods:
            continue

        y_valid = y[valid_mask]
        X_valid = X[valid_mask]

        try:
            X_with_const = np.column_stack([np.ones(len(X_valid)), X_valid])
            betas = np.linalg.lstsq(X_with_const, y_valid, rcond=None)[0]

            alpha = betas[0]
            factor_betas = betas[1:]

            y_pred = X_with_const @ betas
            residuals = y_valid - y_pred
            tss = np.sum((y_valid - y_valid.mean()) ** 2)
            rss = np.sum(residuals ** 2)
            r_squared = 1 - (rss / (tss + 1e-12))

            factor_contrib = X_valid @ factor_betas
            factor_explained_ret = factor_contrib.mean()
            idio_ret = residuals.mean()

            result = {
                'date': aligned_ret.index[i],
                'alpha': alpha,
                'r_squared': r_squared,
                'factor_explained_ret': factor_explained_ret,
                'idio_ret': idio_ret,
            }

            for j, fname in enumerate(factor_names):
                result[f'beta_{fname}'] = factor_betas[j]

            results.append(result)

        except np.linalg.LinAlgError:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.set_index('date')

    return df


def compute_factor_vs_idio_decomposition(
    portfolio_returns: pd.Series,
    factor_exposures: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decompose portfolio returns into factor-explained and idiosyncratic components.
    
    The decomposition follows:
        portfolio_returns = factor_returns + idiosyncratic_returns
        where factor_returns = sum(factor_exposures * factor_returns)
    
    Args:
        portfolio_returns: Series of portfolio returns (time)
        factor_exposures: DataFrame of factor exposures (time x factors)
        factor_returns: DataFrame of factor returns (time x factors)
        
    Returns:
        DataFrame with columns:
            - total_ret: Original portfolio returns
            - factor_ret: Factor-explained returns
            - idio_ret: Idiosyncratic returns
            - factor_ret_cum: Cumulative factor returns
            - idio_ret_cum: Cumulative idiosyncratic returns
            - total_ret_cum: Cumulative total returns
    """
    # Validate inputs
    if portfolio_returns.empty or factor_exposures.empty or factor_returns.empty:
        return pd.DataFrame()
    
    # Align all three objects to common index
    # Note: pandas align() only accepts 2 objects, so we use a helper function
    try:
        aligned_ret, aligned_exp, aligned_fret = _align_multiple_objects(
            portfolio_returns, factor_exposures, factor_returns,
            join='inner', axis=0
        )
    except ValueError:
        return pd.DataFrame()
    
    # Final validation after alignment
    if aligned_ret.empty or aligned_exp.empty or aligned_fret.empty:
        return pd.DataFrame()

    # Compute factor contribution: sum of (exposure * factor_return) across all factors
    factor_contrib = aligned_exp.multiply(aligned_fret, axis=0).sum(axis=1)

    # Idiosyncratic return is the residual
    idio_ret = aligned_ret - factor_contrib

    decomp = pd.DataFrame({
        'total_ret': aligned_ret,
        'factor_ret': factor_contrib,
        'idio_ret': idio_ret,
    }, index=aligned_ret.index)

    # Compute cumulative returns
    decomp['factor_ret_cum'] = (1 + decomp['factor_ret']).cumprod() - 1
    decomp['idio_ret_cum'] = (1 + decomp['idio_ret']).cumprod() - 1
    decomp['total_ret_cum'] = (1 + decomp['total_ret']).cumprod() - 1

    return decomp


def compute_single_stock_attribution(
    symbol: str,
    factor_vectors: pd.DataFrame,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> dict:
    """
    Compute attribution metrics for a single stock.
    
    Args:
        symbol: Stock symbol to analyze
        factor_vectors: DataFrame with factor exposures per stock (asset x factors + summary cols)
        weights: DataFrame with portfolio weights over time (time x assets)
        returns: DataFrame with stock returns over time (time x assets)
        
    Returns:
        Dictionary with attribution metrics:
            - symbol: Stock symbol
            - total_pnl: Total P&L contribution
            - pnl_per_day: Average P&L per day held
            - days_held: Number of days the stock was held
            - avg_weight: Average weight over all periods
            - avg_weight_when_held: Average weight when position was non-zero
            - factor_exposures: Dictionary of factor exposures
    """
    if symbol not in weights.columns or symbol not in returns.columns:
        return {}

    w_series = weights[symbol].fillna(0.0)
    r_series = returns[symbol].fillna(0.0)

    # Align weights and returns to ensure same index
    w_series, r_series = w_series.align(r_series, join='inner')
    w_series = w_series.fillna(0.0)
    r_series = r_series.fillna(0.0)

    w_lag = w_series.shift(1).fillna(0.0)
    contrib = (w_lag * r_series).fillna(0.0)
    total_pnl = contrib.sum()

    held_mask = w_series.abs() > 1e-12
    days_held = held_mask.sum()

    avg_weight = w_series.mean()
    avg_weight_when_held = w_series[held_mask].mean() if days_held > 0 else 0.0

    pnl_per_day = total_pnl / days_held if days_held > 0 else 0.0

    factor_expo = {}
    if symbol in factor_vectors.index:
        summary_cols = {'total_pnl_contrib', 'pnl_per_day_held', 'mean_score', 
                       'score_vol', 'days_held', 'avg_weight', 'avg_weight_when_held'}
        factor_cols = [c for c in factor_vectors.columns if c not in summary_cols]
        expo_vec = factor_vectors.loc[symbol, factor_cols].to_dict()
        factor_expo = expo_vec

    return {
        'symbol': symbol,
        'total_pnl': float(total_pnl),
        'pnl_per_day': float(pnl_per_day),
        'days_held': int(days_held),
        'avg_weight': float(avg_weight),
        'avg_weight_when_held': float(avg_weight_when_held),
        'factor_exposures': factor_expo,
    }

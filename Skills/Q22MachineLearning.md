import xarray as xr
import qnt.data as qndata
import qnt.backtester as qnbt
import qnt.ta as qnta
import qnt.stats as qns
import qnt.graph as qngraph
import qnt.output as qnout
import qnt.filter as qnfilter
import qnt.exposure as qnexp
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import random

global_lookback_period = 155
global_train_period = 100
global_count_features_for_ml = 1
global_top_assets = 12


class LSTM(nn.Module):
    """
    Class to define our LSTM network.
    """

    def __init__(self, input_dim=3, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        self.lstm1 = nn.LSTMCell(input_dim, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)

    def forward(self, y):
        outputs = []
        n_samples = y.size(0)
        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)

        for time_step in range(y.size(1)):
            x_t = y[:, time_step, :]  # Ensure x_t is [batch, input_dim]

            h_t, c_t = self.lstm1(x_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1).squeeze(-1)
        return outputs


def get_model():
    def set_seed(seed_value=42):
        """Set seed for reproducibility."""
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)
    model = LSTM(input_dim=global_count_features_for_ml, hidden_layers=34)
    return model


def get_features(data):
    close_price = data.sel(field="close").ffill('time').bfill('time').fillna(1)
    close_price_lwma = qnta.lwma(close_price, 10)
    trend = qnta.roc(close_price_lwma, 1).ffill('time').bfill('time').fillna(1)
    features = xr.concat([trend], "feature")
    return features


def get_target_classes(data):
    price_current = data.sel(field='open')
    price_future = qnta.shift(price_current, -1)

    class_positive = 1  # prices goes up
    class_negative = 0  # price goes down

    target_price_up = xr.where(price_future > price_current, class_positive, class_negative)
    return target_price_up


def get_dynamic_stock_data(data):
    is_liquid = data.sel(field="is_liquid")

    # Get the assets that are liquid in the last time step
    last_assets = is_liquid[-1]
    is_liquid_asset_list = last_assets.where(last_assets > 0, drop=True).asset.values

    # Select data for liquid assets
    data_liquid = data.sel(asset=is_liquid_asset_list)

    # Determine the rolling window for volatility calculation
    rolling_window = min(global_lookback_period, len(data_liquid.time) - 1)

    # Filter the top assets by lowest volatility
    low_volatility = qnfilter.filter_volatility(
        data=data_liquid,
        rolling_window=rolling_window,
        top_assets=global_top_assets,
        metric="std",
        ascending=True
    )

    # Select data for the low volatility assets
    last_asset = low_volatility[-1]
    top_assets_indices = last_asset.where(last_asset > 0, drop=True).asset.values
    data_low_volatility_assets = data.sel(asset=top_assets_indices)

    return data_low_volatility_assets


def load_data(period):
    data = qndata.stocks.load_ndx_data(tail=period)
    return data


def train_model(data):
    data = get_dynamic_stock_data(data)
    features_all = get_features(data)
    target_all = get_target_classes(data)
    models = dict()
    asset_name_all = data.asset.values

    for asset_name in asset_name_all:
        model = get_model()
        target_cur = target_all.sel(asset=asset_name).dropna('time', 'any')
        features_cur = features_all.sel(asset=asset_name).dropna('time', 'any')
        target_for_learn_df, feature_for_learn_df = xr.align(target_cur, features_cur, join='inner')
        criterion = nn.MSELoss()
        optimiser = optim.LBFGS(model.parameters(), lr=0.08)
        epochs = 1
        for i in range(epochs):
            def closure():
                optimiser.zero_grad()
                feature_data = feature_for_learn_df.transpose('time', 'feature').values
                in_ = torch.tensor(feature_data, dtype=torch.float32).unsqueeze(0)
                out = model(in_)
                target = torch.zeros(1, len(target_for_learn_df.values))
                target[0, :] = torch.tensor(np.array(target_for_learn_df.values))
                loss = criterion(out, target)
                loss.backward()
                return loss

            optimiser.step(closure)
        models[asset_name] = model
    return models


def predict(models, data, state):
    models_name = list(models.keys())
    data = data.sel(asset=models_name)

    features_all = get_features(data)

    last_time = data.time.values[-1]
    data_last = data.sel(time=slice(last_time, None))
    features_last = features_all.sel(time=slice(last_time, None))

    weights = xr.zeros_like(data_last.sel(field='close'))
    for asset_name in models_name:
        features_cur = features_last.sel(asset=asset_name).dropna('time', 'any')
        if len(features_cur.time) < 1:
            continue
        feature_data = features_cur.transpose('time', 'feature').values
        in_ = torch.tensor(feature_data, dtype=torch.float32).unsqueeze(0)
        out = models[asset_name](in_)
        prediction = out.detach()[0]
        weights.loc[dict(asset=asset_name, time=features_cur.time.values)] = prediction

    # Apply liquidity filter
    weights = weights * data_last.sel(field="is_liquid")

    # state may be null, so define a default value
    if state is None:
        default = xr.zeros_like(data_last.sel(field='close')).isel(time=-1)
        state = {
            "previous_weights": default,
        }

    # Extract previous weights from state
    if isinstance(state, tuple):
        previous_weights = state[-1]['previous_weights']
        # created = state[0]
        # model = state[1]
        # state = state[2]
    else:
        previous_weights = state['previous_weights']

    # Combine the assets from both previous_weights and weights
    # This code is needed because the list of liquid assets may change
    all_assets = np.union1d(previous_weights.asset.values, weights.asset.values)
    # Include new assets from weights in previous_weights, set to 0 if not present
    weights = weights.reindex(asset=all_assets, fill_value=0)
    previous_weights = previous_weights.reindex(asset=all_assets, fill_value=0)

    # Ensure previous_weights includes the last time from weights
    if last_time != previous_weights.time.values:
        previous_weights = xr.concat([previous_weights, weights], dim='time')
        previous_weights = previous_weights.reindex(asset=all_assets, fill_value=0)
        # Shift previous weights by one time step to treat the past date as the current date, filling NaN values with 0
        previous_weights = previous_weights.shift(time=1).fillna(0)
        previous_weights = previous_weights.sel(time=last_time)

    # Calculate average weights
    weights_avg = xr.where(previous_weights == 0, weights, (weights + previous_weights) / 2)
    weights_avg = xr.where(weights_avg < 0.005, 0, weights_avg)

    # Normalize positions and cut big positions
    weights_sum = abs(weights_avg).sum('asset')
    weights_avg = xr.where(weights_sum > 1, weights_avg / weights_sum, weights_avg)
    weights_avg = qnexp.cut_big_positions(weights=weights_avg, max_weight=0.1)

    next_state = {
        "previous_weights": weights_avg.isel(time=-1),
    }
    #
    # print(last_time)
    # print("previous_weights")
    # print(previous_weights)
    # print(weights)
    # print("weights_avg")
    # print(weights_avg.isel(time=-1))

    return weights_avg, next_state
Copy
Multi-pass Version for Development and Testing Strategy¶
For a quick model check, we recommend starting with a short time period. Comment out this code before submitting the strategy to the competition.

weights = qnbt.backtest_ml(
    load_data=load_data,
    train=train_model,
    predict=predict,
    train_period=global_train_period,
    retrain_interval=180,
    retrain_interval_after_submit=1,
    predict_each_day=True,
    competition_type='stocks_nasdaq100',
    lookback_period=global_lookback_period,
    start_date='2020-12-30',
    build_plots=True
)
Copy
Single-pass Version for Participation in the Contest¶
This code helps submissions get processed faster in the contest. The backtest system calculates the weights for each day, while the provided function calculates weights for only one day.

In [ ]:
import gzip
import pickle

from qnt.data import get_env
from qnt.log import log_err, log_info


def state_write(state, path=None):
    if path is None:
        path = get_env("OUT_STATE_PATH", "state.out.pickle.gz")
    try:
        with gzip.open(path, 'wb') as gz:
            pickle.dump(state, gz)
        log_info("State saved: " + str(state))
    except Exception as e:
        log_err(f"Error saving state: {e}")


def state_read(path=None):
    if path is None:
        path = get_env("OUT_STATE_PATH", "state.out.pickle.gz")
    try:
        with gzip.open(path, 'rb') as gz:
            state = pickle.load(gz)
        log_info("State loaded.")
        return state
    except Exception as e:
        log_err(f"Can't load state: {e}")
        return None


state = state_read()
print(state)


# separate cell

def print_stats(data, weights):
    stats = qns.calc_stat(data, weights)
    display(stats.to_pandas().tail())
    performance = stats.to_pandas()["equity"]
    qngraph.make_plot_filled(performance.index, performance, name="PnL (Equity)", type="log")


data_train = load_data(global_train_period)
models = train_model(data_train)

data_predict = load_data(global_lookback_period)

last_time = data_predict.time.values[-1]

if last_time < np.datetime64('2006-01-02'):
    print("The first state should be None")
    state_write(None)
    state = state_read()
    print(state)

weights_predict, state_new = predict(models, data_predict, state)

print_stats(data_predict, weights_predict)
state_write(state_new)
print(state_new)
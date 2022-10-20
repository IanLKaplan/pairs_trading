from datetime import datetime
from typing import Tuple
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import pandas as pd
import numpy as np
from numpy import log
from tabulate import tabulate

from utils.find_date_index import findDateIndex
from read_market_data.MarketData import MarketData

from statsmodels.tsa.stattools import adfuller


trading_days = 252
s_and_p_data = 's_and_p_data'
start_date_str = '2007-01-03'
start_date: datetime = datetime.fromisoformat(start_date_str)
d2008_date_str = '2008-01-03'
d2008_start_date: datetime = datetime.fromisoformat(d2008_date_str)

apple_tuple: Tuple = ('AAPL', 'MPWR')
market_data = MarketData(start_date=start_date, path=s_and_p_data)

close_prices_df = market_data.get_close_data(list(apple_tuple))
close_index = close_prices_df.index

half_year = int(trading_days/2)


def normalize_df(data_df: pd.DataFrame) -> pd.DataFrame:
    min_s = data_df.min()
    max_s = data_df.max()
    norm_df = (data_df - min_s) / (max_s - min_s)
    return norm_df


def sum_df(close_df: pd.DataFrame) -> pd.DataFrame:
    sum_df = close_df[close_df.columns[0]]
    for i in range(0,len(close_df.columns)):
        sum_df = sum_df + close_df[close_df.columns[i]]
    return sum_df


def calc_mean(close_df: pd.DataFrame) -> float:
    sum = sum_df(close_df)
    ts_mean = sum / close_df.shape[1]
    mean = ts_mean.mean()
    return mean


def running_mean(close_df: pd.DataFrame, window: int) -> pd.DataFrame:
    sum = sum_df(close_df) / close_df.shape[1]
    running_mean = pd.DataFrame(pd.Series(sum).rolling(window=window).mean().iloc[window:])
    return running_mean


def running_sd(close_df: pd.DataFrame, window: int) -> pd.DataFrame:
    sum = sum_df(close_df) / close_df.shape[1]
    running_sd = pd.DataFrame(pd.Series(sum).rolling(window=window).std().iloc[window:])
    return running_sd


def compute_halflife(prices_df: pd.DataFrame) -> int:
    """
    Calculate the half-life of a mean reverting series where the series
    is a Ornstein–Uhlenbeck process
    https://quant.stackexchange.com/a/25119
    """
    sum_df = prices_df[prices_df.columns[0]]
    for i in range(0,len(prices_df.columns)):
        sum_df = sum_df + prices_df[prices_df.columns[i]]
    prices_a = sum_df.values
    prices_lag = prices_a[1:]
    prices_trunc = prices_a[0:-1]
    prices_diff = prices_trunc - prices_lag
    prices_lag_m = sm.add_constant(prices_lag)
    res = sm.OLS(prices_diff, prices_lag_m).fit()
    halflife = -log(2) / res.params[1]
    halflife = int(round(halflife,0))
    return halflife


class RegressionResult:
    def __init__(self,
                 slope: pd.DataFrame,
                 intercept: float,
                 residuals: pd.Series):
        self.slope = slope
        self.intercept = intercept
        self.residuals = residuals

def least_squares(time_series_a: pd.DataFrame, time_series_b: pd.DataFrame):
    times_series_b_const = sm.add_constant(time_series_b)
    result_ab = sm.OLS(time_series_a, times_series_b_const).fit()
    time_series_a_const = sm.add_constant(time_series_a)
    result_ba = sm.OLS(time_series_b, time_series_a_const).fit()
    stock_a = time_series_a.columns[0]
    stock_b = time_series_b.columns[0]
    slope_ab = pd.DataFrame([result_ab.params[stock_b]])
    slope_ab.columns = [stock_b]
    slope_ba = pd.DataFrame([result_ba.params[stock_a]])
    slope_ba.columns = [stock_a]
    result = result_ab
    slope = slope_ab
    if slope_ab.values[0] < slope_ba.values[0]:
        result = result_ba
        slope = slope_ba
    intercept = round(result.params['const'], 2)
    slope = round(slope, 2)
    residuals = result.resid
    rsltObj = RegressionResult(slope, intercept, residuals)
    return rsltObj


d2007_ix = 0
d2008_ix = findDateIndex(close_index, d2008_start_date)

d2007_close_df = close_prices_df.iloc[d2007_ix:d2007_ix+half_year]
d2008_close_df = close_prices_df.iloc[d2008_ix:d2008_ix+half_year]

d2007_cor = round(d2007_close_df.corr().iloc[0,1], 2)
d2008_cor = round(d2008_close_df.corr().iloc[0,1], 2)
cor_df = pd.DataFrame([d2007_cor, d2008_cor])
cor_df.index = ['2007', '2008']
cor_df.columns = [f'correlation [{close_prices_df.columns[0]},{close_prices_df.columns[1]}]']

print()
print(tabulate(cor_df, headers=[*cor_df.columns], tablefmt='fancy_grid'))

d2007_half_life = compute_halflife(d2007_close_df)
d2008_half_life = compute_halflife(d2008_close_df)

d2007_close_adj_df = close_prices_df.iloc[d2007_ix:d2007_ix + half_year + d2007_half_life]
d2008_close_adj_df = close_prices_df.iloc[d2008_ix:d2008_ix + half_year + d2008_half_life]

d2007_close_adj_norm_df = normalize_df(d2007_close_adj_df)
d2008_close_adj_norm_df = normalize_df(d2007_close_adj_df)

d2007_running_mean = running_mean(d2007_close_adj_norm_df, d2007_half_life)
d2007_close_win_df = d2007_close_adj_norm_df[d2007_half_life:]
d2007_running_sd = running_sd(d2007_close_adj_norm_df, d2007_half_life)
d2007_sd_top = d2007_running_mean + d2007_running_sd
d2007_sd_top.columns = ['StdDev Top']
d2007_sd_bottom = d2007_running_mean - d2007_running_sd
d2007_sd_bottom.columns = ['StdDev Bottom']
d2007_running_mean.columns = ['mean']

d2007_plot_df = pd.concat([d2007_close_win_df, d2007_sd_top, d2007_running_mean, d2007_sd_bottom], axis=1)
d2007_plot_df.plot(grid=True, title=f'AAPL/MPWR and mean Jan 3, 2007', figsize=(10, 6))
plt.show()

rslt = least_squares(pd.DataFrame(d2007_close_df['MPWR']), pd.DataFrame(d2007_close_df['AAPL']))

johansen_rslt = coint_johansen(d2007_close_df, 0, 1)
hedge = pd.DataFrame([johansen_rslt.evec[0,0]])
hedge.columns = ['AAPL']
critical_vals = pd.DataFrame(johansen_rslt.trace_stat_crit_vals[0]).transpose()
critical_vals.columns = ['90', '95', '99']
trace_stat = johansen_rslt.trace_stat[0]

adf_result = adfuller(rslt.residuals)

stationary_a = d2007_close_df['MPWR'].values - hedge.values * d2007_close_df['AAPL'].values
stationary_df = pd.DataFrame(stationary_a.flatten())
stationary_df.index = d2007_close_df.index
stationary_df.plot(grid=True, title=f'stationary time series', figsize=(10, 6))
stat_mean = stationary_df.mean()[0]
stat_sd = stationary_df.std()[0]
plt.axhline(y=stat_mean, color='black', linewidth=2)
plt.axhline(y=stat_mean + stat_sd, color='red', linewidth=1, linestyle='--')
plt.axhline(y=stat_mean - stat_sd, color='green', linewidth=1, linestyle='--')
plt.show()


pass

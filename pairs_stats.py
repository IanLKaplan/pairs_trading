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

johansen_rslt_ab = coint_johansen(d2007_close_df, 0, 1)
hedge_ab = pd.DataFrame([johansen_rslt_ab.evec[1, 0]/johansen_rslt_ab.evec[0, 0]])
hedge_ab.columns = [d2007_close_df.columns[1]]
critical_vals_ab = pd.DataFrame(johansen_rslt_ab.trace_stat_crit_vals[0]).transpose()
critical_vals_ab.columns = ['10%', '5%', '1%']
trace_stat_ab = johansen_rslt_ab.trace_stat[0]

t_df = d2007_close_df.iloc[:, ::-1]
johansen_rslt_ba = coint_johansen(t_df, 0, 1)
hedge_ba = pd.DataFrame([johansen_rslt_ba.evec[1, 0]/johansen_rslt_ba.evec[0, 0]])
hedge_ba.columns = [t_df.columns[1]]
critical_vals_ba = pd.DataFrame(johansen_rslt_ba.trace_stat_crit_vals[0]).transpose()
critical_vals_ba.columns = ['10%', '5%', '1%']
critical_vals_dict = critical_vals_ba.to_dict('records')[0]
trace_stat_ba = johansen_rslt_ba.trace_stat[0]

adf_result = adfuller(rslt.residuals)

stationary_a = d2007_close_df['MPWR'].values - hedge_ab.values * d2007_close_df['AAPL'].values
stationary_df = pd.DataFrame(stationary_a.flatten())
stationary_df.index = d2007_close_df.index
stationary_df.plot(grid=True, title=f'stationary time series', figsize=(10, 6))
stat_mean = stationary_df.mean()[0]
stat_sd = stationary_df.std()[0]
plt.axhline(y=stat_mean, color='black', linewidth=2)
plt.axhline(y=stat_mean + stat_sd, color='red', linewidth=1, linestyle='--')
plt.axhline(y=stat_mean - stat_sd, color='green', linewidth=1, linestyle='--')
plt.show()


class CointData:
    """
    cointegrated: boolean - true if the time series were cointegrated, false otherwise
    weight: float - in the stationary cointegrtion equation x = A - w * B
    confidence: int - confidence level, in percent: 90, 95, 99
    asset_a: str - the symbol for the time series A in the above equation
    asset_b: str - the symbol for the time series B in the above equation
    """
    def __init__(self,
                 cointegrated: bool,
                 confidence: int,
                 weight: float,
                 asset_a: str,
                 asset_b: str):
        self.cointegrated = cointegrated
        self.confidence = confidence
        self.weight = weight
        self.asset_a = asset_a
        self.asset_b = asset_b

    def __str__(self):
        if self.cointegrated:
            s = f'cointegrated: {self.cointegrated}, confidence: {self.confidence}, weight: {self.weight}, ({self.asset_a}, {self.asset_b})'
        else:
            s = f'cointegrated: {self.cointegrated}'
        return s


class PairStatistics:
    def __init__(self):
        self.decimals = 2
        pass

    def correlation(self, data_a: pd.DataFrame, data_b: pd.DataFrame) -> float:
        c = np.corrcoef(data_a, data_b)
        cor_v = round(c[0, 1], 2)
        return cor_v

    def find_interval(self, coint_stat: float, critical_vals: dict) -> Tuple[bool, int]:
        """
        :param coint_stat: the ADF statistic
        :param critical_vals: a dictionary defining the ADF intervals {'1%': -3.49, '5%': -2.89, '10%': -2.58}. The
                              dictionary values may be either positive or negative.
        :return: if the adf_stat is in the critical value range, return True and the integer value of the interval
                 (e.g., 1, 5, 10). Or False and 0
        """
        cointegrated = False
        interval_key = ''
        interval = 0
        abs_coint_stat = abs(coint_stat)
        for key, value in critical_vals.items():
            abs_value = abs(value)
            if abs_coint_stat > abs_value and abs_value > interval:
                interval = abs_value
                interval_key = key
                cointegrated = True
        key_val = int(interval_key.replace('%','')) if cointegrated else 0
        return cointegrated, key_val


    def engle_granger_coint(self, data_a: pd.DataFrame, data_b: pd.DataFrame) -> CointData:
        sym_a = data_a.columns[0]
        sym_b = data_b.columns[0]
        data_b_const = sm.add_constant(data_b)
        # x = I + b * A
        result_ab = sm.OLS(data_a, data_b_const).fit()
        data_a_const = sm.add_constant(data_a)
        # x = I + b * B
        result_ba = sm.OLS(data_b, data_a_const).fit()
        slope_ab = result_ab.params[data_b.columns[0]]
        slope_ba = result_ba.params[data_a.columns[0]]
        result = result_ab
        slope = slope_ab
        if slope_ab < slope_ba:
            t = sym_a
            sym_a = sym_b
            sym_b = t
            result = result_ba
            slope = slope_ba
        # intercept = round(result.params['const'], self.decimals)
        slope = round(slope, self.decimals)
        residuals = result.resid
        adf_result = adfuller(residuals)
        adf_stat = round(adf_result[0], self.decimals)
        critical_vals = adf_result[4]
        cointegrated, interval = self.find_interval(adf_stat, critical_vals)
        coint_data = CointData(cointegrated=cointegrated, confidence=interval, weight=slope, asset_a=sym_a, asset_b=sym_b)
        return coint_data

    def johansen_coint(self, data_a: pd.DataFrame, data_b: pd.DataFrame) -> CointData:
        ts_df = pd.concat([data_a, data_b], axis=1)
        johansen_rslt_ab = coint_johansen(ts_df, 0, 1)
        hedge_ab = pd.DataFrame([johansen_rslt_ab.evec[1, 0] / johansen_rslt_ab.evec[0, 0]])
        hedge_ab.columns = [d2007_close_df.columns[1]]
        critical_vals_ab = pd.DataFrame(johansen_rslt_ab.trace_stat_crit_vals[0]).transpose()
        critical_vals_ab.columns = ['10%', '5%', '1%']
        critical_vals_dict = critical_vals_ab.to_dict('records')[0]
        trace_stat_ab = johansen_rslt_ab.trace_stat[0]
        cointegrated, interval = self.find_interval(coint_stat=trace_stat_ab, critical_vals=critical_vals_dict)
        sym_a = data_a.columns[0]
        sym_b = data_b.columns[0]
        coint_data = CointData(cointegrated=cointegrated, confidence=interval, weight=hedge_ab, asset_a=sym_a,
                               asset_b=sym_b)
        return coint_data



pair_stat = PairStatistics()
coint_data_granger = pair_stat.engle_granger_coint(pd.DataFrame(d2007_close_df[d2007_close_df.columns[0]]), pd.DataFrame(d2007_close_df[d2007_close_df.columns[1]]) )
coint_data_johansen = pair_stat.johansen_coint(pd.DataFrame(d2007_close_df[d2007_close_df.columns[0]]), pd.DataFrame(d2007_close_df[d2007_close_df.columns[1]]) )

pass

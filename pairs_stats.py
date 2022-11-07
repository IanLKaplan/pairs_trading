from datetime import datetime
from typing import Tuple
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
from numpy import log
from tabulate import tabulate

from utils.find_date_index import findDateIndex
from read_market_data.MarketData import MarketData

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller


def normalize_df(data_df: pd.DataFrame) -> pd.DataFrame:
    min_s = data_df.min()
    max_s = data_df.max()
    norm_df = (data_df - min_s) / (max_s - min_s)
    return norm_df


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
        self.intercept = np.NAN


    def __str__(self):
        if self.has_intercept():
            s = f'cointegrated: {self.cointegrated}, confidence: {self.confidence}, weight: {self.weight}, intercept: {self.get_intercept()} ({self.asset_a}, {self.asset_b})'
        else:
            s = f'cointegrated: {self.cointegrated}, confidence: {self.confidence}, weight: {self.weight}, ({self.asset_a}, {self.asset_b})'
        return s

    def set_intercept(self, intercept: float) -> None:
        self.intercept = intercept

    def get_intercept(self) -> float:
        return self.intercept

    def has_intercept(self) -> bool:
        return not np.isnan(self.intercept)


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
        intercept = round(result.params['const'], self.decimals)
        slope = round(slope, self.decimals)
        residuals = result.resid
        adf_result = adfuller(residuals)
        adf_stat = round(adf_result[0], self.decimals)
        critical_vals = adf_result[4]
        cointegrated, interval = self.find_interval(adf_stat, critical_vals)
        coint_data = CointData(cointegrated=cointegrated, confidence=interval, weight=slope, asset_a=sym_a, asset_b=sym_b)
        coint_data.set_intercept(intercept=intercept)
        return coint_data

    def johansen_coint(self, data_a: pd.DataFrame, data_b: pd.DataFrame) -> CointData:
        ts_df = pd.concat([data_a, data_b], axis=1)
        johansen_rslt = coint_johansen(ts_df, 0, 1)
        hedge_val = round(np.abs(johansen_rslt.evec[0, 0] / johansen_rslt.evec[1, 0]), 2)
        critical_vals = pd.DataFrame(johansen_rslt.trace_stat_crit_vals[0]).transpose()
        critical_vals.columns = ['10%', '5%', '1%']
        critical_vals_dict = critical_vals.to_dict('records')[0]
        trace_stat = johansen_rslt.trace_stat[0]
        cointegrated, interval = self.find_interval(coint_stat=trace_stat, critical_vals=critical_vals_dict)
        sym_a = data_a.columns[0]
        sym_b = data_b.columns[0]
        coint_data = CointData(cointegrated=cointegrated, confidence=interval, weight=hedge_val, asset_a=sym_a,
                               asset_b=sym_b)
        return coint_data

    def compute_halflife(self, z_df: pd.DataFrame) -> int:
        """
        Calculate the half-life of a mean reverting stationary series where the series
        is an Ornstein–Uhlenbeck process
        From example 7.5 in Quantitative Trading by Ernest P. Chan
        """
        prevz = z_df.shift()
        dz = z_df - prevz
        dz = dz[1:]
        prevz = prevz[1:]
        result = sm.OLS(dz, prevz - np.mean(prevz)).fit()
        theta = result.params
        halflife_f = -np.log(2) / theta
        halflife = round(halflife_f, 0)
        return int(halflife)

    def stationary_series(self, data_a: pd.DataFrame, data_b: pd.DataFrame, coint_data: CointData) -> np.ndarray:
        """
        compute the stationary time series x = A - w * B  or x = A - i - w * B if there is an intercept i
        """
        if coint_data.has_intercept():
            stationary_a = data_a.values - coint_data.get_intercept() - coint_data.weight * data_b.values
        else:
            stationary_a = data_a.values - coint_data.weight * data_b.values
        return stationary_a



trading_days = 252
s_and_p_data = 's_and_p_data'
start_date_str = '2007-01-03'
start_date: datetime = datetime.fromisoformat(start_date_str)
d2008_date_str = '2008-01-03'
d2008_start_date: datetime = datetime.fromisoformat(d2008_date_str)

apple_tuple: Tuple = ('AAPL', 'MPWR')
market_data = MarketData(start_date=start_date, path=s_and_p_data)

close_prices_df = market_data.get_close_data(['AAPL', 'MPWR', 'YUM'])
close_index = close_prices_df.index

half_year = int(trading_days/2)

d2007_ix = 0

d2007_close = pd.DataFrame(close_prices_df[['AAPL', 'MPWR', 'YUM']]).iloc[d2007_ix:d2007_ix+half_year]

d2007_cor = round(d2007_close.corr().iloc[0,[1,2]], 2)
cor_df = pd.DataFrame([d2007_cor])
cor_df.index = ['2007']

print()
print(tabulate(cor_df, headers=[*cor_df.columns], tablefmt='fancy_grid'))

d2007_aapl = pd.DataFrame(d2007_close['AAPL'])
d2007_mpwr = pd.DataFrame(d2007_close['MPWR'])
d2007_yum = pd.DataFrame(d2007_close['YUM'])

pair_stat = PairStatistics()

coint_data_granger_aapl_mpwr = pair_stat.engle_granger_coint(data_a=d2007_aapl, data_b=d2007_mpwr)
coint_data_granger_aapl_yum = pair_stat.engle_granger_coint(data_a=d2007_aapl, data_b=d2007_yum)

print(f'Granger test for cointegration (AAPL/MPWR): {coint_data_granger_aapl_mpwr}')
print(f'Granger test for cointegration (AAPL/YUM): {coint_data_granger_aapl_yum}')

coint_data_johansen_aapl_mpwr = pair_stat.johansen_coint(data_a=d2007_aapl, data_b=d2007_mpwr )
coint_data_johansen_aapl_yum = pair_stat.johansen_coint(data_a=d2007_aapl, data_b=d2007_yum )

print(f'Johansen test for cointegration (AAPL/MPWR): {coint_data_johansen_aapl_mpwr}')
print(f'Johansen test for cointegration (AAPL/YUM): {coint_data_johansen_aapl_yum}')


data_a = pd.DataFrame(d2007_close[coint_data_granger_aapl_mpwr.asset_a])
data_b = pd.DataFrame(d2007_close[coint_data_granger_aapl_mpwr.asset_b])

stationary_a = pair_stat.stationary_series(data_a=data_a, data_b=data_b, coint_data=coint_data_granger_aapl_mpwr)



stationary_df = pd.DataFrame(stationary_a.flatten())
stationary_df.index = d2007_close.index

stationary_df.columns = ['Stationary Time Series']
stationary_df.plot(grid=True, title=f'stationary time series AAPL/MPWR', figsize=(10, 6))
stat_mean = stationary_df.mean()[0]
stat_sd = stationary_df.std()[0]
plt.axhline(y=stat_mean, color='black', linewidth=2)
plt.axhline(y=stat_mean + stat_sd, color='red', linewidth=1, linestyle='--')
plt.axhline(y=stat_mean - stat_sd, color='green', linewidth=1, linestyle='--')
plt.show()

half_life = pair_stat.compute_halflife(stationary_df)
print(f'half life: {half_life}')


pass

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# <h2>
# Pairs Trading
# </h2>
# <p>
# The Jupyter notebook explores pairs trading stratagies.
# This document is available on the GitHub repository https://github.com/IanLKaplan/pairs_trading
# </p>
# <blockquote>
# <p>
# Pairs trading is an approach that takes advantage of the
# mispricing between two (or more) co-moving assets, by
# taking a long position in one (many) and shorting the
# other(s), betting that the relationship will hold and that
# prices will converge back to an equilibrium level.
# </p>
# <p>
# <i>Definitive Guide to Pairs Trading</i> availabel from <a href="https://hudsonthames.org/">Hudson and Thames</a>
# </p>
# </blockquote>
# <p>
# Pairs trading is sometimes referred to as a statistical arbitrage trading strategy.
# </p>
# <blockquote>
# <p>
# Statistical arbitrage and pairs trading tries to solve this problem using price relativity. If two assets share the same
# characteristics and risk exposures, then we can assume that their behavior would be similar as well. This has
# the benefit of not having to estimate the intrinsic value of an asset but rather just if it is under or overvalued
# relative to a peer(s). We only have to focus on the relationship between the two, and if the spread happens
# to widen, it could be that one of the securities is overpriced, the other is underpriced, or the mispricing is a
# combination of both.
# </p>
# <p>
# <i>Definitive Guide to Pairs Trading</i> availabel from <a href="https://hudsonthames.org/">Hudson and Thames</a>
# </p>
# </blockquote>
# <p>
# Pairs trading algorithms have been reported to yield portfolios with Sharpe ratios in excess of 1.0 and returns of 10% or
# higher. Pairs trading takes both long and short positions, so the portfolio tends to be market neutral. A pairs trading portfolio
# can have drawdowns, but the drawdowns should be less than a benchmark like the S&P 500 because of the market neutral nature of the
# portfolio.
# </p>
# <p>
# Markets tend toward efficiency and many quantitative approaches fade over time as they are adopted by hedge funds. Pairs trading
# goes back to the mid-1980s. Surprisingly, the approach still seems to be profitable. One reason for this could be that there are a vast
# number of possible pairs and the pairs portfolio's tend to be fairly small (5 to 20 pairs, in most cases). This could
# leave unexploited pairs in the market. Pairs trading may also be difficult to scale to a level that would be attractive to institutional
# traders, like hedge funds, so the strategy has not been arbitraged out of the market.
# </p>
# <p>
# Mathematical finance often uses models that are based on normal distributions, constant means and standard deviations. Actual market
# data is often not normally distributed and changes constantly. The statistics used to select stocks for pairs trading makes an assumption
# that the pair distribution has a constant mean and standard deviation. This assumption holds, at best, over a window of time.
# </p>
# <p>
# The statistics that predict a successful pair will not be accurate in all time periods. For the strategy to be successful, the predicition
# must be right more often than not. To minimize the risk in any particular trade, this suggests that trading a larger portfolio will
# be more successful than trading a small portfolio.
# </p>
# <h3>
# Overview
# </h3>
# <p>
# The primary references used for this notebook are the books <i>Pairs Trading</i> by Ganapathy Vidyamurthy and <i>Algorithmic
# Trading</i> by Ernest P. Chan.
# </p>
# <p>
# The pairs trading strategy attempts to find a pair of stocks that, together, form a mean reverting time series.
# </p>
# <p>
# Implementing the pairs trading strategy involves two logical steps:
# </p>
# <ol>
# <li>
# <p>
# Pairs selection: Identify a pair of stocks that are likely to have mean reverting behavior using a lookback period.
# </p>
# </li>
# <li>
# <p>
# Trading the stocks using the long/short strategy over the trading period. This involves building a trading signal
# from the close prices of the stock pair. When the trading signal is above or below the mean at some threshold value
# a long and short position are taken in the two stocks.
# </p>
# </li>
# </ol>
# <h2>
# Pairs Selection
# </h2>
# <h3>
# S&P 500 Industry Sectors
# </h3>
# <p>
# Pairs are selected from the S&P 500 stock universe. These stocks are have a high trading volume, with a small bid-ask spread. These stocks
# are also easier to short, with lower borrowing fees.
# </p>
# <p>
# In pairs selection we are trying to find pairs that are cointegrated and have mean reverting behavior. The stock pairs should
# have some logical connection. In the book <i>Pairs Trading</i> the author discusses using factor models to select pairs with
# similar factor characteristics.
# </p>
# <p>
# Factor models are often built using company fundamental factors like earnings, corporate debt and cash flow. These factors
# tend to be generic in that many companies in completely different industry sectors may have similar fundamental factors.  When selecting pairs
# we would like to select stocks that are affected by similar market forces. For example, oil companies in the energy sector tend to be
# affected similar economic and market forces. Factors affecting companies outside the energy sector can be much more complicated.
# In many cases the factors that affect S&P 500 companies are broad economic factors which are not obviously useful in choosing pairs
# for mean reversion trading.
# </p>
# <p>
# In lieu of specific factors for most pairs, the S&P 500 industry sector is used as the set from which pairs are drawn.
# Although not perfect, industry sector will tend to select stocks with similar behavior, while reducing the universe of
# stocks from which pairs are selected.
# </p>
# <p>
# Reducing the universe of stock pairs is important because, even with modern computing power, it would be difficult
# to test all possible stock pairs in the S&P 500, since the number of pairs grows exponentially with N, the number of stocks.
# </p>
#
# \$ number \space of \space pairs = \frac{N^2 - N}{2} $
#
# <p>
# The S&P 500 component stocks (in 2022) and their related industries have been downloaded from barchart.com.  The files are included in this
# GitHub repository.
# </p>
# <p>
# The S&P 500 sectors are:
# </p>
# <ol>
# <li>
# Consumer discressionary
# </li>
# <li>
# Consumer staples
# </li>
# <li>
# Energy
# </li>
# <li>
# Financials
# </li>
# <li>
# Health care
# </li>
# <li>
# Industrials
# </li>
# <li>
# Info tech
# </li>
# <li>
# Materials
# </li>
# <li>
# Real estate
# </li>
# <li>
# Communication services
# </li>
# <li>
# Utilities
# </li>
# </ol>
# <h3>
# Stock Market Close Data
# </h3>
# <p>
# The data used to model pairs trading in this notebook uses close price data for all of the S&P 500 stocks from the start date to yesterday
# (e.g., one day in the past). In other models (see Stock Market Cash Trigger and ETF Rotation) the stock data was downloaded the first time the
# notebook was run and stored in temporary files.  In these notebooks the first notebook run incurred the initial overhead of downloading
# the data, but subsequent runs could read the data from local temporary files.
# </p>
# <p>
# Downloading all of the close price data every day would have a high overhead for the S&P 500 stocks. To avoid this, the
# data is downloaded once and stored in local files. When the notebook is run at later times, only the data between the
# end of the last date in the file and the current end date will be downloaded.
# </p>
# <p>
# There are stocks in the S&P 500 list that were listed on the stock exchange later than the start date.  These
# stocks are filtered out, so the final stock set does not include all of the S&P 500 stocks.
# </p>
# <p>
# Filtering stocks in this way can create a survivorship bias. This should not be a problem for back testing pairs trading
# algorithms through the historical time period. The purpose of this backtest is to understand the pairs trading behavior.
# The results do not depend on the stock universe, only on the pairs selected.
# </p>
#

# +

#
# To generate a python file from the notebook use jupytext:
# pip install jupytext --upgrade
# jupytext --to py pairs_trading.ipynb
#

import os
from datetime import datetime
from multiprocessing import Pool
from typing import List, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from numpy import log
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

#
# Local libraries
#
from plot_ts.plot_time_series import plot_ts, plot_two_ts
from read_market_data.MarketData import MarketData
from utils.find_date_index import findDateIndex

# Apply the default theme
sns.set_theme()

s_and_p_file = 's_and_p_sector_components/sp_stocks.csv'
s_and_p_data = 's_and_p_data'
start_date_str = '2007-01-03'
start_date: datetime = datetime.fromisoformat(start_date_str)

trading_days = 252


def read_s_and_p_stock_info(path: str) -> pd.DataFrame:
    """
    Read a file containing the information on S&P 500 stocks (e.g., the symbol, company name and sector)
    :param path: the path to the file
    :return: a DataFrame with columns Symbol, Name and Sector
    """
    s_and_p_stocks = pd.DataFrame()
    if os.access(path, os.R_OK):
        # s_and_p_socks columns are Symbol, Name and Sector
        s_and_p_stocks = pd.read_csv(s_and_p_file, index_col=0)
        new_names = [sym.replace('.', '-') for sym in s_and_p_stocks['Symbol']]
        s_and_p_stocks['Symbol'] = new_names
    else:
        print(f'Could not read file {s_and_p_file}')
    return s_and_p_stocks


def extract_sectors(stocks_df: pd.DataFrame) -> dict:
    """
    Columns in the DataFrame are Symbol,Name,Sector
    :param stocks_df:
    :return: a dictionary where the key is the sector and the value is a list of stock symbols in that sector.
    """
    sector: str = ''
    sector_l: list = list()
    stock_sectors = dict()
    for t, stock_info in stocks_df.iterrows():
        if sector != stock_info['Sector']:
            if len(sector_l) > 0:
                stock_sectors[sector] = sector_l
                sector_l = list()
            sector = stock_info['Sector']
        sector_l.append(stock_info['Symbol'])
    stock_sectors[sector] = sector_l
    return stock_sectors


def calc_pair_counts(sector_info: dict) -> pd.DataFrame:
    """
    Return a DataFrame
    :param sector_info: a dictionary where the key is the sector name and the
                        value is a list of stock symbols for that sector
    :return: a DataFrame where the index is the sector names and the columns are "num stocks" and "num pairs"
             "num stocks" is the number of stocks in the sector. "num pairs" is the number of unique pairs
             that can be formed from the set of sector stocks.  The last row is the sum of the columns.
    """
    column_label = ['num stocks', 'num pairs']
    sectors = list(sector_info.keys())
    counts_l: list = list()
    n_l: list = list()
    for sector in sectors:
        n = len(sector_info[sector])
        n_l.append(n)
        count = ((n ** 2.0) - n) / 2.0
        counts_l.append(count)
    num_stocks = sum(n_l)
    info_df = pd.DataFrame(n_l)
    info_df = pd.concat([info_df, pd.DataFrame(counts_l)], axis=1)
    info_df.columns = column_label
    sum_pairs = sum(counts_l)
    num_stocks_df = pd.DataFrame([num_stocks])
    sum_df = pd.DataFrame([sum_pairs])
    row_df = pd.concat([num_stocks_df, sum_df], axis=1)
    row_df.columns = column_label
    info_df = pd.concat([info_df, row_df], axis=0)
    sectors.append('Sum')
    info_df.index = sectors
    return info_df


# stock_info_df: a DataFrame with columns Symbol, Name, Sector
stock_info_df = read_s_and_p_stock_info(s_and_p_file)
stock_l: list = list(set(stock_info_df['Symbol']))
stock_l.sort()
market_data = MarketData(start_date=start_date, path=s_and_p_data)

# Get close prices for the S&P 500 list
close_prices_df = market_data.get_close_data(stock_l)
final_stock_list = list(close_prices_df.columns)
mask = stock_info_df['Symbol'].isin(final_stock_list)
# Some stocks were listed on the stock exchange later than start_date. In this case the stock will not
# be returned by MarketData.get_close_data(). final_stock_info has the Symbol, Name, Sector for the
# set of stocks that it was possible to obtain close prices for the date range.
final_stock_info_df = stock_info_df[mask]

sectors = extract_sectors(final_stock_info_df)
pairs_info_df = calc_pair_counts(sectors)

# -

#

# <p>
# The table below shows the number of unique pairs for each S&P 500 sector and the total number of pairs. By drawing pairs from sectors,
# rather than the whole S&P 500 set of stocks, the number of possible pairs is reduced from 124,750.
# </p>

# +
print(tabulate(pairs_info_df, headers=[*pairs_info_df.columns], tablefmt='fancy_grid'))

# An experiment to replace tabulate tables. Unfortunately, this table does not display when the notebook is examined
# on GitHub.
#
# import plotly.graph_objects as go
#
# fig = go.Figure(data=[go.Table(
#     header=dict(values=['S&P 500 sector', *pairs_info_df.columns],
#                 fill_color='paleturquoise',
#                 align='left'),
#     cells=dict(values=[pairs_info_df.index, pairs_info_df['num stocks'], pairs_info_df['num pairs']],
#                fill_color='lavender',
#                align='left'))
# ])
#
# fig.show()
# -

# <h3>
# Lookback Time Period
# </h3>
# <p>
# Pairs are selected for trading using a lookback period. The longer the lookback period (with more data points) the
# less error there will be in the selection statistics, assuming that the data is stable
# (e.g., constant mean and standard deviation).  Stock price time series are not stable over time, however. The mean and the
# standard deviation changes, as do other statistics like correlation and mean reversion.
# </p>
# <p>
# In using a lookback period to choose trading pairs we are making the assumption that the past
# will resemble the future trading period. The longer the lookback period, the less likely it is that the statistics will match
# the trading period. This creates a tension between statistical accuracy and statistics that are more likely to reflect
# the future trading period.
# </p>
# <p>
# A half year period is used for the lookback period. In practice the statistics for a year period (252 trading days) and a
# six month period (126 trading days) seem to be similar. We assume that the six month period will more accurately resemble the
# future trading period.
# </p>
#
# <h2>
# Mean Reversion
# </h2>
# <p>
# A single stock price series (or log price) is rarely stationary and mean reverting.
# In selecting stock pairs we are looking for a stock pair that, when combined, is stationary and mean reverting. A stationary time series
# is a time series that has a constant mean and standard deviation. Stock time series are constantly changing, so we are looking for
# a pair that can form a stationary and mean reverting time series over a particular time window.
# </p>
# <p>
# When a pair forms a mean reverting, stationary time series, it is referred to as a cointegrated time series.
# </p>
# <blockquote>
# <p>
# This (linear) price data combination of n different time series into one price data series is called cointegration and the resulting
# price series w.r.t. financial data is called a cointegrated pair.
# </p>
# <p>
# Cointegration Analysis of Financial Time Series Data by Johannes Steffen, Pascal Held and Rudolf Kruse, 2014
# </p>
# <p>
# https://www.inf.ovgu.de/inf_media/downloads/forschung/technical_reports_und_preprints/2014/02_2014.pdf
# </p>
# </blockquote>
# <p>
# In the equation below, <i>m</i> is a stationary mean reverting time series, P<sub>A</sub> is the price series for stock A,
# P<sub>B</sub> is the price series for stock B and β is the weight factor (for one share of stock A there will be β shares of
# stock B).
# </p>
#
# \$  m = P_A - \beta P_B $
#
# \$ m = \mu  \space when \space P_A = \beta P_B $
#
# \$ m > \mu + \delta \space when \space P_A > \beta P_B \space (short \space P_A, \space long \space P_B) $
#
# \$ m < \mu + \delta \space when \space P_A < \beta P_B \space (long \space P_A, \space sort \space P_B) $
#
# <p>
# When <i>m</i> is above the mean at some level (perhaps one standard deviation), a short position will be taken in stock A
# and a long position will be taken in stock B.  When <i>m</i> is below the mean at some level (perhaps one standard deviation)
# a long position will be taken in stock A and a short position will be taken in stock B. The position taken in stock B will be
# larger than the postion in stock A by a factor of β.
# </p>
# <p>
# In identifying a pair for pairs trading a determination is made on whether <i>m</i> is mean reverting.  The process of determining
# mean reversion will also provide the value of β.
# </p>
# <h2>
# Testing for Cointegration and Mean Reversion
# </h2>
# <p>
# Pairs are selected from a common industry sector. This means that there is a good chance that the two stocks are affected by
# similar market or economic dynamics. Once a pair with high correlation is identified, the next step is to test whether the
# pair is cointegrated and mean reverting. Two tests are commonly used to test for mean reversion:
# </p>
# <ol>
# <li>
# Engle-Granger Test: Linear Regression and the Augmented Dickey Fuller (ADF) test
# </li>
# <li>
# The Johansen Test
# </li>
# </ol>
# <p>
# Each of these tests has advantages and disadvantages, which will be discussed below.
# </p>
# <h3>
# Engle-Granger Test: Linear Regression and the Augmented Dickey Fuller Test
# </h3>
# <p>
# Two linear regressions are performed on the price series of the stock pair. The residuals of the regression with the highest
# slope are tested with the Augmented Dickey Fuller (ADF) test to determine whether mean reversion is likely.
# </p>
# <p>
# Linear regression is designed to provide a measure of the effect of a dependent variable (on the x-axis) to the independent
# variable (on the y-axis).  An example might be the body mass index (a measure of body fat) on the x-axis to the blood cholesterol
# on the y-axis.
# </p>
# <p>
# In looking for stock pairs, we pick pairs that have relatively high correlation from a common industry sector, which means that some
# process is acting on both stocks. However, the movement of one stock does not necessarily cause movement in the other stock. Also,
# both stock price series tend to be driven by an underlying random process. This means that linear regression is not perfectly suited
# for analyzing pairs. Two linear regressions are performed since we don't know which stock to pick for the dependent variable. The regression
# with the highest slope is used to analyze mean reversion and build the cointegrated time series.
# </p>
# <p>
# The result of the (Engle) Granger test is the weight value for the equation above, the linear regression intercept and
# an estimate for the mean reversion confidence.
# </p>
#

# +
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

    def correlation(self, data_a_df: pd.DataFrame, data_b_df: pd.DataFrame) -> float:
        data_a = np.array(data_a_df).flatten()
        data_b = np.array(data_b_df).flatten()
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

    def stationary_series(self, data_a: pd.DataFrame, data_b: pd.DataFrame, coint_data: CointData) -> pd.DataFrame:
        """
        compute the stationary time series x = A - w * B  or x = A - i - w * B if there is an intercept i
        """
        data_df = pd.concat([data_a, data_b], axis=1)
        data_a_df = pd.DataFrame(data_df[coint_data.asset_a])
        data_b_df = pd.DataFrame(data_df[coint_data.asset_b])
        if coint_data.has_intercept():
            stationary_a = data_a_df.values - coint_data.get_intercept() - coint_data.weight * data_b_df.values
        else:
            stationary_a = data_a_df.values - coint_data.weight * data_b_df.values
        stationary_df = pd.DataFrame(stationary_a.flatten())
        stationary_df.index = data_df.index
        return stationary_df


def plot_stationary_ts(stationary_df: pd.DataFrame, plus_delta: float, minus_delta: float, title: str) -> None:
    stationary_df.plot(grid=True, title=title, figsize=(10, 6))
    stat_mean = stationary_df.mean()[0]
    plt.axhline(y=stat_mean, color='black', linewidth=2)
    plt.axhline(y=stat_mean + plus_delta, color='red', linewidth=1, linestyle='--')
    plt.axhline(y=stat_mean - minus_delta, color='green', linewidth=1, linestyle='--')
    plt.show()


half_year = int(trading_days/2)

close_index = close_prices_df.index

d2007_ix = 0

d2007_close = pd.DataFrame(close_prices_df[['AAPL', 'MPWR', 'YUM']]).iloc[d2007_ix:d2007_ix+half_year]

d2007_cor = round(d2007_close.corr().iloc[0,[1,2]], 2)
cor_df = pd.DataFrame([d2007_cor])
cor_df.index = ['2007']

# -

# <h3>
# Example: AAPL/MPWR
# </h3>
# <p>
# AAPL (Apple Inc), MPWR (Monolithic Power Systems, Inc) are in the technology industry sector.  YUM (Yum brands is a food company in a
# different industry sector). The correlations with AAPL in the first half of 2007 are shown below.
# </p>

# +

print(tabulate(cor_df, headers=[*cor_df.columns], tablefmt='fancy_grid'))

# -

# <p>
# Pairs may have high correlation without being cointegrated.  The (Engle) Granger test shows that in the first half of 2007
# AAPL and MPWR were cointegrated at the 99% level.
# </p>

# +

pair_stat = PairStatistics()

d2007_aapl = pd.DataFrame(d2007_close['AAPL'])
d2007_mpwr = pd.DataFrame(d2007_close['MPWR'])
d2007_yum = pd.DataFrame(d2007_close['YUM'])

coint_data_granger_aapl_mpwr = pair_stat.engle_granger_coint(data_a=d2007_aapl, data_b=d2007_mpwr)
print(f'Granger test for cointegration (AAPL/MPWR): {coint_data_granger_aapl_mpwr}')

data_a = pd.DataFrame(d2007_close[coint_data_granger_aapl_mpwr.asset_a])
data_b = pd.DataFrame(d2007_close[coint_data_granger_aapl_mpwr.asset_b])

stationary_df = pair_stat.stationary_series(data_a=data_a, data_b=data_b, coint_data=coint_data_granger_aapl_mpwr)

# -

# <p>
# The condidence level represents the error percent. So 1 = 1% error or 99% confidence, 5 = 5% or 95% confidence and 10 = 10% or
# 90% confidence.
# </p>
# <p>
# The plot below shows the stationary time series formed by
# </p>
#
# \$  m = MPWR - intercept - 3.7 * AAPL $
#
# <p>
# The dotted lines are a plus one and minus one standard deviation. The intercept adjusts the time series so that the mean is zero.
# </p>

# +

std_dev = stationary_df.std()[0]
plot_stationary_ts(stationary_df=stationary_df, plus_delta=std_dev, minus_delta=std_dev, title='Granger AAPL/MPWR stationary time series, 1st half of 2007' )

# -

# <p>
# AAPL and MPWR are both technology stocks that have related businesses (MPWR's products are used by companies like Apple). We would
# expect that the two stocks might be cointegrated.
# </p>
# <p>
# The test for cointegration is performed in a lookback period. The next period is the trading period were the close prices of the
# pair are combined with the weight (and perhaps the intercept) to form what we hope will be a stationary mean reverting time series
# that can be profitably traded.
# </p>
# <p>
# The test below applies the Granger test to the second half of 2007 to see if mean reversion persisted. As the test results shows,
# this is not the case.
# </p>

# +
second_half_start = d2007_ix+half_year
d2007_aapl_2 = pd.DataFrame(close_prices_df['AAPL']).iloc[second_half_start:second_half_start+half_year]
d2007_mpwr_2 = pd.DataFrame(close_prices_df['MPWR']).iloc[second_half_start:second_half_start+half_year]

coint_data_granger_aapl_mpwr_2 = pair_stat.engle_granger_coint(data_a=d2007_aapl_2, data_b=d2007_mpwr_2)

print(f'Granger test for cointegration (AAPL/MPWR): second half of 2007 {coint_data_granger_aapl_mpwr_2}')
# -

# <p>
# The pair AAPL and YUM (Yum Brands, a food company) would not be expected to be cointegrated (although they have a surprisingly
# high correlation). As expected, the Granger test does not show cointegration and mean reversion.
# </p>

coint_data_granger_aapl_yum = pair_stat.engle_granger_coint(data_a=d2007_aapl, data_b=d2007_yum)
print(f'Granger test for cointegration (AAPL/YUM): {coint_data_granger_aapl_yum}')


# <h3>
# Correlation and Cointegration
# </h3>
# <p>
# In choosing pairs for pairs trading, we are looking for a stock pair that is influenced by similar factors. Industry sector and high
# correlation can be used as a first filter for pairs.
# </p>
# <p>
# The tests for cointegration may find that a pair with a low low correlation value is cointegrated and mean reverting. One
# example is AAPL and technology sector stock GPN (Global Payments Inc.)  For the first half of 2007, AAPL and GPN have
# a low correlation.
# </p>

# +
d2007_gpn = pd.DataFrame(close_prices_df['GPN']).iloc[d2007_ix:d2007_ix+half_year]

cor_df = pd.DataFrame([pair_stat.correlation(data_a_df=d2007_aapl, data_b_df=d2007_gpn)])
cor_df.index = ['2007']
cor_df.columns = ['Correlation AAPL/GPN']
print(tabulate(cor_df, headers=[*cor_df.columns], tablefmt='fancy_grid'))


# -

# <p>
# The normalized close prices for the two stocks in the first half of 2007 are shown below.
# </p>

# +
def normalize_df(data_df: pd.DataFrame) -> pd.DataFrame:
    min_s = data_df.min()
    max_s = data_df.max()
    norm_df = (data_df - min_s) / (max_s - min_s)
    return norm_df


d2007_aapl_norm = normalize_df(d2007_aapl)
d2007_gpn_norm = normalize_df(d2007_gpn)

plot_two_ts(data_a=d2007_aapl_norm, data_b=d2007_gpn_norm, title='Normalized AAPL/GPN first half of 2007',x_label='date', y_label='Normalized Price')
# -

#
# <p>
# The Granger test shows that AAPL and GPN are cointegrated with 99% confidence (1% error). The Johansen test (see below) also shows
# that AAPL and GPN are cointegrated with 99% confidence.
# </p>

coint_data_granger_aapl_gpn = pair_stat.engle_granger_coint(data_a=d2007_aapl, data_b=d2007_gpn)
print(f'Granger test for cointegration (AAPL/GPN)  : {coint_data_granger_aapl_gpn}')

# <p>
# The plot below shows the stationary time series formed from AAPL and GPN close prices.
# </p>

stationary_df = pair_stat.stationary_series(data_a=d2007_aapl, data_b=d2007_gpn, coint_data=coint_data_granger_aapl_gpn)
stat_sd = stationary_df.std()[0]
title='Granger AAPL/GPN stationary time series, 1st half of 2007'
plot_stationary_ts(stationary_df=stationary_df, plus_delta=stat_sd, minus_delta=stat_sd, title=title)

# <p>
# Given their low correlation and unrelated businesses (computer and media hardware vs payments) this may be an example of spurious
# cointegration. If the cointegratoin is spurious, cointegration may be more likely to break down in a future time period and the
# pair may not be profitable to trade.
# </p>
# <p>
# The Granger test for the second half of 2007 (e.g., the six months following the period above) is shown below:
# </p>

# +

d2007_gpn_2 = pd.DataFrame(close_prices_df['GPN']).iloc[second_half_start:second_half_start+half_year]
coint_data_granger_aapl_gpn_2 = pair_stat.engle_granger_coint(data_a=d2007_aapl_2, data_b=d2007_gpn_2)

print(f'Granger test for cointegration (AAPL/GPN): second half of 2007 {coint_data_granger_aapl_gpn_2}')
# -

# <p>
# The Granger test shows that there is no cointegration in the six momnth period following the first half of 2007, which reinforces the idea
# that the previous cointegration was spurious.
# </p>
# <h3>
# The Johansen Test
# </h3>
# <p>
# Unlike the Granger linear regression based test, the Johansen test can be used on more than two assets. The result is a multi-factor
# linear model for the cointegration mean reverting time series.
# </p>
# <p>
# The Johansen test uses eigenvalue decomposition for the estimation of cointegration. The Granger test has two steps: linear regression
# and the ADF test. The Johansen test is a single test that also provides the weight factor. There is no linear constant (regression intercept)
# as there is with the Granger test.
# </p>
# <p>
# The Johansen test and the Granger test do not always agree. The Johansen test is applied to AAPL/MPWR for the close prices from the first
# half of 2007. The Johansen test shows no cointegration, although the Granger test showed cointegeration at the 99% confidence level.
# </p>

coint_data_johansen_aapl_mpwr = pair_stat.johansen_coint(data_a=d2007_aapl, data_b=d2007_mpwr)
print(f'Johansen test for cointegration (AAPL/MPWR), first half of 2007 : {coint_data_johansen_aapl_mpwr}')

# <p>
# Looking that the literature, there doesn't seem to be a consensus on whether the Granger or Johansen tests are better. Some authors suggest
# using both tests, but they don't provide any emperical insight into why this is advantageous.
# </p>
#
# <h2>
# Correlation
# </h2>
# <p>
# After selecting stocks based on their industry sector, the next filter used is the pair correlation of the
# natural log of the close prices.
# </p>
# <p>
# Stocks that are strongly correlated are more likely to also exhibit mean reversion since they have similar market behavior.
# This section examines the correlation distribution for the S&P 500 sector pairs.
# </p>
#

# +


def get_pairs(sector_info: dict) -> List[Tuple]:
    """
    Return the sector stock pairs, where the pairs are selected from the S&P 500 sector.

    :param sector_info: A dictionary containing the sector info. For example:
                        energies': ['APA', 'BKR', 'COP', ...]
                       Here 'energies' is the dictionary key for the S&P 500 sector. The dictionary value is the
                       list of stocks in the sector.

    :return: A list of Tuples, where each tuple contains the symbols for the stock pair and the sector.
            For example:
              [('AAPL', 'ACN', 'information-technology'),
               ('AAPL', 'ADBE', 'information-technology'),
               ('AAPL', 'ADI', 'information-technology'),
               ('AAPL', 'ADP', 'information-technology'),
               ('AAPL', 'ADSK', 'information-technology')]
    """
    pairs_list = list()
    sectors = list(sector_info.keys())
    for sector in sectors:
        stocks = sector_info[sector]
        num_stocks = len(stocks)
        for i in range(num_stocks):
            stock_a = stocks[i]
            for j in range(i + 1, num_stocks):
                stock_b = stocks[j]
                pairs_list.append((stock_a, stock_b, sector))
    return pairs_list


def calc_pair_correlation(stock_close_df: pd.DataFrame, pair: Tuple, window: int) -> pd.Series:
    """
    Calculate the windowed correlations for a stock pair over the entire data set.
    :param stock_close_df: A data frame containing the stock close prices
    :param pair: the stock pair (e.g., a Tuple consisting of two strings for the stock symbols)
    :param window: The data window
    :return: a numpy array of windowed correlations for the pair over the entire time period.
    """
    window = int(window)
    cor_v = np.zeros(0)
    stock_a = pair[0]
    stock_b = pair[1]
    a_close = stock_close_df[stock_a]
    b_close = stock_close_df[stock_b]
    a_log_close = log(a_close)
    b_log_close = log(b_close)

    index = stock_close_df.index
    date_l: List = []
    assert len(a_log_close) == len(b_log_close)
    for i in range(0, len(a_log_close), window):
        sec_a = a_log_close[i:i + window]
        sec_b = b_log_close[i:i + window]
        c = np.corrcoef(sec_a, sec_b)
        cor_v = np.append(cor_v, c[0, 1])
        date_l.append(index[i])
    cor_s: pd.Series = pd.Series(cor_v)
    cor_s.index = date_l
    return cor_s


def calc_windowed_correlation(stock_close_df: pd.DataFrame, pairs_list: List[Tuple], window: int) -> np.array:
    """
    Calculate the windowed pair correlation over the entire time period, for all the pairs.

    :param stock_close_df: A data frame containing the stock close prices. The columns are the stock tickers.
    :param pairs_list: A list of the pairs formed from the S&P 500 sectors.
    :param window: the window over which the correlation is calculated.
    :return: A numpy array with the correlations.
    """
    window = int(window)
    all_cor_v = np.zeros(0)
    for pair in pairs_list:
        cor_v: np.array = calc_pair_correlation(stock_close_df, pair, window)
        all_cor_v = np.append(all_cor_v, cor_v)
    return all_cor_v


def display_histogram(data_v: np.array, x_label: str, y_label: str) -> None:
    num_bins = int(np.sqrt(data_v.shape[0])) * 4
    fix, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.hist(data_v, bins=num_bins, color="blue", ec="blue")
    plt.show()

lookback_window = int(trading_days / 2)
apple_tuple: Tuple = ('AAPL', 'MPWR')
apple_tuple_cor_s = calc_pair_correlation(stock_close_df=close_prices_df, pair=apple_tuple, window=lookback_window)


# -

# <p>
# The lookback period for pairs selection is six months (126 trading days). As a first step, all of the S&P 500 sector
# pairs will be tested for correlation over the lookback period.
# </p>
# <p>
# Correlation is calculated on the log of the price for each stock pair.
# </p>
# <p>
# The windowed correlation is not stable. The plot below shows the correlation between two stocks, AAPL and MPWR, over windowed
# periods from the start date.
# </p>

# +

plot_ts(data_s=apple_tuple_cor_s, title=f'correlation between {apple_tuple[0]} and {apple_tuple[1]}',
        x_label='Window Start Date', y_label=f'Correlation over {lookback_window} day window')
# -

# <p>
# Since correlation is not stable, a stock pair that is highly correlated in one time period be uncorrelated (or negatively
# correlated) in another time period.
# </p>
# <p>
# A brief digression: Adjusted for splits, Apple Inc (AAPL) was about $4.33 per share. Apple is now (2022) one of the most
# valuable companies in the world. A number of investment funds bought Apple shares and were able to beat the overall market
# for a number of years. My father invested in such a fund. Every quarter they sent out a news letter on their "market outlook".
# This was filled with blather to make fund investors think that the people managing the fund has special insight into the
# stock market. In fact, their only insight was their position in Apple shares. When Apple's share returns plateaued for a time
# so did the funds returns.
# </p>
# <p>
# Monolithic Power Systems, Inc. (MPWR) stock grew at a rate that was similar to Apple's, although their everall market
# capitalization is a faction of Apples now. I was unfamiliar with MPWR until I wrote this notebook. Bloomberg's describes
# MPWR's business as:
# </p>
# <blockquote>
# Monolithic Power Systems, Inc. designs and manufactures power management solutions. The Company provides power conversion, LED lighting, load switches, cigarette lighter adapters, chargers, position sensors, analog input, and other electrical components. Monolithic Power Systems serves customers globally.
# </blockquote>
# <p>
# The histogram below shows the aggregate distribution of the pair correlation over all half year time periods.
# </p>
#

# +


pairs_list = get_pairs(sectors)

cor_a = calc_windowed_correlation(close_prices_df, pairs_list, lookback_window)
display_histogram(cor_a, 'Correlation between pairs', 'Count')


# -

# <p>
# The pairs are selected from a common S&P industry sector, so there are a significant number of pairs that
# have a correlation above 0.75.
# </p>
# <p>
# There are a small number of pairs that have a strong negative correlation (a negative correlation -0.75 to approximately -0.9).
# Initially we look at pairs that have a strong positive correlation, but it may be unwise to ignore the negative correlations as
# well.
# </p>

# +

class WindowedCorrelationDist:
    """
    Calculate the pairs correlation by time period
    """
    def __init__(self,
                 stock_close_df: pd.DataFrame,
                 pairs_list: List[Tuple],
                 window: int,
                 cutoff: float):
        self.stock_close_df = stock_close_df
        self.window = window
        self.cutoff = cutoff
        self.pairs_list = pairs_list

    def window_correlation(self, start_ix: int) -> int:
        count = 0
        for pair in self.pairs_list:
            stock_a = pair[0]
            stock_b = pair[1]
            price_a = self.stock_close_df[stock_a].iloc[start_ix:start_ix + self.window]
            price_b = self.stock_close_df[stock_b].iloc[start_ix:start_ix + self.window]
            log_price_a = log(price_a)
            log_price_b = log(price_b)
            c = np.corrcoef(log_price_a, log_price_b)
            if c[0, 1] >= self.cutoff:
                count = count + 1
        return count

    def calc_correlation_dist(self) -> pd.Series:
        start_list = [ix for ix in range(0, self.stock_close_df.shape[0], self.window)]
        with Pool() as mp_pool:
            count_l = mp_pool.map(self.window_correlation, start_list)
        dist_s = pd.Series(count_l)
        index = self.stock_close_df.index
        dist_s.index = index[start_list]
        return dist_s


correlation_cutoff = 0.75
cor_dist_obj = WindowedCorrelationDist(stock_close_df=close_prices_df, pairs_list=pairs_list, window=lookback_window,
                                       cutoff=correlation_cutoff)
cor_dist = cor_dist_obj.calc_correlation_dist()
# -

#
# <p>
# The plot below shows the number of pairs, in a half year time period period, with a correlation above a particular cutoff.
# </p>
#

# +

spy_close_df = market_data.read_data('SPY')
spy_close_s = spy_close_df[spy_close_df.columns[0]]
spy_close_s.columns = spy_close_df.columns
cor_dist.columns = ['Correlation']
plot_two_ts(data_a=cor_dist, data_b=spy_close_s, title=f"Number of pairs with a correlation >= {correlation_cutoff}, by time period and SPY",
        x_label='Window Start Date', y_label=f'Number of pairs in the {lookback_window} day window')
# -

#
# <p>
# In the plot above, about 75% of the pairs are highly correlated around 2008. The corresponds to the 2008-2009 stock market crash
# caused by the financial crisis. This lends validity to the financial market maxim that in a market crash all assets become correlated.
# </p>
# <p>
# To the extent that correlation is a predictor for mean reversion, this also suggests that mean reversion statistics may be volatile.
# </p>
# <h3>
# Stability of Correlation
# </h3>
# <p>
# In this section I look at whether a strong correlation between pairs makes it likely that there will be a strong correlation in
# the next time period. This is an interesting statistics because correlation is related to cointegration. If correlation persists
# between periods then cointegration and mean reversion are likely to persist.
# </p>
#

# +
#
# A problem with Jupyter notebooks, especially large notebooks like this one is lack of locality.  To review:
# pairs_list - a list of all of the pairs, including sector information.
#

class SerialCorrResult:
    def __init__(self, pair: Tuple, corr_list: List[float]):
        self.pair_list = pair
        self.corr_list = corr_list


class SerialCorrelation:
    def __init__(self, stock_close_df: pd.DataFrame, pairs_list: List[Tuple], cutoff: float, window: int):
        self.stock_close_df = stock_close_df
        self.pairs_list = pairs_list
        self.cutoff = cutoff
        self.window = window

    def calc_pair_serial_correlation(self, pair) -> SerialCorrResult:
        stock_a_sym = pair[0]
        stock_b_sym = pair[1]
        stock_a_df = self.stock_close_df[stock_a_sym]
        stock_b_df = self.stock_close_df[stock_b_sym]
        corr_list = list()
        for ix in range(0, self.stock_close_df.shape[0], self.window):
            stock_a_win = log(stock_a_df.iloc[ix:ix + self.window])
            stock_b_win = log(stock_b_df.iloc[ix:ix + self.window])
            c = np.corrcoef(stock_a_win, stock_b_win)
            corr =  round(c[0, 1], 2)
            corr_list.append(corr)
        serial_corr_result = SerialCorrResult(pair, corr_list)
        return serial_corr_result

    def serial_correlation(self) -> List[SerialCorrResult]:
        # for pair in self.pairs_list:
        #     serial_corr = self.calc_pair_serial_correlation(pair)
        #     serial_corr_list.append(serial_corr)
        with Pool as mp_pool:
            serial_corr_list = mp_pool.map(self.calc_pair_serial_correlation, self.pairs_list)
        return serial_corr_list


serial_correlation = SerialCorrelation(close_prices_df, pairs_list, correlation_cutoff, half_year)
corr_obj_list = serial_correlation.serial_correlation()

# The variable you want to predict is called the dependent variable. The variable you are using to predict the
# other variable's value is called the independent variable.

independent_var = np.zeros(0)
dependent_var = np.zeros(0)
for corr_obj in corr_obj_list:
    corr_list = corr_obj.corr_list
    n = len(corr_list)
    ind_a = np.array(corr_list[0:n-1])
    dep_a = np.array(corr_list[1:n])
    independent_var = np.concatenate([independent_var, ind_a])
    dependent_var = np.concatenate([dependent_var, dep_a])


pass

# -

#
# <h2>
# References
# </h2>
# <ol>
# <li>
# <i>Pairs Trading: Quantitative Method and Analysis</i> by Ganapathy Vidyamurthy, 2004, Wiley Publishing
# </li>
# <li>
# Algorithmic Trading: Winning Strategies and Their Rationale by Ernie Chan, 2013, Wiley Publishing
# </li>
# <li>
# <a href="https://medium.com/@financialnoob/granger-causality-test-in-pairs-trading-bf2fd939e575">Granger causality test in pairs trading</a> by
# Alexander Pavlov (behind the Medium paywall)
# </li>
# <li>
# <a href="https://letianzj.github.io/cointegration-pairs-trading.html">Quantitative Trading and Systematic Investing by Letian Wang</a> This
# post includes a discussion on how the results of Johansen cointegration can be interpreted.
# </li>
# <li>
# <a href="https://www.quantrocket.com/codeload/quant-finance-lectures/quant_finance_lectures/Lecture42-Introduction-to-Pairs-Trading.ipynb.html">Introduction to Pairs Trading</a> by Delaney Mackenzie and Maxwell Margenot
# </li>
# <li>
# <p>
# <a href="https://quantdevel.com/pdf/betterHedgeRatios.pdf">Better Hedge Ratios for Spread Trading</a>, by Paul Teetor, November 2011
# </p>
# <p>
# This note discusses the problem of using ordinary least squares to produce a hedge ratio.
# </p>
# </li>
# <li>
# <a href="http://jonathankinlay.com/2019/02/pairs-trading-part-2-practical-considerations/">Pairs Trading – Part 2: Practical Considerations</a> by Jonathan Kinlay
# </li>
# <li>
# <a href="https://www.quantconnect.com/tutorials/strategy-library/intraday-dynamic-pairs-trading-using-correlation-and-cointegration-approach">Intraday Dynamic Pairs Trading using Correlation and Cointegration</a>
# </li>
# <li>
# <a href="https://bsic.it/pairs-trading-building-a-backtesting-environment-with-python/">Pairs Trading: building a backtesting environment with Python</a>
# </li>
# <li>
# <a href="https://www.sciencedirect.com/science/article/pii/S037843712100964X">Applying Hurst Exponent in pair trading strategies
# on Nasdaq 100 index</a>
# by Quynh Bui and Robert Ślepaczuk
# </li>
# <li>
# <a href="https://www.sciencedirect.com/science/article/pii/S2214845021000880">Pairs trading: is it applicable to exchange-traded funds?</a>
# </li>
# <li>
# <a href="https://hudsonthames.org/an-introduction-to-cointegration/">An Introduction to Cointegration for Pairs Trading By Yefeng Wang</a>
# </li>
# <li>
# <a href="https://www.tradelikeamachine.com/blog/cointegration-pairs-trading/part-1-using-cointegration-for-a-pairs-trading-strategy"><i>Using Cointegration for a Pairs Trading Strategy</i> Martyn Tinsley</a>
# </li>
# <li>
# <a href="https://www.inf.ovgu.de/inf_media/downloads/forschung/technical_reports_und_preprints/2014/02_2014.pdf">Cointegration Analysis
# of Financial Time Series Data by Johannes Steffen, Pascal Held and Rudolf Kruse, 2014</a>
# <li>
# <a href="https://robotwealth.com/practical-pairs-trading/">Pairs Trading on the Robot Wealth blog by Kris Longmore</a>
# </li>
# </ol>

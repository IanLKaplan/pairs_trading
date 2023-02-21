# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
import ast
import faulthandler
import os
import random
from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Dict, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from iteration_utilities import deepflatten
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate

from pairs.pairs import get_pairs
#
# Local libraries
#
from plot_ts.plot_time_series import plot_two_ts
from read_market_data.MarketData import MarketData, read_s_and_p_stock_info, extract_sectors
from s_and_p_filter import s_and_p_directory, s_and_p_stock_file
from utils import find_date_index

# <h2>
# Backtesting a Pairs Trading Strategy
# </h2>
# <p>
# This notebook is a sequel to the notebook <i>Exploratory Statistics of Pairs Trading</i>
# (https://github.com/IanLKaplan/pairs_trading/blob/master/pairs_trading.ipynb). The previous notebook explores the algorithms for selecting
# pairs and the statistics of pairs trading. This statistical exploration provides the foundation for the strategy that his backtested in
# this notebook. For a discussion of pairs trading, the algorithms used to select pairs and the background for the strategy that is
# backetested in this notebook, please see the pevious notebook.
# </p>
# <h2>
# Pairs Trading Strategy
# </h2>
# <h3>
# Shorting Stocks in Pairs Trading
# </h3>
# <p>
# This section discusses the mechanics of taking a short position in a stock, which is more complicated than taking a long position.
# </p>
# <p>
# Pairs trading is a market neutral long/short strategy where the long and short positions for a pair have approximately equal
# dollar values when the position is opened. A profit is realized when there is mean reversion in the prices of a pair.
# </p>
# <p>
# When a stock is "shorted", the stock is borrowed and then immediately sold, realizing cash.  For example, if 100 shares at a current market
# price of 10 is shorted, the brokerage account will be credited with 1000 (100 x 10).  At some point in the future, the borrowed stock must be paid
# back buy buying the stock at the current market price.  A short position is profitable when the market price of the stock goes down.
# For example, if the market price of the stock shorted at 10 goes down to 6, there is a profit of 4 per share (4 x 100 = 400).
# </p>
# <p>
# Short positions can have unlimited loss when the stock price goes up.  For example, if the market price of the 10 stock rises to
# 14 per share, there is a 400 loss on the 100 share purchase. If the stock price doubles to 20 there will be a loss of 10 per share
# or 1000 for the 100 share short.
# </p>
# <p>
# Shorting stocks is often considered a risky investment strategy because of the potential of unlimited risk in short positions. Pairs
# trading uses market neutral positions where a long position is opened that is, initially, approximately equal to the dollar value of the short position.
# The pairs that are traded are chosen from the same industry sector and are highly correlated and cointegrated. If the market price of
# the shorted stock rises, the value of the long member of the pair should rise as well. The profit in the long position will tend
# to offset the loss in the short position. This makes the pairs trading strategy much less risky than a short only strategy.
# </p>
# <p>
# When a stock is shorted the stock is borrowed. This is treated as a margin loan. The brokerage requires that the customer maintain a
# balance with liquid assets of 150 percent of the amount borrowed. This includes the proceeds of the short sale, plus 50 percent.
# For example, if 100 shares of a 10 dollar stock are shorted, the account will be credited with 1000. The account must also
# have an addition balace of 500. The margin requirement can be met with cash or highly tradable "blue chip" stocks (e.g., S&P 500 stocks).
# </p>
# <p>
# When the pairs spread crosses a threshold, a long-short position is opened in the pair. The dollar value of the long and short positions
# will be approximately equal (they will usually not be exactly equal because we are trading whole share amounts).  This involves the following
# steps:
# </p>
# <ol>
# <li>
# <p>
# Open the short position. This will result in cash from the short sale.
# </p>
# <p>
# Stock A has a price of 10. Shorting 100 shares results in 1000 in cash.
# </p>
# </li>
# <li>
# <p>
# The proceeds from the short sale are used to pay for the long position. If the cash value of the short position was less than the long
# position, some additional cash will be needed to open the long position.
# </p>
# <p>
# Stock B has a price of 20 per share. A long position is taken in 50 shares. The 1000 realized from the short of stock A is used
# to pay for the long position.
# </p>
# </li>
# </ol>
# <p>
# Interactive Brokers charges a fee for short positions is 0.25 percent or 0.25/360 percent per day that the position is held. This fee is small enough that
# it can be ignored.
# </p>
# <p>
# The pairs trading strategy will have a portfolio of short and long positions which are opened and closed as the pair spread moves.
# At any time, the aggregate value of the short positions and the long positions, plus margin cash, must be within the margin
# requirements.
# </p>
# <p>
# The long positions are used for the margin requirement. Additional cash or highly liquid stocks are required for
# the margin requirement so that the total equals 150 percent of the margin at the point where the margin (short) position
# is opened. SEC regulation T requires that there must be at
# least 25% as the prices of the stocks change. Interactive Brokers (IB) calculates the margin requirements in real time and will liquidate
# account assets that cross the Reg T margin line.
# </p>
# <p>
# If there is a liquidity deficit relative to the margin, IB will liquidate the deficit amount times 4 (ouch!)
# </p>
# <h4>
# Interactive Brokers Margin reference
# </h4>
# <ul>
# <li>
# <a href="https://www.interactivebrokers.com/en/general/education/pdfnotes/WN-UnderstandingMargin.php">Understanding Margin Webinar Notes</a>
# </li>
# </ul>
# <h3>
# Stock Price Data Issues
# </h3>
# <p>
# The backtest in this notebook uses the daily close price for the stocks. If a large number of stocks are traded (i.e., 100 stocks)
# the trading application would use the intraday price (perhaps with a 15 minute delay). The intraday prices will generally not be the same as
# the close price. The purpose of the backtest is not provide an indication of the profitability and risk of the pairs trading strategy, so
# this difference is acceptable.
# </p>
# <h3>
# In-sample and out-of-sample time periods
# </h3>
# <p>
# The pairs trading set is constructed by looking back over the past in-sample period. The out-of-sample period is the trading period.
# </p>
# <ul>
# <li>
# <p>
# In-sample period: six months (126 trading days)
# </p>
# </li>
# <li>
# <p>
# Out-of-sample (trading) period: three months (63 trading days). A 63 day period should be long enough to capture mean reversion.
# By using a relatively short out-of-sample period risk of holding pairs is reduced and the statistics for pairs selection can
# be calculated after the out-of-sample period.
# </p>
# </li>
# </ul>
# <h4>
# Strategy
# </h4>
# <p>
# Get pairs for each S&P 500 industrial sectors
# </p>
# <p>
# For each 126 day in-sample window (moving forward every 63 days):
# </p>
# <ol>
# <li>
# Select the pairs with close price series correlation greater than or equal to 0.75
# </li>
# <li>
# Select the high correlation pairs that show Granger cointegration
# </li>
# <li>
# Remove pairs that have the same stock (all stocks should be unique and be present in only one pair)
# </li>
# <li>
# Sort the pair spread time series by volatility (high to low volatility). Higher volatility (standard deviation) pairs
# are more likely to be profitable.
# </li>
# <li>
# Select N pairs from the unique pair list
# </li>
# </ol>
# <h4>
# Out-of-sample trading period
# </h4>
# <p>
# The pairs trading backtest is intended to be as close to actual trading as possible in order to understand whether this strategy is worth
# pursuing for actual trading.
# </p>
# <p>
# At the start date of the backtest, there is an investment of N dollars (e.g., 100,000). Of these funds, approximately 60,000 is used for
# long and short positions (e.g., 60,000 in short positions and 60,000 in long positions). The remaining approximately 40,000 is used to
# satisify the margin requirement.
# </p>
# <p>
# Positions are opened for whole share values.
# </p>
# <p>
# At the end of each trading period, any open positions will closed. The resulting cash is used in the next trading period.
# </p>
# <p>
# For each pair (in the N pair set) in the out-of-sample trading period:
# </p>
# <ol>
# <li>
# Calculate the current spread value from the current pair close prices.
# $ spread_t = Price{_A}{_t} - \beta Price{_B}{_t} $
# </li>
# <li>
# If
# $ spread_t >= \mu + \sigma \times 0.75 $
# then open a short position in
# $ A $
# and a long position in
# $ \beta B $
# </li>
# <li>
# If
# $ spread_t <= \mu + \sigma \times 0.75 $
# then open a long position in
# $ A $
# and a short position in
# $ \beta B $
# </li>
# <li>
# If there is an open pair position that has a spread that crosses the mean, the positions will be closed. The profit and
# loss amount will be updated.
# </li>
# </ol>
# <h4>
# Trading Period Statistics
# </h4>
# <ol>
# <li>
# Running margin values, by day. The margin increases when
# $ S_t > P $
# where
# $ S_t $
# is the current stock price and
# $ P $
# is the entry price for the short position.
# </li>
# <li>
# Return for each pair
# </li>
# <li>
# Overalll return for the trading period
# </li>
# <li>
# Standard deviation for the trading period
# </li>
# <li>
# Number of pairs that had a loss and a profit
# </li>
# <li>
# Maximum drawdown for the trading period
# </li>
# </ol>
# <h4>
# Yearly Results
# </h4>
# <li>
# Yearly return
# </li>
# <li>
# Yearly standard deviation
# </li>
# <li>
# Yearly maximum drawdown
# </li>
# <li>
# Sharpe Ratio
# </li>
# <li>
# VaR and CVaR
# </li>
# </ol>
# <h4>
#
# +
#
# To generate a python file from the notebook use jupytext:
# pip install jupytext --upgrade
# jupytext --to py pairs_trading_backtest.ipynb
#

# Apply the default theme
sns.set_theme()

# Set the random seed for generating a random set of pairs.
random_seed = 7723
random.seed(random_seed)

s_and_p_file = s_and_p_directory + os.path.sep + s_and_p_stock_file

start_date_str = '2007-01-03'
# start_date_str = '2020-01-03'
start_date: datetime = datetime.fromisoformat(start_date_str)

trading_days = 252
half_year = int(trading_days / 2)
quarter = int(trading_days / 4)

stock_info_df = read_s_and_p_stock_info(s_and_p_file)
stock_l: list = list(set(stock_info_df['Symbol']))
stock_l.sort()
market_data = MarketData(start_date=start_date)

# Get close prices for the S&P 500 list
close_prices_df = market_data.get_close_data(stock_l)
final_stock_list = list(close_prices_df.columns)

mask = stock_info_df['Symbol'].isin(final_stock_list)
# Some stocks were listed on the stock exchange later than start_date. In this case the stock will not
# be returned by MarketData.get_close_data(). final_stock_info has the Symbol, Name, Sector for the
# set of stocks that it was possible to obtain close prices for the date range.
final_stock_info_df = stock_info_df[mask]

sectors = extract_sectors(final_stock_info_df)

pairs_list = get_pairs(sectors)


class CointData:

    def __init__(self, stock_a: str, stock_b: str, weight: float, intercept: float):
        self.stock_a: str = stock_a
        self.stock_b: str = stock_b
        self.key = f'{self.stock_a}:{self.stock_b}'
        self.weight: float = weight
        self.intercept: float = intercept
        self.stddev: float = 0.0

    def __str__(self):
        s = f'{self.key} weight: {self.weight} intercept: {self.intercept} stddev: {self.stddev}'
        return s


class PairStatisticsBase:
    def __init__(self) -> None:
        self.decimals = 2

    def pair_regression(self, pair: Tuple, close_prices: pd.DataFrame) -> Tuple[Tuple[str, str], float, float, RegressionResults]:
        stock_A: str = pair[0]
        stock_B: str = pair[1]
        close_a = close_prices[stock_A]
        close_b = close_prices[stock_B]
        close_b_const = sm.add_constant(close_b)
        # x = I + b * A
        result_ab = sm.OLS(close_a, close_b_const).fit()
        close_a_const = sm.add_constant(close_a)
        # x = I + b * B
        result_ba = sm.OLS(close_b, close_a_const).fit()
        slope_ab = result_ab.params[stock_B]
        slope_ba = result_ba.params[stock_A]
        result = result_ab
        slope = slope_ab
        if slope_ab < slope_ba:
            t = stock_A
            stock_A = stock_B
            stock_B = t
            result = result_ba
            slope = slope_ba
        slope: float = round(slope, self.decimals)
        intercept: float = round(result.params['const'], self.decimals)
        ordered_pair: Tuple[str,str] = (stock_A, stock_B)
        return ordered_pair, slope, intercept, result


class InSamplePairBase:
    def __init__(self, corr_cutoff: float, num_pairs: int) -> None:
        self.corr_cutoff = corr_cutoff
        self.num_pairs = num_pairs

    @classmethod
    @abstractmethod
    def get_in_sample_pairs(self, pairs_list: List[Tuple], close_prices: pd.DataFrame) -> List[CointData]:
        pass

class RandomInSamplePairs(InSamplePairBase):
    def __init__(self, corr_cutoff: float, num_pairs: int) -> None:
        super().__init__(corr_cutoff, num_pairs)

    def unique_random_index(self, set_len: int, set_range: int) -> List[int]:
        index_set: Set[int] = set()
        while len(index_set) < set_len:
            rand_ix = random.randint(0, set_range)
            if rand_ix not in index_set:
                index_set.add(rand_ix)
        return list(index_set)

    def get_in_sample_pairs(self, pairs_list: List[Tuple], close_prices: pd.DataFrame) -> List[CointData]:
        pair_stats_obj = PairStatisticsBase()
        random_index: List[int] = self.unique_random_index(self.num_pairs, len(pairs_list) - 1)
        # random_pair_list = pairs_list[ random_index ]
        random_pair_list: List[Tuple] = list(map(lambda ix: pairs_list[ix], random_index))
        coint_list: List[CointData] = list()
        for pair in random_pair_list:
            pair_syms, slope, intercept, result = pair_stats_obj.pair_regression(pair, close_prices=close_prices_df)
            coint_data = CointData(stock_a=pair_syms[0], stock_b=pair_syms[1], weight=slope, intercept=intercept)
            coint_list.append(coint_data)
        return coint_list


class PairStatistics(PairStatisticsBase):

    def __init__(self) -> None:
        super().__init__()
        pass

    def pair_correlation(self, pair: Tuple, close_prices: pd.DataFrame) -> float:
        stock_list = close_prices.columns
        """
        :param pair: the pair is a Tuple like  ('ALB', 'APD', 'materials') So pair[0] is 'ALB' and pair[1] is 'APD'
        :return: the correlation for the pair in the close price period
        """
        stock_A: str = pair[0]
        stock_B: str = pair[1]
        correlation: float = -2.0  # bad correlation value
        if stock_A in stock_list and stock_B in stock_list:
            data_a = close_prices[stock_A]
            data_b = close_prices[stock_B]
            c = np.corrcoef(data_a, data_b)
            correlation = round(c[0, 1], 2)
        return correlation

    def check_coint(self, coint_stat: float, critical_vals: dict) -> bool:
        """
        :param coint_stat: the ADF statistic
        :param critical_vals: a dictionary defining the ADF intervals {'1%': -3.49, '5%': -2.89, '10%': -2.58}. The
                              dictionary values may be either positive or negative.
        :return: if the adf_stat is in the critical value range, return True, False otherwise
        """
        cointegrated = False
        abs_coint_stat = abs(coint_stat)
        for key, value in critical_vals.items():
            abs_value = abs(value)
            if abs_coint_stat > abs_value:
                cointegrated = True
                break
        return cointegrated

    def engle_granger_coint(self, pair: Tuple, close_prices: pd.DataFrame) -> CointData:
        stock_list = close_prices.columns
        coint_data = None
        stock_A = pair[0]
        stock_B = pair[1]
        if stock_A in stock_list and stock_B in stock_list:
            pair_syms, slope, intercept, result = self.pair_regression(pair, close_prices=close_prices)
            # A hack that attempts to get rid of outlier pairs. The values for the slope and intercept cutoffs
            # were arrived at by looking at the distributions of the data. Still, it's a bit arbitrary.
            if slope <= 6 and abs(intercept) <= 100:
                residuals = result.resid
                adf_result = adfuller(residuals)
                adf_stat = round(adf_result[0], self.decimals)
                critical_vals = adf_result[4]
                cointegrated = self.check_coint(adf_stat, critical_vals)
                if cointegrated:
                    coint_data = CointData(stock_a=pair_syms[0], stock_b=pair_syms[1], weight=slope, intercept=intercept)
        return coint_data

    def add_spread_stats(self, coint_data_list: List[CointData], close_prices: pd.DataFrame) -> None:
        """
        Add the spread standard deviation to the CointData object
        :param coint_data_list:
        :return:
        """
        for coint_pair in coint_data_list:
            stock_a = coint_pair.stock_a
            stock_b = coint_pair.stock_b
            close_a = close_prices[stock_a]
            close_b = close_prices[stock_b]
            weight = coint_pair.weight
            spread = close_a - coint_pair.intercept - weight * close_b
            coint_pair.stddev = np.std(spread)


class InSamplePairs(InSamplePairBase):

    def __init__(self, corr_cutoff: float, num_pairs: int) -> None:
        super().__init__(corr_cutoff=corr_cutoff, num_pairs=num_pairs)
        self.pair_stats_obj = PairStatistics()

    def select_pairs(self, pairs_list: List[Tuple], in_sample_close: pd.DataFrame) -> List[CointData]:
        """
        Select pairs with high correlation and cointegratoin
        :param pairs_list: a list of pairs
        :return: a list of CointData for pairs that have a correlation greater than self.corr_cutoff and
        are cointegrated.
        """
        coint_list: List = list()
        for pair in pairs_list:
            pair_cor = self.pair_stats_obj.pair_correlation(pair, close_prices=in_sample_close)
            if pair_cor >= self.corr_cutoff:
                coint_data = self.pair_stats_obj.engle_granger_coint(pair, close_prices=in_sample_close)
                if coint_data is not None:
                    coint_list.append(coint_data)
        return coint_list


    def get_in_sample_pairs(self, pairs_list: List[Tuple], close_prices: pd.DataFrame) -> List[CointData]:
        coint_data_list: List[CointData] = self.select_pairs(pairs_list, in_sample_close=close_prices)
        self.pair_stats_obj.add_spread_stats(coint_data_list, close_prices=close_prices)
        # Sort by declining standard deviation value
        coint_data_list.sort(key=lambda elem: elem.stddev, reverse=True)
        truncated_list = coint_data_list[0:self.num_pairs]
        return truncated_list


def normalize_df(data_df: pd.DataFrame) -> pd.DataFrame:
    min_s = data_df.min()
    max_s = data_df.max()
    norm_df = (data_df - min_s) / (max_s - min_s)
    return norm_df


def plot_stationary_ts(stationary_df: pd.DataFrame, plus_delta: float, minus_delta: float, title: str) -> None:
    stationary_df.plot(grid=True, title=title, figsize=(10, 6))
    stat_mean = stationary_df.mean()[0]
    plt.axhline(y=stat_mean, color='black', linewidth=2)
    plt.axhline(y=stat_mean + plus_delta, color='red', linewidth=1, linestyle='--')
    plt.axhline(y=stat_mean - minus_delta, color='green', linewidth=1, linestyle='--')
    plt.show()


def plot_pair_data(close_df: pd.DataFrame, pair: CointData, title_prefix: str) -> None:
    stock_a_df: pd.DataFrame = pd.DataFrame(close_df[pair.stock_a])
    stock_b_df: pd.DataFrame = pd.DataFrame(close_df[pair.stock_b])
    stock_a_norm_df = normalize_df(stock_a_df)
    stock_b_norm_df = normalize_df(stock_b_df)
    plot_two_ts(data_a=stock_a_norm_df, data_b=stock_b_norm_df,
                title=f'{title_prefix} normalized {pair.stock_a},{pair.stock_b}',
                x_label='date', y_label='Normalized Price')
    spread_df = pd.DataFrame(stock_a_df.values - pair.intercept - pair.weight * stock_b_df.values)
    spread_df.index = close_df.index
    plot_stationary_ts(stationary_df=spread_df, plus_delta=pair.stddev, minus_delta=pair.stddev,
                       title=f'{title_prefix} spread for {pair.stock_a} and {pair.stock_b}')


corr_cutoff = 0.75
num_pairs = 100

in_sample_start = find_date_index.findDateIndex(close_prices_df.index, start_date)
in_sample_end = in_sample_start + half_year
in_sample_df = close_prices_df.iloc[in_sample_start:in_sample_end]
pairs_stats_obj = PairStatistics()
period_backtest = InSamplePairs(corr_cutoff=corr_cutoff, num_pairs=num_pairs)
coint_list = period_backtest.get_in_sample_pairs(pairs_list, close_prices=in_sample_df)

spead_stddev = np.array(list(elem.stddev for elem in coint_list))
plt.hist(spead_stddev, bins='auto')
plt.title('Standard Deviation of the Pairs Spread')
plt.show()

out_of_sample_start = in_sample_end
out_of_sample_end = out_of_sample_start + quarter
out_of_sample_df = close_prices_df.iloc[out_of_sample_start:out_of_sample_end]

pair = coint_list[0]
plot_pair_data(in_sample_df, pair, 'In-sample')

plot_pair_data(out_of_sample_df, pair, 'Out-of-sample')


def plot_mean_spread(pair: CointData, close_prices_df: pd.DataFrame, start_ix: int, data_window: int,
                     mean_window: int, stddev_delta: float, title: str ) -> None:
    """
    Starting at start_ix calculate the spread from start_ix to start_ix + data_window.
    Calculate the rolling mean from start_ix

    :param close_prices_df:
    :param start_ix:
    :param data_window:
    :param mean_window:
    :return:
    """
    end_ix = start_ix + mean_window + data_window
    index = close_prices_df.index[start_ix:end_ix]
    index_from_window = index[window:]
    stock_a_df = close_prices_df[pair.stock_a].iloc[start_ix:end_ix]
    stock_b_df = close_prices_df[pair.stock_b].iloc[start_ix:end_ix]
    spread_df = pd.DataFrame(stock_a_df.values - pair.intercept - pair.weight * stock_b_df.values)
    spread_df.columns = ['Spread']
    spread_mean_df = spread_df.rolling(mean_window).mean().iloc[mean_window:]
    spread_mean_df.columns = ['Mean']
    spread_mean_df.index = index_from_window
    spread_stddev_df = stddev_delta * spread_df.rolling(mean_window).std().iloc[mean_window:]
    mean_plus_stddev = pd.DataFrame(spread_mean_df.values + spread_stddev_df.values)
    mean_plus_stddev.index = index_from_window
    mean_plus_stddev.columns = [f'Mean + {stddev_delta} x stddev']
    mean_minus_stddev = pd.DataFrame(spread_mean_df.values - spread_stddev_df.values)
    mean_minus_stddev.index = index_from_window
    mean_minus_stddev.columns = [f'Mean - {stddev_delta} x stddev']
    spread_df: pd.DataFrame = spread_df.iloc[window:]
    spread_df.index = index_from_window
    spread_df.columns = ['Spread']
    data_df = pd.concat([spread_df, spread_mean_df, mean_plus_stddev, mean_minus_stddev], axis=1)
    data_df.plot(grid=True, title=title, figsize=(10, 6))


delta = 2.0

window = trading_days // 12
window_in_of_sample_start = in_sample_start
plot_mean_spread(pair=pair, close_prices_df=close_prices_df, start_ix=window_in_of_sample_start,
                 data_window=half_year - window, mean_window=window, stddev_delta=delta,
                 title=f'{pair.key} in-sample with mean and stddev')

window_out_of_sample_start = in_sample_end - window
plot_mean_spread(pair=pair, close_prices_df=close_prices_df, start_ix=window_out_of_sample_start,
                 data_window=quarter, mean_window=window, stddev_delta=delta,
                 title=f'{pair.key} out-of-sample with mean and stddev')

plt.show()


class OpenPosition(Enum):
    NOT_OPEN = 1
    SHORT_A_LONG_B = 2
    LONG_A_SHORT_B = 3
    SHARE_PRICE_OUT_OF_BUDGET = 4
    OUT_OF_MARGIN = 5


class DayTransactions:

    def __init__(self,
                 day_date: datetime,
                 positive_trades: int,
                 negative_trades: int,
                 days_open: List[int],
                 day_profit: float,
                 day_return: float,
                 num_open_positions: int,
                 margin: int):
        """
        :param day_date: the date for the day
        :param positive_trades: number of pair transactions that were profitable
        :param negative_trades: number of pair transactions that were at a loss
        :param days_open: a list of the number of days that the pair positions were open. len(days_open) = num pairs
        :param day_profit: the total profit (or loss)
        :param day_return: the total return
        :param margin: the open margin for the day, calculated from the open positions.
        """
        self.day_date = day_date
        self.positive_trades = positive_trades
        self.negative_trades = negative_trades
        self.days_open = days_open
        self.day_profit = day_profit
        self.day_return = day_return
        self.num_open_positions = num_open_positions
        self.margin = margin

    def __str__(self):
        format = '%Y-%m-%d'
        s1 = f'date: {self.day_date.strftime(format)} win trades: {self.positive_trades} loss trades: {self.negative_trades} profit: {round(self.day_profit, 2)}'
        s2 = f'return: {round(self.day_return, 4)} margin: {self.margin}'
        return s1 + s2


class PairTransaction:
    """
    A container for the information on a pair transaction.
    """

    def __init__(self,
                 pair: str,
                 close_date: datetime,
                 days_open: int,
                 total_profit: float,
                 pair_return: float,
                 margin: int) -> None:
        """
        :param pair the pair ticker symbols concatenated as a string. Example 'HUM:VRTX'
        :param close_date: the date the transaction was closed
        :param days_open the number of days that the pair position was open
        :param total_profit the profit (or loss) from closing the pair position
        :param pair_return: the return of the long-short position
        :param margin: the margin cash required while the short position was open
        """
        self.pair = pair
        self.close_date = close_date
        self.days_open = days_open
        self.total_profit = total_profit
        self.pair_return = pair_return
        self.margin = margin

    def __str__(self):
        format = '%Y-%m-%d'
        s1 = f'{self.pair} days open: {self.days_open}, close date: {self.close_date.strftime(format)}, '
        s2 = f'profit: {self.total_profit}, return: {self.pair_return}, margin: {self.margin}'
        s = s1 + s2
        return s


class Position:
    initial_margin_percent = 0.50

    def __init__(self,
                 pair_str: str,
                 position_type: OpenPosition,
                 open_date: datetime,
                 price_a: float,
                 price_b: float,
                 day_index: int,
                 budget: int):
        """
        spread = stock_A - Weight * Stock_B
        :param pair_str a string representation of the pair 'FOO:BAR'
        :param position_type: an enumeration value indicating the position type
        :param open_date the date the position was opened
        :param price_a: the current price of stock A
        :param price_b: the current price of stock B
        :param day_index: the index in the close price series for the current day. Used to calculate days open
        :param budget: the cash that can be allocated for the long/short position

        """
        self.pair_str = pair_str
        self.position_type = position_type
        self.open_date = open_date
        self.price_a = price_a  # share price for stock A at time of position open
        self.price_b = price_b  # share price for stock B at time of position open
        self.day_index = day_index
        self.shares_a = 0
        self.shares_b = 0
        self.margin: int = 0  # The margin amount above the value of the long position
        if position_type == OpenPosition.SHORT_A_LONG_B:
            # Short A
            self.shares_a = budget // self.price_a
            cost_a = round(self.shares_a * self.price_a, 0)
            # Cash is the amount from the short of A plus any cash left over
            cash = cost_a + (budget - cost_a)
            # An approximately equal amount of cash is used to for the long postion.
            self.shares_b = cash // self.price_b
            cost_b = round(self.shares_b * self.price_b, 0)
            # The required margin is 150% of the short position. The long position can be used for the margin
            required_margin = round(cost_a * (1 + self.initial_margin_percent), 0)
            self.margin = max(required_margin - cost_b, 0)
        elif position_type == OpenPosition.LONG_A_SHORT_B:
            # Short weight * B
            self.shares_b = budget // self.price_b
            cost_b = round(self.shares_b * self.price_b, 0)
            cash = cost_b + (budget - cost_b)
            # Long A
            self.shares_a = cash // self.price_a
            cost_a = round(self.shares_a * self.price_a, 0)
            required_margin = round(cost_b * (1 + self.initial_margin_percent), 0)
            self.margin = max(required_margin - cost_a, 0)

    def __str__(self) -> str:
        s = f'{self.pair_str} open date: {self.open_date} price: {self.price_a}:{self.price_b} shares: {self.shares_a}:{self.shares_b} margin: {self.margin}'
        return s


class OutOfSampleBacktest:

    def __init__(self, holdings: int) -> None:
        self.holdings = holdings

    def backtest_pairs_list(self,
                            close_prices_df: pd.DataFrame,
                            pair_list: List[CointData],
                            start_date: datetime,
                            back_window: int,
                            trading_period: int,
                            delta: float):
        """
        Backtest a list of pairs over a single out-of-sample trading period.

        :param close_prices_df: the close prices for the stock data set
        :param pair_list: the list of pairs to be back tested
        :param start_date: the start date where the test begins. This date must be the date that followed the
                           in-sample period used to select the pairs
        :param back_window: a window backward used for the running mean and standard deviation
        :param trading_period: the width of the trading period
        :param delta: the multiplier for the standard deviation used in the Bollinger band
        :return:
        """
        date_index = close_prices_df.index
        start_ix = find_date_index.findDateIndex(date_index, start_date)
        assert start_ix > 0
        assert (start_ix - window) >= 0


class DailyStats:
    def __init__(self, key: str, mean: float, stddev: float, day_spread: float, stock_a: float, stock_b: float):
        self.key = key
        self.mean = mean
        self.stddev = stddev
        self.day_spread = day_spread
        self.stock_a = stock_a
        self.stock_b = stock_b

    def __str__(self) -> str:
        s = f'{self.key} price: {self.stock_a}:{self.stock_b} spread: {self.day_spread} mean: {self.mean} stddev: {self.stddev}'
        return s


class HistoricalBacktest:
    reg_T_margin_percent = 0.25

    #             in_sample_pairs_obj = InSamplePairs(corr_cutoff=self.corr_cutoff, num_pairs=self.num_pairs)

    def __init__(self,
                 pairs_list: List[Tuple],
                 initial_holdings: int,
                 num_pairs: int,
                 in_sample_days: int,
                 out_of_sample_days: int,
                 back_window: int,
                 delta: float,
                 in_sample_pairs_obj: InSamplePairBase ) -> None:
        """
        Back test pairs trading through a historical period
        :param pairs_list: the list of possible pairs from the S&P 500. Each Tuple consists of the pair stock
                           symbols and the industry sector. For example: ('AAPL', 'ACN', 'information-technology')
        :param initial_holdings: The trading capital. Because this is a long-short strategy, this is the amount of cash
                         that can be used for the required margin.
        :param num_pairs: the total number of pairs to be traded
        :param in_sample_days: the number of days in the in-sample period used to select pairs
        :param out_of_sample_days: the number of days in th out-of-sample trading period
        :param back_window: the look-back window used to calculate the running mean and standard deviation
        :param delta: The offset from the mean for opening positions
        """
        assert back_window < in_sample_days
        self.pairs_list = pairs_list
        self.initial_holdings = initial_holdings
        self.num_pairs = num_pairs
        self.in_sample_days = in_sample_days
        self.out_of_sample_days = out_of_sample_days
        self.back_window = back_window
        self.delta = delta
        self.in_sample_pairs_obj = in_sample_pairs_obj

    def spread_stats(self, pair: CointData,
                     back_win_stock_a: pd.DataFrame,
                     back_win_stock_b: pd.DataFrame,
                     stock_a_day: float,
                     stock_b_day: float) -> DailyStats:
        intercept: float = pair.intercept
        weight: float = pair.weight

        back_spread_df = pd.DataFrame(
            back_win_stock_a.values - pair.intercept - pair.weight * back_win_stock_b.values)
        mean = np.mean(back_spread_df.values).round(2)
        stddev = np.std(back_spread_df.values).round(2)
        day_spread = stock_a_day - pair.intercept - pair.weight * stock_b_day
        stats = DailyStats(key=pair.key, mean=mean, stddev=stddev, day_spread=day_spread, stock_a=stock_a_day,
                           stock_b=stock_b_day)
        return stats


    def update_margin(self, position: Position, day_stats: DailyStats, current_date: datetime) -> None:
        """
        Interactive Brokers adjusts the margin as the price changes. This means that as the price
        changes, additional margin may be required. This function calculates the maximum margin amount
        for the position.
        :param day_stats: the current statistics for the day
        :param current_date: the current date
        :return: None
        """
        assert position is not None
        # Short A, Long B
        short_shares = position.shares_a
        long_shares = position.shares_b
        short_price = day_stats.stock_a
        long_price = day_stats.stock_b
        if position.position_type == OpenPosition.LONG_A_SHORT_B:
            # Short B, Long A
            short_shares = position.shares_b
            long_shares = position.shares_a
            short_price = day_stats.stock_b
            long_price = day_stats.stock_a
        short_position = short_shares * short_price
        long_position = long_shares * long_price
        required_margin = round(short_position * (1 + self.reg_T_margin_percent), 2)
        required_cash = max(required_margin - long_position, 0)
        position.margin = max(position.margin, required_cash)

    def close_position(self, position: Position, close_date: datetime, day_index: int, price_a: float,
                       price_b: float) -> PairTransaction:
        """
        A short position has a positive return when the position close price is less than the open price.
        A long position has a positive return when the position close is greater than the open price.
        :param position: A Position object
        :param close_date: The date the position is closed
        :param price_a: The current price for stock A
        :param price_b: The current price for stock B
        :return: a PairTransaction object
        Eric Zivot's slides on return calculation: https://faculty.washington.edu/ezivot/econ424/returncalculationslides.pdf
        """
        transaction = None
        if position.position_type == OpenPosition.LONG_A_SHORT_B or \
                position.position_type == OpenPosition.SHORT_A_LONG_B:
            # Short A, Long B
            long_shares = position.shares_b
            short_shares = position.shares_a
            long_share_price = position.price_b
            short_share_price = position.price_a
            close_long = long_shares * price_b
            close_short = short_shares * price_a
            if position.position_type == OpenPosition.LONG_A_SHORT_B:
                long_shares = position.shares_a
                short_shares = position.shares_b
                long_share_price = position.price_a
                short_share_price = position.price_b
                close_long = long_shares * price_a
                close_short = short_shares * price_b
            short_position = short_shares * short_share_price
            long_position = long_shares * long_share_price
            # Example: open short at 20/share and 80 shares for a total of 1600
            #          close short at 15 and 80 shares for a total of 1200
            #          short profit = 1600 - 1200 = 400
            short_profit = short_position - close_short
            # Example: open long at 20/share and 80 shares for 1600.
            #          close long at 22/share and 80 shares for 1760
            #          long profit = 1760 - 1600 = 160
            long_profit = close_long - long_position
            total_profit = round(short_profit + long_profit, 2)
            # Short return:
            #   short opens at 20
            #   short closes at 15
            #   R = (20 / 15) - 1 = 0.33
            # Long return
            #   long opens at 15
            #   long closes at 20
            #   R = (20/15) - 1 = 0.33
            ret_short = (short_position / close_short) - 1
            ret_long = (close_long / long_position) - 1
            # total long/short portfolio return
            #   long position ~ 50%
            #   short positin ~50%
            #   ret_short = 0.02
            #   ret_long = 0.03
            #   R_portfolio = w1 * R_long + w2 * R_short where w1 = w2 = 0.5
            #   total = 0.5 x 0.02 + 0.5 x 0.03 = 0.025
            weight_short = 0.5
            weight_long = 0.5
            total_return = round((weight_short * ret_short) + (weight_long * ret_long), 4)
            days_open = (day_index - position.day_index) + 1
            transaction = PairTransaction(pair=position.pair_str,
                                          close_date=close_date,
                                          days_open=days_open,
                                          total_profit=total_profit,
                                          pair_return=total_return,
                                          margin=int(position.margin))
        return transaction

    def calc_open_position_margin(self, open_positions: Dict[str, Position]) -> int:
        """
        Calculate the margin required for the open positions.

        :param open_positions: A dictionary containing the open position data
        :return: the margin required for the day
        """
        open_margin: int = 0
        for key, position in open_positions.items():
            open_margin += position.margin
        return round(open_margin, 0)


    def process_day_transactions(self,
                                 day_date: datetime,
                                 transaction_l: List[PairTransaction],
                                 open_positions: Dict[str, Position],
                                 holdings: int) -> Tuple[int, DayTransactions]:
        cur_holdings: int = holdings
        day_transactions = None
        num_trades = len(transaction_l)
        if num_trades > 0:
            port_weight = 1.0 / num_trades
            open_days_l: List[int] = list()
            day_return = 0.0
            day_profit = 0.0
            win_trades = 0
            loss_trades = 0
            for trans in transaction_l:
                open_days_l.append(trans.days_open)
                day_return = day_return + port_weight * trans.pair_return
                if trans.total_profit > 0:
                    win_trades += 1
                else:
                    loss_trades += 1
                day_profit = day_profit + trans.total_profit
            margin = self.calc_open_position_margin(open_positions=open_positions)
            cur_holdings = cur_holdings + day_profit
            day_transactions = DayTransactions(day_date=day_date,
                                               positive_trades=win_trades,
                                               negative_trades=loss_trades,
                                               days_open=open_days_l,
                                               day_profit=day_profit,
                                               day_return=day_return,
                                               num_open_positions=len(open_positions),
                                               margin=margin)
        return cur_holdings, day_transactions

    def close_open_positions(self,
                             day_close_df: pd.DataFrame,
                             holdings: int,
                             open_positions: Dict[str, Position],
                             current_date: datetime,
                             day_index: int) -> Tuple[int, DayTransactions]:
        close_transactions: List[PairTransaction] = list()
        for key, position in open_positions.items():
            sym_l = key.split(':')
            stock_a_sym = sym_l[0]
            stock_b_sym = sym_l[1]
            close_a = day_close_df[stock_a_sym]
            close_b = day_close_df[stock_b_sym]
            transaction = self.close_position(position=position,
                                              close_date=current_date,
                                              day_index=day_index,
                                              price_a=close_a,
                                              price_b=close_b)
            close_transactions.append(transaction)
        cur_holdings, day_transactions = self.process_day_transactions(day_date=current_date,
                                                                       transaction_l=close_transactions,
                                                                       open_positions=open_positions,
                                                                       holdings=holdings)

        return cur_holdings, day_transactions

    def manage_position(self,
                        day_stats: DailyStats,
                        current_date: datetime,
                        day_index: int,
                        daily_transactions: List[PairTransaction],
                        open_positions: Dict[str, Position]) -> None:
        transaction = None
        assert day_stats.key in open_positions
        position = open_positions[day_stats.key]
        self.update_margin(position, day_stats, current_date)
        # spread = A - intercept - (W * B)
        # SHORT_A_LONG_B: spread >= mean + delta * stddev : close when spread <= mean
        # LONG_A_SHORT_B: spread <= mean + delta * stddev : close when spread >= mean
        if position.position_type == OpenPosition.SHORT_A_LONG_B:
            if day_stats.day_spread <= day_stats.mean:
                transaction = self.close_position(position, current_date, day_index, day_stats.stock_a,
                                                  day_stats.stock_b)
        elif position.position_type == OpenPosition.LONG_A_SHORT_B:
            if day_stats.day_spread >= day_stats.mean:
                transaction = self.close_position(position, current_date, day_index, day_stats.stock_a,
                                                  day_stats.stock_b)
        if transaction is not None:
            daily_transactions.append(transaction)
            del open_positions[day_stats.key]


    def open_position(self,
                      day_stats: DailyStats,
                      current_date: datetime,
                      day_index: int,
                      stock_budget: int,
                      open_positions: Dict[str, Position]) -> None:
        position = None
        position_type: OpenPosition = OpenPosition.NOT_OPEN
        if day_stats.day_spread >= day_stats.mean + (self.delta * day_stats.stddev):
            position_type = OpenPosition.SHORT_A_LONG_B
        elif day_stats.day_spread <= day_stats.mean - (self.delta * day_stats.stddev):
            position_type = OpenPosition.LONG_A_SHORT_B
        if position_type == OpenPosition.SHORT_A_LONG_B or position_type == OpenPosition.LONG_A_SHORT_B:
            position = Position(pair_str=day_stats.key,
                                open_date=current_date,
                                price_a=day_stats.stock_a,
                                price_b=day_stats.stock_b,
                                day_index=day_index,
                                budget=stock_budget,
                                position_type=position_type)
            if position.shares_a == 0 or position.shares_b == 0:
                position_type = OpenPosition.SHARE_PRICE_OUT_OF_BUDGET
        if position_type == OpenPosition.SHORT_A_LONG_B or position_type == OpenPosition.LONG_A_SHORT_B:
            open_positions[day_stats.key] = position

    def calc_pair_budget(self, holdings: int) -> int:
        # Holdings of 100,000. A short requires 150 percent in margin. This is the proceeds of the short plus 50 percent.
        # So if we short 200,000 of stock we get 200,000 from the short proceeds which is used to open a long
        # position of 200,000.  In addition, we need 100,000 for the 50 percent margin.
        #
        # All pairs are not traded at the same time, so the actual margin will be much less.
        #
        trade_capital = (2 * holdings)
        # required margin would be trade_capital * 0.5 or 80,000
        stock_budget = int(trade_capital // self.num_pairs)
        return stock_budget

    def out_of_sample_test(self, start_ix: int,
                           out_of_sample_df: pd.DataFrame,
                           pairs_list: List[CointData],
                           holdings: int) -> Tuple[int, pd.DataFrame]:
        open_positions: Dict[str, Position] = dict()
        out_of_sample_index = out_of_sample_df.index
        end_ix = out_of_sample_df.shape[0]
        row_date = None
        row_ix = 0
        out_of_sample_day = pd.DataFrame()
        day_transactions_l: List[DayTransactions] = list()
        for row_ix in range(start_ix, end_ix):
            pair_budget = self.calc_pair_budget(holdings)
            daily_transactions: List[PairTransaction] = list()
            out_of_sample_back = out_of_sample_df.iloc[row_ix - self.back_window:row_ix]
            out_of_sample_day = out_of_sample_df.iloc[row_ix]
            row_date = pd.to_datetime(out_of_sample_index[row_ix])
            for pair in pairs_list:
                back_win_stock_a = pd.DataFrame(out_of_sample_back[pair.stock_a])
                back_win_stock_b = pd.DataFrame(out_of_sample_back[pair.stock_b])
                stock_a_day = out_of_sample_day[pair.stock_a]
                stock_b_day = out_of_sample_day[pair.stock_b]
                day_stats: DailyStats = self.spread_stats(pair=pair,
                                                          back_win_stock_a=back_win_stock_a,
                                                          back_win_stock_b=back_win_stock_b,
                                                          stock_a_day=stock_a_day,
                                                          stock_b_day=stock_b_day)
                if pair.key in open_positions:
                    self.manage_position(day_stats=day_stats,
                                         current_date=row_date,
                                         day_index=row_ix,
                                         daily_transactions=daily_transactions,
                                         open_positions=open_positions)
                else:  # Possibly open a position depending on the spread
                    self.open_position(day_stats,
                                       current_date=row_date,
                                       day_index=row_ix,
                                       stock_budget=pair_budget,
                                       open_positions=open_positions)
            holdings, day_transactions = self.process_day_transactions(day_date=row_date,
                                                                       transaction_l=daily_transactions,
                                                                       open_positions=open_positions,
                                                                       holdings=holdings)
            if day_transactions is not None:
                day_transactions_l.append(day_transactions)
        if len(open_positions) > 0:
            # At the end of the trading period (e.g., the quarter) all open positions must be closed at the last price
            holdings, day_transactions = self.close_open_positions(day_close_df=out_of_sample_day,
                                                                   holdings=holdings,
                                                                   open_positions=open_positions,
                                                                   current_date=row_date,
                                                                   day_index=row_ix)
            if day_transactions is not None:
                day_transactions_l.append(day_transactions)
        day_trans_df = pd.DataFrame()
        if len(day_transactions_l) > 0:
            day_trans_df = pd.DataFrame(trans.__dict__ for trans in day_transactions_l)
        return holdings, day_trans_df

    def historical_backtest(self,
                            close_prices_df: pd.DataFrame,
                            start_date: datetime,
                            delta: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        date_index = close_prices_df.index
        start_ix = find_date_index.findDateIndex(date_index, start_date)
        assert start_ix >= 0
        end_ix = close_prices_df.shape[0]  # number of rows in close_prices_df
        print(f'index range: {start_ix} - end_ix: {end_ix}')
        all_transactions_df = pd.DataFrame()
        holdings_l: List[float] = list()
        holdings_date_l = list()
        holdings = self.initial_holdings
        for ix in range(start_ix, end_ix - (self.in_sample_days + self.out_of_sample_days), self.out_of_sample_days):
            in_sample_end_ix = ix + self.in_sample_days
            in_sample_date_start = date_index[ix]
            in_sample_date_end = date_index[in_sample_end_ix]
            out_of_sample_start = in_sample_end_ix - self.back_window
            out_of_sample_end = in_sample_end_ix + self.out_of_sample_days
            out_of_sample_date_start = date_index[out_of_sample_start]
            out_of_sample_date_end = date_index[out_of_sample_end]
            print(f'in-sample: {ix}:{in_sample_end_ix} {in_sample_date_start}:{in_sample_date_end}')
            print(
                f'out-of-sample: {out_of_sample_start}:{out_of_sample_end} {out_of_sample_date_start}:{out_of_sample_date_end}')
            in_sample_close_df = pd.DataFrame(close_prices_df.iloc[ix:in_sample_end_ix])
            selected_pairs: List[CointData] = self.in_sample_pairs_obj.get_in_sample_pairs(pairs_list=self.pairs_list, close_prices=in_sample_close_df)
            out_of_sample_df = pd.DataFrame(close_prices_df.iloc[out_of_sample_start:out_of_sample_end])
            holdings, day_transactions_df = self.out_of_sample_test(start_ix=self.back_window,
                                                                    out_of_sample_df=out_of_sample_df,
                                                                    pairs_list=selected_pairs,
                                                                    holdings=holdings)
            if day_transactions_df.shape[1] > 0:
                all_transactions_df = pd.concat([all_transactions_df, day_transactions_df], axis=0)
            print(f'holdings: {holdings}')
            holdings_l.append(holdings)
            holdings_date_l.append(out_of_sample_date_end)
        holdings_df = pd.DataFrame(holdings_l)
        holdings_df.index = holdings_date_l
        holdings_df.columns = ['portfolio balance']
        return all_transactions_df, holdings_df


class ReturnCalculation:
    def simple_return(self, time_series: np.array, period: int = 1) -> List:
        return list(((time_series[i] / time_series[i - period]) - 1.0 for i in range(period, len(time_series), period)))

    def return_df(self, time_series_df: pd.DataFrame) -> pd.DataFrame:
        r_df: pd.DataFrame = pd.DataFrame()
        time_series_a: np.array = time_series_df.values
        return_l = self.simple_return(time_series_a, 1)
        r_df = pd.DataFrame(return_l)
        date_index = time_series_df.index
        r_df.index = date_index[1:len(date_index)]
        r_df.columns = time_series_df.columns
        return r_df

    def apply_return(self, start_val: float, return_df: pd.DataFrame) -> np.array:
        port_a: np.array = np.zeros(return_df.shape[0] + 1)
        port_a[0] = start_val
        return_a = return_df.values
        for i in range(1, len(port_a)):
            port_a[i] = port_a[i - 1] + port_a[i - 1] * return_a[i - 1]
        return port_a

faulthandler.enable()

pairs_result_dir = 'back_test_data'
pairs_result_file = 'pairs_backtest.csv'
rand_pairs_result_file = 'rand_pairs_backtest.csv'
pairs_holdings_file = 'holdings.csv'
rand_pairs_holdings_file = 'rand_holdings.csv'
pairs_result_path = pairs_result_dir + os.path.sep + pairs_result_file
holdings_path = pairs_result_dir + os.path.sep + pairs_holdings_file
rand_pairs_result_path = pairs_result_dir + os.path.sep + rand_pairs_result_file
rand_holdings_path = pairs_result_dir + os.path.sep + rand_pairs_holdings_file


if not os.path.exists(pairs_result_dir):
    os.mkdir(pairs_result_dir)
if not os.path.exists(pairs_result_path):
    initial_holdings = 100000
    in_sample_pair_obj = InSamplePairs(corr_cutoff=corr_cutoff, num_pairs=num_pairs)
    historical_backtest = HistoricalBacktest(pairs_list=pairs_list,
                                             initial_holdings=initial_holdings,
                                             num_pairs=num_pairs,
                                             in_sample_days=half_year,
                                             out_of_sample_days=quarter,
                                             back_window=quarter // 3,
                                             delta=delta,
                                             in_sample_pairs_obj=in_sample_pair_obj)
    all_transactions_df, holdings_df = historical_backtest.historical_backtest(close_prices_df=close_prices_df,
                                                                               start_date=start_date,
                                                                               delta=delta)
    all_transactions_df.to_csv(pairs_result_path)
    holdings_df.to_csv(holdings_path)

    in_sample_random_pair_obj = RandomInSamplePairs(corr_cutoff=corr_cutoff, num_pairs=num_pairs)
    random_historical_backtest = HistoricalBacktest(pairs_list=pairs_list,
                                             initial_holdings=initial_holdings,
                                             num_pairs=num_pairs,
                                             in_sample_days=half_year,
                                             out_of_sample_days=quarter,
                                             back_window=quarter // 3,
                                             delta=delta,
                                             in_sample_pairs_obj=in_sample_random_pair_obj)
    all_rand_transactions_df, rand_holdings_df = random_historical_backtest.historical_backtest(close_prices_df=close_prices_df,
                                                                               start_date=start_date,
                                                                               delta=delta)
    all_rand_transactions_df.to_csv(rand_pairs_result_path)
    rand_holdings_df.to_csv(rand_holdings_path)
else:
    all_transactions_df = pd.read_csv(pairs_result_path, index_col=0)
    holdings_df = pd.read_csv(holdings_path, index_col=0)

# 'day_date', 'positive_trades', 'negative_trades', 'days_open',
#        'day_profit', 'day_return', 'num_open_positions', 'margin'

print(tabulate(holdings_df, headers=[*holdings_df.columns], tablefmt='fancy_grid'))



transaction_index = all_transactions_df['day_date']
m_df = pd.DataFrame(all_transactions_df['margin'].values)
m_df.index = pd.to_datetime(transaction_index)
m_df.columns = ['Required Margin']
h_df = pd.DataFrame(holdings_df.values)
h_df.index = pd.to_datetime( holdings_df.index)
h_df.columns = ['Holdings']

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(h_df, color='red', label='holdings', linewidth=4)
ax.set_ylabel('Holdings')
ax.plot(m_df, color='blue', label='Required Margin', linewidth=1)
ax.set_ylabel('Dollars')
plt.legend(loc='upper left')
plt.show()



# ax = m_df.plot(grid=True, title='Required Margin', figsize=(10, 6))


open_position_count_df = pd.DataFrame( all_transactions_df['num_open_positions'])
open_position_count_df.index = transaction_index
open_position_count_df.plot(grid=True, title='Open Positions', figsize=(10, 6))
plt.show()

days_open_l = list(all_transactions_df['days_open'])
# When the all_transactions_df DataFrame is built from the back test code it contains a list of lists
# for the days_open: [[6, 3, 3], [5], [8, 6, 7], [10, 5, 10, 5, 2, 9]] When the DataFrame is read from
# a file it contains a list of strings for the lists: ['[6, 3, 3]', '[5]', '[8, 6, 7]', '[10, 5, 10, 5, 2, 9]']
# If it is a list of string, the string need to be converted to lists. This is done via the ast.literal_eval
# function.
if type(days_open_l[0]) == str:
    days_open_l = list(map(ast.literal_eval, days_open_l))
days_open = list(deepflatten(days_open_l))

plt.hist(days_open, bins='auto')
plt.title('Number of days a pair trade is open')
plt.show()


returns_df = all_transactions_df['day_return']

pass

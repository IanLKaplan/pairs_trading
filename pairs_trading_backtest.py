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

import os
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from dateutil.utils import today
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate

from pairs.pairs import get_pairs
#
# Local libraries
#
from plot_ts.plot_time_series import plot_two_ts, plot_four_ts
from read_market_data.MarketData import MarketData, read_s_and_p_stock_info, extract_sectors

from s_and_p_filter import s_and_p_directory, s_and_p_stock_file
from utils import find_date_index

# Apply the default theme
sns.set_theme()

s_and_p_file = s_and_p_directory + os.path.sep + s_and_p_stock_file

start_date_str = '2007-01-03'
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

class PairStatistics:

    def __init__(self, period_close_prices_df: pd.DataFrame) -> None:
        """
        :param period_close_prices_df: close prices for an in-sample period
        """
        self.close_prices: pd.DataFrame = period_close_prices_df
        self.stock_list = self.close_prices.columns
        self.decimals = 2

    def pair_correlation(self, pair: Tuple) -> float:
        """
        :param pair: the pair is a Tuple like  ('ALB', 'APD', 'materials') So pair[0] is 'ALB' and pair[1] is 'APD'
        :return: the correlation for the pair in the close price period
        """
        stock_A: str = pair[0]
        stock_B: str = pair[1]
        correlation: float = -2.0  # bad correlation value
        if stock_A in self.stock_list and stock_B in self.stock_list:
            data_a = self.close_prices[stock_A]
            data_b = self.close_prices[stock_B]
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

    def engle_granger_coint(self, pair: Tuple) -> CointData:
        coint_data = None
        stock_A: str = pair[0]
        stock_B: str = pair[1]
        if stock_A in self.stock_list and stock_B in self.stock_list:
            close_a = self.close_prices[stock_A]
            close_b = self.close_prices[stock_B]
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
            slope = round(slope, self.decimals)
            intercept = round(result.params['const'], self.decimals)
            # A hack that attempts to get rid of outlier pairs. The values for the slop and intercept cutoffs
            # were arrived at by looking at the distributions of the data. Still, it's a bit arbitrary.
            if slope <= 6 and abs(intercept) <= 100:
                residuals = result.resid
                adf_result = adfuller(residuals)
                adf_stat = round(adf_result[0], self.decimals)
                critical_vals = adf_result[4]
                cointegrated = self.check_coint(adf_stat, critical_vals)
                if cointegrated:
                    coint_data = CointData(stock_a=stock_A, stock_b=stock_B, weight=slope, intercept=intercept)
        return coint_data


class InSamplePairs:

    def __init__(self, in_sample_close_df: pd.DataFrame, corr_cutoff: float) -> None:
        self.corr_cutoff = corr_cutoff
        self.in_sample_close_df = in_sample_close_df
        self.pairs_stats = PairStatistics(self.in_sample_close_df)

    def select_pairs(self, pairs_list: List[Tuple]) -> List[CointData]:
        """
        Select pairs with high correlation and cointegratoin
        :param pairs_list: a list of pairs
        :return: a list of CointData for pairs that have a correlation greater than self.corr_cutoff and
        are cointegrated.
        """
        coint_list: List = list()
        for pair in pairs_list:
            pair_cor = self.pairs_stats.pair_correlation(pair)
            if pair_cor >= self.corr_cutoff:
                coint_data = self.pairs_stats.engle_granger_coint(pair)
                if coint_data is not None:
                    coint_list.append(coint_data)
        return coint_list

    def add_spread_stats(self, coint_data_list: List[CointData]) -> None:
        """
        Add the spread standard deviation to the CointData object
        :param coint_data_list:
        :return:
        """
        for coint_pair in coint_data_list:
            stock_a = coint_pair.stock_a
            stock_b = coint_pair.stock_b
            close_a = self.in_sample_close_df[stock_a]
            close_b = self.in_sample_close_df[stock_b]
            weight = coint_pair.weight
            spread = close_a - coint_pair.intercept - weight * close_b
            coint_pair.stddev = np.std(spread)

    def filter_pairs_list(self, coint_data_list: List[CointData]) -> List[CointData]:
        """
        Filter the pairs list so that the stocks in the list are unique. That is, no stock appears
        in more than one pair.

        This is done by building a dictionary with the key for stock_a from CointData and
        a list of the CointData elements that have stock_a. The maximum standard deviation
        is then used to find the maximum element.
        :param coint_data_list:
        :return:
        """
        filtered_pairs: List[CointData] = list()
        pairs_dict: Dict = dict()
        for pair_info in coint_data_list:
            stock_key = pair_info.stock_a
            if stock_key not in pairs_dict:
                pairs_dict[stock_key] = list()
            l: List = pairs_dict[stock_key]
            l.append(pair_info)
        for key in pairs_dict.keys():
            l: List = pairs_dict[key]
            max_elem = max(l, key=lambda elem: elem.stddev)
            if max_elem is not None:
                filtered_pairs.append(max_elem)
        return filtered_pairs

    def get_in_sample_pairs(self, pairs_list: List[Tuple]) -> List[CointData]:
        coint_data_list: List[CointData] = self.select_pairs(pairs_list)
        self.add_spread_stats(coint_data_list)
        filtered_list = self.filter_pairs_list(coint_data_list)
        # Sort by declining standard deviation value
        filtered_list.sort(key=lambda elem: elem.stddev, reverse=True)
        return filtered_list


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
    plot_two_ts(data_a=stock_a_norm_df, data_b=stock_b_norm_df, title=f'{title_prefix} normalized {pair.stock_a},{pair.stock_b}',
                x_label='date', y_label='Normalized Price')
    spread_df = pd.DataFrame(stock_a_df.values - pair.intercept - pair.weight * stock_b_df.values)
    spread_df.index = close_df.index
    plot_stationary_ts(stationary_df=spread_df, plus_delta=pair.stddev, minus_delta=pair.stddev,
                       title=f'{title_prefix} spread for {pair.stock_a} and {pair.stock_b}')


corr_cutoff = 0.75

in_sample_start = find_date_index.findDateIndex(close_prices_df.index, start_date)
in_sample_end = in_sample_start + half_year
in_sample_df = close_prices_df.iloc[in_sample_start:in_sample_end]
period_backtest = InSamplePairs(in_sample_close_df=in_sample_df, corr_cutoff=corr_cutoff)
coint_list = period_backtest.get_in_sample_pairs(pairs_list)

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
                     mean_window: int, stddev_delta: float, title: str) -> None:
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
    data_df.plot(grid=True, title=title, figsize=(10,6))


stddev_delta = 1.0

window = trading_days // 12
window_in_of_sample_start = in_sample_start
plot_mean_spread(pair=pair, close_prices_df=close_prices_df, start_ix=window_in_of_sample_start,
                 data_window=half_year-window, mean_window=window, stddev_delta=stddev_delta, title=f'{pair.key} in-sample with mean and stddev')


window_out_of_sample_start = in_sample_end - window
plot_mean_spread(pair=pair, close_prices_df=close_prices_df, start_ix=window_out_of_sample_start,
                 data_window=quarter, mean_window=window, stddev_delta=stddev_delta, title=f'{pair.key} out-of-sample with mean and stddev')

plt.show()


class OpenPosition(Enum):
    NOT_OPEN = 1
    SHORT_A_LONG_B = 2
    LONG_A_SHORT_B = 3
    SHARE_PRICE_OUT_OF_BUDGET = 4


class PairTransaction:
    """
    A container for the information on a pair transaction.
    """
    def __init__(self,
                 pair: str,
                 open_date: datetime,
                 close_date: datetime,
                 total_profit: float,
                 pair_return: float,
                 margin: int) -> None:
        """
        :param open_date: the date the transaction was opened
        :param close_date: the date the transaction was closed
        :param pair_return: the return of the long-short position
        :param initial_margin: the margin cash required when the position was opened
        """
        self.pair = pair
        self.open_date = open_date
        self.close_date = close_date
        self.total_profit = total_profit
        self.pair_return = pair_return
        self.margin = margin
        # return on investment relative to the required margin cash
        self.ROI = self.total_profit / self.margin

    def __str__(self):
        format = '%Y-%m-%d'
        s1 = f'{self.pair} open date: {self.open_date.strftime(format)}, close date: {self.close_date.strftime(format)}, '
        s2 = f'profit: {self.total_profit}, return: {self.pair_return}, margin: {self.margin} ROI: {self.ROI}'
        s = s1 + s2
        return s


class Position:
    initial_margin_percent = 0.50
    def __init__(self, pair_str: str, open_date: datetime, price_a: float, price_b: float, budget: int, position_type: OpenPosition):
        """
        spread = stock_A - Weight * Stock_B
        :param price_a: the current price of stock A
        :param price_b: the current price of stock B
        :param weight: the weighting factor for stock_B
        :param budget: the cash that can be allocated for the long/short position
        :param position_type: an enumeration value indicating the position type
        """
        self.pair_str = pair_str
        self.position_type = position_type
        self.open_date = open_date
        self.price_a = price_a  # share price for stock A at time of position open
        self.price_b = price_b  # share price for stock B at time of position open
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
            required_margin = round(cost_a + cost_a * self.initial_margin_percent, 0)
            self.margin = max(required_margin - cost_b, 0)
        elif position_type == OpenPosition.LONG_A_SHORT_B:
            # Short weight * B
            self.shares_b = budget // self.price_b
            cost_b = round(self.shares_b * self.price_b, 0)
            cash = cost_b + (budget - cost_b)
            # Long A
            self.shares_a = cash // self.price_a
            cost_a = round(self.shares_a * self.price_a, 0)
            required_margin = round(cost_b + cost_b * self.initial_margin_percent, 0)
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


class HistoricalBacktest:
    reg_T_margin_percent = 0.25

    def __init__(self,
                 pairs_list: List[Tuple],
                 initial_holdings: int,
                 num_pairs: int,
                 corr_cutoff: float,
                 in_sample_days: int,
                 out_of_sample_days: int,
                 back_window: int,
                 delta: float) -> None:
        """
        Back test pairs trading through a historical period
        :param pairs_list: the list of possible pairs from the S&P 500. Each Tuple consists of the pair stock
                           symbols and the industry sector. For example: ('AAPL', 'ACN', 'information-technology')
        :param holdings: The trading capital. Because this is a long-short strategy, this is the amount of cash
                         that can be used for the required margin.
        :param num_pairs: the total number of pairs to be traded
        :param corr_cutoff: the correlation cutoff to be used as the first filter in pairs selection.
                            For example: 0.75
        :param in_sample_days: the number of days in the in-sample period used to select pairs
        :param out_of_sample_days: the number of days in th out-of-sample trading period
        :param back_window: the look-back window used to calculate the running mean and standard deviation
        :param delta: The offset from the mean for opening positions
        """
        assert back_window < in_sample_days
        self.pairs_list = pairs_list
        self.initial_holdings = holdings
        self.num_pairs = num_pairs
        self.corr_cutoff = corr_cutoff
        self.in_sample_days = in_sample_days
        self.out_of_sample_days = out_of_sample_days
        self.back_window = back_window
        self.delta = delta
        self.open_positions: Dict[str, Position] = dict()
        self.all_transactions = pd.DataFrame()

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


    def out_of_sample_stats(self, pair: CointData,
                            back_win_stock_a: pd.Series,
                            back_win_stock_b: pd.Series,
                            stock_a_day: float,
                            stock_b_day: float) -> DailyStats:
        intercept: float = pair.intercept
        weight: float = pair.weight
        back_spread = back_win_stock_a.values - intercept - (weight * back_win_stock_b.values)
        mean = np.mean(back_spread).round(2)
        stddev = np.std(back_spread).round(2)
        day_spread = stock_a_day - intercept - (weight * stock_b_day)
        stats = self.DailyStats(key=pair.key, mean=mean, stddev=stddev, day_spread=day_spread, stock_a=stock_a_day, stock_b=stock_b_day)
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
        required_margin = round(short_position + short_position * self.reg_T_margin_percent, 2)
        required_cash = max(required_margin - long_position, 0)
        position.margin = max(position.margin, required_cash)

    def close_position(self, position: Position, close_date: datetime, price_a: float, price_b: float) -> PairTransaction:
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
            transaction = PairTransaction(pair=position.pair_str,
                                          open_date=position.open_date,
                                          close_date=close_date,
                                          total_profit=total_profit,
                                          pair_return=total_return,
                                          margin=int(position.margin))
        return transaction


    def process_day_transactions(self, transaction_l: List[PairTransaction], holdings: int) -> int:
        cur_holdings = holdings
        if len(transaction_l) > 0:
            day_profit_loss_l = list(trans.total_profit for trans in transaction_l)
            profit_loss_total = sum(day_profit_loss_l)
            cur_holdings = cur_holdings + profit_loss_total
            day_trans_df = pd.DataFrame(trans.__dict__ for trans in transaction_l)
            self.all_transactions = pd.concat([self.all_transactions, day_trans_df], axis=0)
        return cur_holdings

    def close_open_positions(self, day_close_df: pd.DataFrame, holdings: int) -> int:
        pass

    def manage_position(self, day_stats: DailyStats, current_date: datetime, daily_transactions: List[PairTransaction]) -> None:
        assert day_stats.key in self.open_positions
        position = self.open_positions[day_stats.key]
        self.update_margin(position, day_stats, current_date)
        transaction = None
        # spread = A - intercept - (W * B)
        # SHORT_A_LONG_B: spread >= mean + delta * stddev : close when spread <= mean
        # LONG_A_SHORT_B: spread <= mean + delta * stddev : close when spread >= mean
        if position.position_type == OpenPosition.SHORT_A_LONG_B and day_stats.day_spread <= day_stats.mean:
            transaction = self.close_position(position, current_date, day_stats.stock_a, day_stats.stock_b)
        elif position.position_type == OpenPosition.LONG_A_SHORT_B and day_stats.day_spread >= day_stats.mean:
            transaction = self.close_position(position, current_date, day_stats.stock_a, day_stats.stock_b)
        if transaction is not None:
            daily_transactions.append(transaction)
            del self.open_positions[day_stats.key]

    def open_position(self, day_stats: DailyStats, current_date: datetime, stock_budget: int) -> None:
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
                                budget=stock_budget,
                                position_type=position_type)
            if position.shares_a == 0 or position.shares_b == 0:
                position_type = OpenPosition.SHARE_PRICE_OUT_OF_BUDGET
        if position_type == OpenPosition.SHORT_A_LONG_B or position_type == OpenPosition.LONG_A_SHORT_B:
            self.open_positions[day_stats.key] = position

    def calc_pair_budget(self, holdings: int) -> int:
        # Holdings of 100,000. A short requires 150 percent in margin. This is the proceeds of the short plus 50 percent.
        # So if we short 200,000 of stock we get 200,000 from the short proceeds which is used to open a long
        # position of 200,000.  In addition, we need 100,000 for the 50 percent margin.
        #
        # To be conservative we short 160,000 of stock and open a long position of 160,000. This requires a margin of
        # 80,000 but we allocate 100,000
        trade_capital = (2 * holdings) * 0.8
        # required margin would be trade_capital * 0.5 or 80,000
        stock_budget = int(trade_capital // self.num_pairs)
        return stock_budget

    def out_of_sample_test(self, start_ix: int, out_of_sample_df: pd.DataFrame, pairs_list: List[CointData], holdings: int) -> int:
        out_of_sample_index = out_of_sample_df.index
        end_ix = out_of_sample_df.shape[0]
        out_of_sample_day = pd.DataFrame()
        for row_ix in range(start_ix,end_ix):
            pair_budget = self.calc_pair_budget(holdings)
            daily_transactions: List[PairTransaction] = list()
            out_of_sample_back = out_of_sample_df.iloc[row_ix - self.back_window:row_ix]
            out_of_sample_day = out_of_sample_df.iloc[row_ix]
            row_date = pd.to_datetime(out_of_sample_index[row_ix])
            for pair in pairs_list:
                back_win_stock_a = out_of_sample_back[pair.stock_a]
                back_win_stock_b = out_of_sample_back[pair.stock_b]
                stock_a_day = out_of_sample_day[pair.stock_a]
                stock_b_day = out_of_sample_day[pair.stock_b]
                day_stats: HistoricalBacktest.DailyStats = self.out_of_sample_stats(pair=pair,
                                                                                    back_win_stock_a=back_win_stock_a,
                                                                                    back_win_stock_b=back_win_stock_b,
                                                                                    stock_a_day=stock_a_day,
                                                                                    stock_b_day=stock_b_day)
                if pair.key in self.open_positions:
                    self.manage_position(day_stats, current_date=row_date, daily_transactions=daily_transactions)
                else: # Possibly open a position depending on the spread
                    self.open_position(day_stats, current_date=row_date, stock_budget=pair_budget)
            holdings = self.process_day_transactions(daily_transactions, holdings)
        if len(self.open_positions) > 0:
            holdings = self.close_open_positions(day_close_df=out_of_sample_day, holdings=holdings)
        return holdings


    def historical_backtest(self,
                            close_prices_df: pd.DataFrame,
                            start_date: datetime,
                            delta: float):
        date_index = close_prices_df.index
        start_ix = find_date_index.findDateIndex(date_index, start_date)
        assert start_ix >= 0
        end_ix = close_prices_df.shape[0] # number of rows in close_prices_df
        print(f'index range: {start_ix} - end_ix: {end_ix}')
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
            print(f'out-of-sample: {out_of_sample_start}:{out_of_sample_end} {out_of_sample_date_start}:{out_of_sample_date_end}')
            in_sample_close_df = pd.DataFrame(close_prices_df.iloc[ix:in_sample_end_ix])
            in_sample_pairs_obj = InSamplePairs(in_sample_close_df=in_sample_close_df, corr_cutoff=self.corr_cutoff)
            selected_pairs: List[CointData] = in_sample_pairs_obj.get_in_sample_pairs(pairs_list=self.pairs_list)
            out_of_sample_df = pd.DataFrame(close_prices_df.iloc[out_of_sample_start:out_of_sample_end])
            holdings = self.out_of_sample_test(start_ix=self.back_window,
                                               out_of_sample_df=out_of_sample_df,
                                               pairs_list=selected_pairs,
                                               holdings=holdings)
            pass


holdings = 100000
num_pairs = 100
delta = 1.0

# Holdings of 100,000. A short requires 150 percent in margin. This is the proceeds of the short plus 50 percent.
# So if we short 200,000 of stock we get 200,000 from the short proceeds which is used to open a long
# position of 200,000.  In addition, we need 100,000 for the 50 percent margin.
#
# To be conservative we short 160,000 of stock and open a long position of 160,000. This requires a margin of
# 80,000 but we allocate 100,000
trade_capital = (2 * holdings) * 0.8
# required margin would be trade_capital * 0.5 or 80,000
num_pairs = 100
stock_budget = int(trade_capital // num_pairs)


historical_backtest = HistoricalBacktest(pairs_list=pairs_list,
                                         initial_holdings=holdings,
                                         num_pairs=num_pairs,
                                         corr_cutoff=corr_cutoff,
                                         in_sample_days=half_year,
                                         out_of_sample_days=quarter,
                                         back_window=quarter//3,
                                         delta=delta)
historical_backtest.historical_backtest(close_prices_df=close_prices_df, start_date=start_date, delta=delta)
pass

class OutOfSampleBacktest_v1:
    initial_margin_percent = 0.50
    reg_T_margin_percent = 0.25

    def __init__(self) -> None:
        pass

    def update_margin(self, position: Position, price_a: float, price_b: float) -> None:
        """
        Interactive Brokers adjusts the margin as the price changes. This means that as the price
        changes, additional margin may be required. This function calculates the maximum margin amount
        for the position.
        :param position: The Postion object
        :param price_a: The current price for stock A
        :param price_b: The current price for stock B
        :return: None
        """
        # Short A, Long B
        short_shares = position.shares_a
        long_shares = position.shares_b
        short_price = price_a
        long_price = price_b
        if position.position_type == OpenPosition.LONG_A_SHORT_B:
            # Short B, Long A
            short_shares = position.shares_b
            long_shares = position.shares_a
            short_price = price_b
            long_price = price_a
        short_position = short_shares * short_price
        long_position = long_shares * long_price
        required_margin = round(short_position + short_position * self.reg_T_margin_percent, 2)
        required_cash = max(required_margin - long_position, 0)
        position.margin = max(position.margin, required_cash)

    def close_position(self, position: Position, close_date: datetime, price_a: float, price_b: float, open_cash: float) -> PairTransaction:
        """
        A short position has a positive return when the position close price is less than the open price.
        A long position has a positive return when the position close is greater than the open price.
        :param position: A Position object
        :param close_date: The date the position is closed
        :param price_a: The current price for stock A
        :param price_b: The current price for stock B
        :return: a PairTransaction object
        Summing returns: https://financetrain.com/how-to-annualize-monthly-returns-example
        """
        transaction = None
        if position.position_type == OpenPosition.LONG_A_SHORT_B or \
           position.position_type == OpenPosition.SHORT_A_LONG_B:
            # Short A, Long B
            long_shares = position.shares_b
            short_shares = position.shares_a
            long_position = position.cost_b
            short_position = position.cost_a
            close_long = long_shares * price_b
            close_short = short_shares * price_a
            if position.position_type == OpenPosition.LONG_A_SHORT_B:
                long_shares = position.shares_a
                short_shares = position.shares_b
                long_position = position.cost_a
                short_position = position.cost_b
                close_long = long_shares * price_a
                close_short = short_shares * price_b
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
            # total return
            #   ret_short = 0.02
            #   ret_long = 0.03
            #   total = ((1 + 0.02) * (1 + 0.03)) - 1 = 0.0506
            total_return = round(((1 + ret_short) * (1 + ret_long)) - 1, 4)
            transaction = PairTransaction(open_date=position.open_date,
                                          close_date=close_date,
                                          total_profit=total_profit,
                                          pair_return=total_return,
                                          margin=int(position.margin))
        return transaction

    def backtest_pair(self,
                      pair_info: CointData,
                      out_of_sample_close_df: pd.DataFrame,
                      start_ix: int,
                      window: int,
                      delta: float,
                      pair_budget: int) -> List[PairTransaction]:
        position_type: OpenPosition = OpenPosition.NOT_OPEN
        out_of_sample_index = out_of_sample_close_df.index
        available_cash = pair_budget
        position = None
        current_date = today()
        price_a = 0.0
        price_b = 0.0
        weight = pair_info.weight
        intercept = pair_info.intercept
        stock_a = pair_info.stock_a
        stock_b = pair_info.stock_b
        pair_close = out_of_sample_close_df[[stock_a, stock_b]]
        pair_transaction_l = list()
        num_rows = pair_close.shape[0]
        for row_ix in range(start_ix,num_rows):
            transaction = None
            pair_close_back = pair_close.iloc[row_ix-window:row_ix]
            price_a_back = pair_close_back[stock_a]
            price_b_back = pair_close_back[stock_b]
            back_spread = price_a_back.values - intercept - (weight * price_b_back.values)
            mean = np.mean(back_spread).round(2)
            stddev = np.std(back_spread).round(2)
            day_close = pair_close.iloc[row_ix]
            current_date = pd.to_datetime(out_of_sample_index[row_ix])
            price_a = day_close[stock_a]
            price_b = day_close[stock_b]
            day_spread = price_a - intercept - (weight * price_b)
            if position_type == OpenPosition.NOT_OPEN:
                if day_spread >= mean + (delta * stddev):
                    position_type = OpenPosition.SHORT_A_LONG_B
                elif day_spread <= mean - (delta * stddev):
                    position_type = OpenPosition.LONG_A_SHORT_B
                if position_type == OpenPosition.SHORT_A_LONG_B or position_type == OpenPosition.LONG_A_SHORT_B:
                    position = self.Position(open_date=current_date, price_a=price_a, price_b=price_b, budget=available_cash,
                                             position_type=position_type)
                    if position.shares_a == 0 or position.shares_b == 0:
                        position_type = OpenPosition.SHARE_PRICE_OUT_OF_BUDGET
                        position.position_type = OpenPosition.SHARE_PRICE_OUT_OF_BUDGET
                        position = None
            elif position_type == OpenPosition.SHORT_A_LONG_B or position_type == OpenPosition.LONG_A_SHORT_B:
                self.update_margin(position, price_a, price_b)
                if position_type == OpenPosition.SHORT_A_LONG_B and day_spread <= mean:
                    transaction = self.close_position(position, current_date, price_a, price_b, available_cash)
                elif position_type == OpenPosition.LONG_A_SHORT_B and day_spread >= mean:
                    transaction = self.close_position(position, current_date, price_a, price_b, available_cash)
                if transaction is not None:
                    pair_transaction_l.append(transaction)
                    position_type = OpenPosition.NOT_OPEN
                    transaction = None
                    position = None
        # if there is an open position, then close the position at the end of the time period.
        if position is not None:
            transaction = self.close_position(position, current_date, price_a, price_b, available_cash)
            pair_transaction_l.append(transaction)
        return pair_transaction_l



# out_of_sample_start = in_sample_end
out_of_sample_win_start = out_of_sample_start - window
out_of_sample_end = out_of_sample_start + quarter
out_of_sample_df = close_prices_df.iloc[out_of_sample_win_start:out_of_sample_end]

out_of_sample_test = OutOfSampleBacktest_v1()

# https://www.wallstreetmojo.com/portfolio-return-formula/

all_transactions: List[List[PairTransaction]] = list()
for cur_pair in coint_list:
    pair_transactions: List[PairTransaction] = out_of_sample_test.backtest_pair(pair_info=cur_pair,
                                                                                out_of_sample_close_df=out_of_sample_df,
                                                                                start_ix=window,
                                                                                window=window,
                                                                                delta=stddev_delta,
                                                                                pair_budget=stock_budget)
    all_transactions.append(pair_transactions)

quarter_start_date = close_prices_df.index[out_of_sample_start]
quarter_end_date = close_prices_df.index[out_of_sample_end]

trades_per_pair = list(len(pair_trans) for pair_trans in all_transactions)
plt.hist(trades_per_pair, bins='auto')
plt.title(f'Trades per Pair {quarter_start_date} - {quarter_end_date}')
plt.show()

abs_profit_loss = 0
quarter_return = 0.0
w = 1.0/100.0
for trans_list in all_transactions:
    pair_return = 1.0
    for pair_trans in trans_list:
        pair_return = pair_return * (1 + pair_trans.pair_return)
        abs_profit_loss += pair_trans.total_profit
    pair_return = pair_return - 1
    quarter_return = quarter_return + (w * pair_return)

print(f'2nd quarter 2007 return: {round(100 * quarter_return, 2)}')

quarter_profit_percent = abs_profit_loss/holdings
year_profit_percent = ((1 + quarter_profit_percent) ** 4) - 1
print(f'Profit: {round(abs_profit_loss,0)} = {round(100*(quarter_profit_percent), 2)} percent per quarter, {round(100*year_profit_percent,2)} percent per year')

pass

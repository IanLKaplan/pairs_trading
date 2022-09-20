import os
from datetime import datetime
from datetime import timedelta
from typing import List, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from numpy import log
from pandas import DatetimeIndex
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate

from multiprocessing import Pool

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

s_and_p_file = 's_and_p_sector_components/sp_stocks.csv'
s_and_p_data = 's_and_p_data'
start_date_str = '2007-01-03'
start_date: datetime = datetime.fromisoformat(start_date_str)

trading_days = 252


def convert_date(some_date):
    if type(some_date) == str:
        some_date = datetime.fromisoformat(some_date)
    elif type(some_date) == np.datetime64:
        ts = (some_date - np.datetime64('1970-01-01T00:00')) / np.timedelta64(1, 's')
        some_date = datetime.utcfromtimestamp(ts)
    return some_date


def findDateIndex(date_index: DatetimeIndex, search_date: datetime) -> int:
    '''
    In a DatetimeIndex, find the index of the date that is nearest to search_date.
    This date will either be equal to search_date or the next date that is less than
    search_date
    '''
    index: int = -1
    i = 0
    search_date = convert_date(search_date)
    date_t = datetime.today()
    for i in range(0, len(date_index)):
        date_t = convert_date(date_index[i])
        if date_t >= search_date:
            break
    if date_t > search_date:
        index = i - 1
    else:
        index = i
    return index


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


class MarketData:
    """
    This class supports retrieving and storing stock market close data from Yahoo.
    """

    def __init__(self, start_date: datetime, path: str, update_data: bool = True):
        self.start_date = start_date
        self.path = path
        self.end_date: datetime = datetime.today() - timedelta(days=1)
        self.update_data = update_data

    def get_market_data(self,
                        symbol: str,
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        data_col = 'Close'
        if type(symbol) == str:
            t = list()
            t.append(symbol)
            symbols = t
        panel_data = yf.download(tickers=symbol, start=start_date, end=end_date, progress=False)
        if panel_data.shape[0] > 0:
            close_data: pd.DataFrame = panel_data[data_col]
        else:
            close_data = pd.DataFrame()
        close_data = close_data.round(2)
        close_data_df = pd.DataFrame(close_data)
        return close_data_df

    def symbol_file_path(self, symbol: str) -> str:
        path: str = self.path + os.path.sep + symbol.upper() + '.csv'
        return path

    def read_data(self, symbol: str) -> pd.DataFrame:
        file_path = self.symbol_file_path(symbol)
        if os.access(file_path, os.R_OK):
            symbol_df = pd.read_csv(file_path, index_col='Date')
            if self.update_data:
                last_row = symbol_df.tail(1)
                last_date = convert_date(last_row.index[0])
                if last_date.date() < self.end_date.date():
                    sym_start_date = last_date + timedelta(days=1)
                    new_data_df = self.get_market_data(symbol, sym_start_date, datetime.today())
                    if new_data_df.shape[0] > 0:
                        symbol_df = pd.concat([symbol_df, new_data_df], axis=0)
                        ix = symbol_df.index
                        ix = pd.to_datetime(ix)
                        symbol_df.index = ix
                        symbol_df.to_csv(file_path)
        else:
            symbol_df = self.get_market_data(symbol, self.start_date, self.end_date)
            if symbol_df.shape[0] > 0:
                if not os.access(self.path, os.R_OK):
                    os.mkdir(self.path)
                symbol_df.to_csv(file_path)
                if type(symbol_df) != pd.DataFrame:
                    symbol_df = pd.DataFrame(symbol_df)
        if symbol_df.shape[0] > 0:
            symbol_df.columns = [symbol]
        return symbol_df

    def get_close_data(self, stock_list: list) -> pd.DataFrame:
        # fetch the close data in parallel
        with Pool() as mp_pool:
            close_list = mp_pool.map(market_data.read_data, stock_l)
        close_df = pd.DataFrame()
        for close_data in close_list:
            stock_start_date = convert_date(close_data.head(1).index[0])
            if stock_start_date.date() == self.start_date.date():
                close_df = pd.concat([close_df, close_data], axis=1)
        return close_df


stock_info_df = read_s_and_p_stock_info(s_and_p_file)
market_data = MarketData(start_date=start_date, path=s_and_p_data, update_data=True)
stock_l: list = list(set(stock_info_df['Symbol']))
stock_l.sort()
close_prices_df = market_data.get_close_data(stock_l)
final_stock_list = list(close_prices_df.columns)
mask = stock_info_df['Symbol'].isin(final_stock_list)
final_stock_info_df = stock_info_df[mask]

sectors = extract_sectors(final_stock_info_df)
pairs_info_df = calc_pair_counts(sectors)

print(tabulate(pairs_info_df, headers=[*pairs_info_df.columns], tablefmt='fancy_grid'))


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


def window_correlation(stock_close_df: pd.DataFrame, pairs_list: List[Tuple], window: int, cutoff: float,
                       start_ix: int) -> int:
    count = 0
    for pair in pairs_list:
        stock_a = pair[0]
        stock_b = pair[1]
        price_a = stock_close_df[stock_a].iloc[start_ix:start_ix + window]
        price_b = stock_close_df[stock_b].iloc[start_ix:start_ix + window]
        log_price_a = log(price_a)
        log_price_b = log(price_b)
        c = np.corrcoef(log_price_a, log_price_b)
        if c[0, 1] >= cutoff:
            count = count + 1
    return count


class WindowedCorrelationDist:

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
        """
        For each windowed time period, calculate the number of pairs with a correlation greater than or equal to cutoff
        :param stock_close_df:
        :param pairs_list:
        :param window:
        :return: a Series where the data index is the date for the start of the period and the value is the number of stocks
        """

        start_list = [ix for ix in range(0, self.stock_close_df.shape[0], self.window)]
        with Pool() as mp_pool:
            count_l = mp_pool.map(self.window_correlation, start_list)
        dist_s = pd.Series(count_l)
        index = self.stock_close_df.index
        dist_s.index = index[start_list]
        return dist_s


def calc_windowed_correlation(stock_close_df: pd.DataFrame, pairs_list: List[Tuple], window: int) -> np.array:
    """
    Calculate the windowed pair correlation over the entire time period, for all the pairs.

    :param stock_close_df: A data frame containing the stock close prices. The columns are the stock tickers.
    :param pairs_list: A list of the pairs formed from the S&P 500 sectors.
    :return: A numpy array with the correlcations.
    """
    window = int(window)
    all_cor_v = np.zeros(0)
    for pair in pairs_list:
        cor_s: pd.Series = calc_pair_correlation(stock_close_df, pair, window)
        cor_v = np.array(cor_s.values)
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


def plot_ts(data_s: pd.Series, title: str, x_label: str, y_label: str) -> None:
    fix, ax = plt.subplots(figsize=(10, 8))
    ax.grid(True)
    plt.title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    data_s.plot()
    ax.axhline(y=0, color='black')
    plt.show()


lookback_window = int(trading_days / 2)
apple_tuple: Tuple = ('AAPL', 'MPWR')
apple_tuple_cor_s = calc_pair_correlation(stock_close_df=close_prices_df, pair=apple_tuple, window=lookback_window)
plot_ts(data_s=apple_tuple_cor_s, title=f'correlation between {apple_tuple[0]} and {apple_tuple[1]}',
        x_label='Window Start Date', y_label=f'Correlation over {lookback_window} day window')

pairs_list = get_pairs(sectors)

correlation_cutoff = 0.75
cor_dist_obj = WindowedCorrelationDist(stock_close_df=close_prices_df, pairs_list=pairs_list, window=lookback_window,
                                       cutoff=correlation_cutoff)
cor_dist = cor_dist_obj.calc_correlation_dist()
plot_ts(data_s=cor_dist, title=f"Pair correlation >= {correlation_cutoff}, by time period", x_label='Correlation',
        y_label='Count')

cor_a = calc_windowed_correlation(close_prices_df, pairs_list, lookback_window)

display_histogram(cor_a, 'Correlation between pairs', 'Count')

period_start_date_str = '2008-01-03'
period_start_date: datetime = datetime.fromisoformat(period_start_date_str)
start_ix = findDateIndex(close_prices_df.index, period_start_date)
pair_df = close_prices_df[['AAPL', 'MPWR']].iloc[start_ix:start_ix + lookback_window]
aapl_s = pair_df['AAPL']
mpwr_s = pair_df['MPWR']
log_aapl_s = log(aapl_s)
log_mpwr_s = log(mpwr_s)
c = np.corrcoef(log_aapl_s, log_mpwr_s)
pair_cor = round(c[0, 1], 2)
log_pair_df = pd.concat([log_aapl_s, log_mpwr_s], axis=1)

# https://seaborn.pydata.org/tutorial/regression.html
s = sns.regplot(x=log_pair_df.columns[0], y=log_pair_df.columns[1], data=log_pair_df, scatter_kws={"color": "blue"},
                line_kws={"color": "red"});
s.figure.set_size_inches(10, 6)
s.set(title=f'Correlation {pair_cor}')
plt.show()

def simple_return(time_series: np.array, period: int = 1) -> List:
    return list(((time_series[i] / time_series[i - period]) - 1.0 for i in range(period, len(time_series), period)))


def return_df(time_series_df: pd.DataFrame) -> pd.DataFrame:
    r_df: pd.DataFrame = pd.DataFrame()
    time_series_a: np.array = time_series_df.values
    return_l = simple_return(time_series_a, 1)
    r_df = pd.DataFrame(return_l)
    date_index = time_series_df.index
    r_df.index = date_index[1:len(date_index)]
    r_df.columns = time_series_df.columns
    return r_df

def apply_return(start_val: float, return_df: pd.DataFrame) -> np.array:
    port_a: np.array = np.zeros(return_df.shape[0] + 1)
    port_a[0] = start_val
    return_a = return_df.values
    for i in range(1, len(port_a)):
        port_a[i] = port_a[i - 1] + port_a[i - 1] * return_a[i - 1]
    return port_a

aapl_df = pd.DataFrame(aapl_s)
mpwr_df = pd.DataFrame(mpwr_s)
ret_aapl = return_df(aapl_df)
ret_mpwr = return_df(mpwr_df)
adj_aapl = apply_return(0, ret_aapl)
adj_mpwr = apply_return(0, ret_mpwr)

plot_df = pd.concat([adj_aapl, adj_mpwr], axis=1)
plot_df.plot(grid=True, title=f'AAPL/MPWR', figsize=(10, 6))


class PairStats:
    """
    A container for information about a pair
    """

    def __init__(self,
                 stock_a: str,
                 stock_b: str,
                 cor_v: float,
                 slope: float,
                 intercept: float,
                 residuals: pd.Series,
                 adf_stat: float,
                 p_value: float,
                 critical_vals: dict):
        self.stock_a = stock_a
        self.stock_b = stock_b
        self.cor_v = cor_v
        self.slope = slope
        self.intercept = intercept
        self.residuals = residuals
        self.adf_stat = adf_stat
        self.p_value = p_value
        self.critical_vals = critical_vals

    def __str__(self):
        s1: str = f'({self.stock_a},{self.stock_b}) cor_v: {self.cor_v} slope: {self.slope} intercept: {self.intercept}'
        s2: str = f'adf: {self.adf_stat} p-value: {self.p_value}'
        s3: str = f'1%: {self.critical_vals["1%"]}, 5%: {self.critical_vals["5%"]}, 10%: {self.critical_vals["10%"]}'
        s = s1 + '\n' + s2 + '\n' + s3
        return s


class PairsSelection:
    decimals = 4

    def __init__(self,
                 close_prices: pd.DataFrame,
                 correlation_cutoff: float):
        self.correlation_cutoff = correlation_cutoff
        self.close_prices = close_prices

    def pairs_correlation(self,
                          start_ix: int,
                          end_ix: int,
                          pairs_list: List[Tuple]) -> List[Tuple]:

        """
        Find the pairs with a log(price) correlation greater than or equal to cutoff within a close price window
        from start_ix to end_ix
        :param start_ix: the start index in stock_close_df
        :param end_ix: the end index in stock_close_df
        :param pairs_list: a list of pairs
        :return: A list of tuples consisting of tuple[stock_a, stock_b, sector, correlation]
        """
        selected_pairs_l = list()
        for pair in pairs_list:
            stock_a: str = pair[0]
            stock_b: str = pair[1]
            log_close_a = log(self.close_prices[stock_a][start_ix:end_ix + 1])
            log_close_b = log(self.close_prices[stock_b][start_ix:end_ix + 1])
            c = np.corrcoef(log_close_a, log_close_b)
            cor_v = round(c[0, 1], 2)
            if cor_v >= self.correlation_cutoff:
                sector = pair[2]
                selected_pairs_l.append((stock_a, stock_b, sector, cor_v))
        return selected_pairs_l

    def stationary_analysis(self, start_ix: int, end_ix: int, pair: Tuple) -> PairStats:
        stock_a: str = pair[0]
        stock_b: str = pair[1]
        cor_v: float = pair[3]
        log_close_a = log(self.close_prices[stock_a][start_ix:end_ix])
        log_close_b = log(self.close_prices[stock_b][start_ix:end_ix])
        log_close_b_const = sm.add_constant(log_close_b)
        result_ab = sm.OLS(log_close_a, log_close_b_const).fit()
        log_close_a_const = sm.add_constant(log_close_a)
        result_ba = sm.OLS(log_close_b, log_close_a_const).fit()
        slope_ab = result_ab.params[stock_b]
        slope_ba = result_ba.params[stock_a]
        result = result_ab
        slope = slope_ab
        if slope_ab < slope_ba:
            result = result_ba
            slope = slope_ba
        intercept = round(result.params['const'], self.decimals)
        slope = round(slope, self.decimals)
        residuals = result.resid
        # References
        # https://machinelearningmastery.com/time-series-data-stationary-python/
        # https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing-Part-II/
        # p-value <= 0.05 stationary mean reverting TS
        # ADF more negative means a stronger mean reverting process
        adf_result: Tuple = adfuller(residuals)
        adf_stat = round(adf_result[0], self.decimals)
        p_value = round(adf_result[1], self.decimals)
        critical_vals = adf_result[4]
        pair_stats = PairStats(stock_a=stock_a,
                               stock_b=stock_b,
                               cor_v=cor_v,
                               slope=slope,
                               intercept=intercept,
                               residuals=residuals,
                               adf_stat=adf_stat,
                               p_value=p_value,
                               critical_vals=critical_vals)
        return pair_stats

    def select_pairs(self, start_ix: int, end_ix: int, pairs_list: List[Tuple], threshold: str) -> List[PairStats]:
        """

        :param start_ix: start index in close price DataFrame
        :param end_ix:  end index in close price DataFrame
        :param pairs_list: a list of tuples (stock_a, stock_b, sector)
        :param threshold: a string equal to '1%', '5%' or '10%'
        :return: pairs that are correlated and have regression residuals that show mean reversion to the threshold level
                 Only unique pairs of stocks are returned. If there are two canidate pairs that both have AAPL, for example,
                 only one pair will be returned so that a stock does not appear twice in the set of pairs.
        """
        selected_pairs = self.pairs_correlation(start_ix=start_ix, end_ix=end_ix, pairs_list=pairs_list)

        stock_set: Set = set()
        pair_stat_l: List[PairStats] = list()
        for pair in selected_pairs:
            stats_ = self.stationary_analysis(start_ix=start_ix, end_ix=end_ix, pair=pair)
            threshold_level = stats_.critical_vals[threshold]
            if stats_.adf_stat < threshold_level:
                stock_a: str = pair[0]
                stock_b: str = pair[1]
                if stock_a not in stock_set and stock_b not in stock_set:
                    stock_set.add(stock_a)
                    stock_set.add(stock_b)
                    pair_stat_l.append(stats_)
        return pair_stat_l


pairs_selection = PairsSelection(close_prices=close_prices_df, correlation_cutoff=correlation_cutoff)
stats_l = pairs_selection.select_pairs(start_ix=0, end_ix=lookback_window, pairs_list=pairs_list, threshold='1%')

print(
    f'Number of candidate pairs: {len(pairs_list)}, number of pairs after selection: {len(stats_l)}: {round((len(stats_l) / len(pairs_list)) * 100, 2)} percent')

cor_l = [stat.cor_v for stat in stats_l]
cor_a = np.array(cor_l)
display_histogram(cor_a, 'Pairs Correlation', 'Count')

res_s = pd.Series(stats_l[0].residuals)
plot_ts(data_s=res_s, title=f'linear regression residuals for {stats_l[0].stock_a} and {stats_l[0].stock_b}',
        x_label='Window Start Date', y_label='')
pass


def compute_halflife(prices: pd.Series, lookback_window: int) -> float:
    """
    Calculate the half-life of a mean reverting series where the series
    is a Ornstein–Uhlenbeck process
    https://quant.stackexchange.com/a/25119
    """
    prices_a = prices.values[-lookback_window:]
    prices_lag = prices_a[1:]
    prices_trunc = prices_a[0:-1]
    prices_diff = prices_trunc - prices_lag
    prices_lag_m = sm.add_constant(prices_lag)
    res = sm.OLS(prices_diff, prices_lag_m).fit()
    halflife = -log(2) / res.params[1]
    return halflife


halflife = compute_halflife(prices=close_prices_df['AAPL'], lookback_window=lookback_window)

# https://onriskandreturn.com/2019/10/28/pairs-trade-practice/
# A Bollinger band is formed with three lines.  The middle line is a moving average of a time series, which
# would be replaced with the price spread obtained by subtracting price of y with price of x times hedge
# ratio of the cointegration pair.   The lookback period of the moving average should be in the same scale
# as the half-life we calculated earlier.  The top and bottom lines represent the trading range of certain
# standard deviation above and below the middle line.   A trade is entered when price spread hits either the
# top or bottom Bollinger range and exited when cutting cross the middle of the range.

first_pair_df = close_prices_df[['AAPL', 'MPWR']]

# https://seaborn.pydata.org/tutorial/regression.html
sns.regplot(x=first_pair_df.columns[0], y=first_pair_df.columns[1], data=first_pair_df, scatter_kws={"color": "blue"},
            line_kws={"color": "red"});
plt.show()

pass

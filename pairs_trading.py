
import os
import pandas as pd
from tabulate import tabulate
from datetime import datetime
from typing import List
from pandas_datareader import data
from datetime import timedelta

s_and_p_file = 's_and_p_sector_components/sp_stocks.csv'
s_and_p_data = 's_and_p_data'
start_date_str = '2007-01-02'
start_date: datetime = datetime.fromisoformat(start_date_str)


def convert_date(some_date):
    if type(some_date) == str:
        some_date = datetime.fromisoformat(some_date)
    elif type(some_date) == np.datetime64:
        ts = (some_date - np.datetime64('1970-01-01T00:00')) / np.timedelta64(1, 's')
        some_date = datetime.utcfromtimestamp(ts)
    return some_date


def read_stock_data(path: str) -> pd.DataFrame:
    s_and_p_stocks = pd.DataFrame
    if os.access(path, os.R_OK):
        s_and_p_stocks = pd.read_csv(s_and_p_file)
    else:
        print(f'Could not read file {s_and_p_file}')
    return s_and_p_stocks


def extract_sectors(stocks_df: pd.DataFrame) -> dict:
    """
    Columns in the DataFrame are Symbol,Name,Sector
    :param stocks_df:
    :return:
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
    column_label = ['num stocks', 'num pairs']
    sectors = list(sector_info.keys())
    counts_l: list = list()
    n_l: list = list()
    for sector in sectors:
        n = len(sector_info[sector])
        n_l.append(n)
        count = sum(range(1, n-1))
        counts_l.append(count)
    info_df = pd.DataFrame(n_l)
    info_df = pd.concat([info_df, pd.DataFrame(counts_l)], axis=1)
    info_df.columns = column_label
    sum_pairs = sum(counts_l)
    blank_df = pd.DataFrame([' '])
    sum_df = pd.DataFrame([sum_pairs])
    row_df = pd.concat([blank_df, sum_df], axis=1)
    row_df.columns = column_label
    info_df = pd.concat([info_df, row_df], axis=0)
    sectors.append('Sum')
    info_df.index = sectors
    return info_df


stock_info_df = read_stock_data(s_and_p_file)
sectors = extract_sectors(stock_info_df)
info_df = calc_pair_counts(sectors)

class MarketData:
    def __init__(self, start_date: datetime, path: str):
        self.start_date = start_date
        self.path = path
        self.end_date: datetime = datetime.today() - timedelta(days=1)

    def get_market_data(self,
                        symbols: List,
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        data_source = 'yahoo'
        data_col = 'Close'
        if type(symbols) == str:
            t = list()
            t.append(symbols)
            symbols = t
        panel_data: pd.DataFrame = data.DataReader(symbols, data_source, start_date, end_date)
        close_data: pd.DataFrame = panel_data[data_col]
        assert len(close_data) > 0, f'Error reading data for {symbols}'
        return close_data

    def symbol_file_path(self, symbol: str) -> str:
        path: str = self.path + os.path.sep + symbol.upper() + '.csv'
        return path

    def read_data(self, symbol: str) -> pd.DataFrame:
        file_path = self.symbol_file_path(symbol)
        if os.access(file_path, os.R_OK):
            symbol_df = pd.read_csv(file_path, index_col='Date')
            last_row = symbol_df.tail(1)
            last_date = convert_date(last_row.index[0])
            if last_date < self.end_date:
                sym_start_date = last_date + timedelta(days=1)
                new_data_df = self.get_market_data(symbol, sym_start_date, self.end_date)
                symbol_df = pd.concat([symbol_df, new_data_df], axis=0)
                symbol_df.to_csv(file_path)
        else:
            symbol_df = self.get_market_data(symbol, self.start_date, self.end_date)
            if not os.access(self.path, os.R_OK):
                os.mkdir(self.path)
            symbol_df.to_csv(file_path)
        return symbol_df

    def get_close_data(self, stock_list: list) -> pd.DataFrame:
        close_df = pd.DataFrame()
        for stock in stock_list:
            stock_df = self.read_data(stock)
            close_df = pd.concat([close_df, stock_df], axis=1)
        return close_df


market_data = MarketData(start_date, s_and_p_data)
xon_df = market_data.get_close_data(['XOM'])
xon_df = market_data.get_close_data(['XOM'])

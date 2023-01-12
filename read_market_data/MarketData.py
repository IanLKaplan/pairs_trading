
import os
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from utils.convert_date import convert_date
from multiprocessing import Pool


class MarketData:
    """
    This class supports retrieving and storing stock market close data from Yahoo.
    """

    def __init__(self, start_date: datetime):
        self.start_date = convert_date(start_date)
        # self.end_date: datetime = convert_date(datetime.today() - timedelta(days=1))
        self.end_date: datetime = convert_date(datetime.today())
        self.path = 's_and_p_data'

    def get_market_data(self,
                        symbol: str,
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        data_col = 'Close'
        if type(symbol) == str:
            t = list()
            t.append(symbol)
        panel_data = yf.download(tickers=symbol, start=start_date, end=end_date, progress=False)
        if panel_data.shape[0] > 0:
            close_data: pd.DataFrame = panel_data[data_col]
        else:
            close_data = pd.DataFrame()
        if close_data.shape[0] > 0:
            close_data = close_data.round(2)
            close_data_df = pd.DataFrame(close_data)
            index = pd.to_datetime(close_data_df.index.strftime('%Y-%m-%d'))
            close_data_df.index = index
        else:
            close_data_df = pd.DataFrame()
        return close_data_df

    def symbol_file_path(self, symbol: str) -> str:
        path: str = self.path + os.path.sep + symbol.upper() + '.csv'
        return path

    def df_last_date(self, data_df: pd.DataFrame) -> datetime:
        last_row = data_df.tail(1)
        last_date = convert_date(last_row.index[0])
        return last_date

    def findDateIndexFromEnd(self, data_df: pd.DataFrame, search_date: datetime) -> int:
        found_index = -1
        search_date = convert_date(search_date)
        index = data_df.index
        for i in range(len(index)-1, -1, -1):
            ix_date = convert_date(index[i])
            if ix_date == search_date:
                found_index = i
                break
        return found_index


    def read_data(self, symbol: str) -> pd.DataFrame:
        changed = False
        file_path = self.symbol_file_path(symbol)
        # Check to see if the file exists
        symbol_df = pd.DataFrame()
        if os.access(file_path, os.R_OK):
            # The file exists, so read the CSV data
            symbol_df = pd.read_csv(file_path, index_col='Date')
        if symbol_df.shape[0] == 0:
            # Either the file contained no data or it didn't exist
            symbol_df = self.get_market_data(symbol, self.start_date, self.end_date)
        if symbol_df.shape[0] > 0:
            last_date = self.df_last_date(symbol_df)
            if last_date.date() < (self.end_date - timedelta(days=1)).date():
                sym_start_date = last_date - timedelta(weeks=1)
                new_data_df = self.get_market_data(symbol, sym_start_date, self.end_date)
                if new_data_df.shape[0] > 0:
                    new_last_date = self.df_last_date(new_data_df)
                    if new_last_date > last_date:
                        last_date_ix = self.findDateIndexFromEnd(new_data_df, last_date)
                        if last_date_ix+1 < new_data_df.shape[0]:
                            new_data_sec = new_data_df.iloc[last_date_ix+1:]
                            symbol_df = pd.concat([symbol_df, new_data_sec], axis=0)
                            ix = symbol_df.index
                            ix = pd.to_datetime(ix)
                            symbol_df.index = ix
                            changed = True
            if not os.access(self.path, os.R_OK):
                os.mkdir(self.path)
            if type(symbol_df) != pd.DataFrame:
                symbol_df = pd.DataFrame(symbol_df)
            if changed:
                symbol_df.to_csv(file_path)
            symbol_df.columns = [symbol]
        return symbol_df

    def get_close_data(self, stock_list: list) -> pd.DataFrame:
        # fetch the close data in parallel
        close_df = pd.DataFrame()
        assert len(stock_list) > 0
        with Pool() as mp_pool:
            close_list = mp_pool.map(self.read_data, stock_list)
        # close_list = list()
        # for sym in stock_list:
        #     sym_close_df = self.read_data(sym)
        #     close_list.append(sym_close_df)
        for close_data in close_list:
            close_df = pd.concat([close_df, close_data], axis=1)
        # The last row may be fetched from "today" and be NaN values. Remove this row
        last_row = close_df[-1:]
        if all(last_row.isna().all()):
            close_df = close_df[:-1]
        # drop the stocks with different start dates
        close_df = close_df.dropna(axis='columns')
        return close_df


def read_s_and_p_stock_info(path: str) -> pd.DataFrame:
    """
    Read a file containing the information on S&P 500 stocks (e.g., the symbol, company name and sector)
    :param path: the path to the file
    :return: a DataFrame with columns Symbol, Name and Sector
    """
    s_and_p_stocks = pd.DataFrame()
    if os.access(path, os.R_OK):
        # s_and_p_socks columns are Symbol, Name and Sector
        s_and_p_stocks = pd.read_csv(path, index_col=0)
        new_names = [sym.replace('.', '-') for sym in s_and_p_stocks['Symbol']]
        s_and_p_stocks['Symbol'] = new_names
    else:
        print(f'Could not read file {path}')
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

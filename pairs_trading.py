
import os
import pandas as pd
from tabulate import tabulate

s_and_p_file = 's_and_p_sector_components/sp_stocks.csv'


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
sectors_df = extract_sectors(stock_info_df)
info_df = calc_pair_counts(sectors_df)

print(tabulate(info_df, headers=[*info_df.columns], tablefmt='fancy_grid'))